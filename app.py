from flask import Flask, request, jsonify
import pandas as pd
from prophet import Prophet
import logging

app = Flask(__name__)

# Enable logging
logging.basicConfig(level=logging.INFO)

# Load dataset
data = pd.read_csv('synthetic_data.csv')
data['Date'] = pd.to_datetime(data['Date'])

# Create a dictionary for installed capacity lookup
installed_capacity = data[['District', 'Installed Capacity (MW)']].drop_duplicates().set_index('District').to_dict()['Installed Capacity (MW)']

# Train Prophet models for each district
models = {}

for district in data['District'].unique():
    district_data = data[data['District'] == district]

    # Load Demand Model
    load_model = Prophet()
    load_model.add_regressor('Price (₹/unit)')
    load_model.add_regressor('Installed Capacity (MW)')
    load_model.fit(district_data.rename(columns={'Date': 'ds', 'Load Demand (MW)': 'y'}))

    # Price Model
    price_model = Prophet()
    price_model.add_regressor('Load Demand (MW)')
    price_model.fit(district_data.rename(columns={'Date': 'ds', 'Price (₹/unit)': 'y'}))

    # Blackout Risk Model
    blackout_model = Prophet()
    blackout_model.add_regressor('Load Demand (MW)')
    blackout_model.add_regressor('Installed Capacity (MW)')
    blackout_model.fit(district_data.rename(columns={'Date': 'ds', 'Blackout Risk (%)': 'y'}))

    models[district] = {
        'load': load_model,
        'price': price_model,
        'blackout': blackout_model
    }

@app.route('/predict', methods=['GET'])
def predict():
    district = request.args.get('district')
    future_date = request.args.get('date')

    if district not in models:
        return jsonify({'error': 'Invalid district'}), 400

    if not future_date:
        return jsonify({'error': 'Missing future date'}), 400

    try:
        future_date = pd.to_datetime(future_date)

        # Prepare future dataframe
        future = pd.DataFrame({'ds': [future_date]})

        # Installed capacity for the district
        installed_capacity_value = installed_capacity.get(district, None)

        # Predict Load Demand
        future['Price (₹/unit)'] = data.loc[data['District'] == district, 'Price (₹/unit)'].mean()
        future['Installed Capacity (MW)'] = installed_capacity_value
        load_pred = models[district]['load'].predict(future)['yhat'].iloc[0]

        # Predict Price
        future['Load Demand (MW)'] = load_pred
        price_pred = models[district]['price'].predict(future)['yhat'].iloc[0]

        # Predict Blackout Risk
        future['Load Demand (MW)'] = load_pred
        future['Installed Capacity (MW)'] = installed_capacity_value
        blackout_pred = models[district]['blackout'].predict(future)['yhat'].iloc[0]

        response = {
            'District': district,
            'Date': future_date.strftime('%Y-%m-%d'),
            'Load Demand (MW)': round(load_pred, 2),
            'Price (₹/unit)': round(price_pred, 2),
            'Blackout Risk (%)': round(blackout_pred, 2),
            'Installed Capacity (MW)': installed_capacity_value
        }

        return jsonify(response)

    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return jsonify({'error': 'Prediction failed', 'message': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)

