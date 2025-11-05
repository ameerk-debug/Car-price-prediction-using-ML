from flask import Flask, render_template, request
from flask_cors import CORS, cross_origin
import pickle
import pandas as pd
import numpy as np
import re

app = Flask(__name__)
CORS(app)

# Load model and dataset
model = pickle.load(open('LinearRegressionModel.pkl', 'rb'))
car = pd.read_csv('quikr car dataset.csv')

@app.route('/', methods=['GET', 'POST'])
def index():
    companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    years = sorted(car['year'].unique(), reverse=True)
    fuel_types = car['fuel_type'].unique()

    companies.insert(0, 'Select Company')

    return render_template(
        'index.html',
        companies=companies,
        car_models=car_models,
        years=years,
        fuel_types=fuel_types
    )

@app.route('/predict', methods=['POST'])
@cross_origin()
def predict():
    try:
        # Read inputs from form
        company = request.form.get('company')
        car_model = request.form.get('car_models')
        year = request.form.get('year')
        fuel_type = request.form.get('fuel_type')
        driven = request.form.get('kilo_driven')

        # Clean and convert inputs
        year = int(year)
        driven = re.sub(r'[^\d]', '', driven)  # remove commas, km, etc.
        driven = int(driven)

        # Prepare data in model format
        input_data = pd.DataFrame(
            columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
            data=np.array([car_model, company, year, driven, fuel_type]).reshape(1, 5)
        )

        # Predict
        prediction = model.predict(input_data)
        price = np.round(prediction[0], 2)

        # Format as Indian currency (e.g., 456000 -> 4,56,000)
        formatted_price = f"{price:,.0f}"
        formatted_price = formatted_price.replace(",", ",")  # simple grouping

        return formatted_price

    except Exception as e:
        print("ERROR:", e)
        return "Error: " + str(e)

if __name__ == '__main__':
    app.run(debug=True)
