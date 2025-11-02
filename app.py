from flask import Flask, request, render_template
import joblib
import numpy as np

app = Flask(__name__)
model = joblib.load("random_forest_model.pkl")

@app.route('/', methods=['GET'])
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Parse form data
        size = float(request.form['size'])
        total_sqft = float(request.form['total_sqft'])
        bath = float(request.form['bath'])
        balcony_count = int(request.form['balcony'])
        area_type = request.form['area_type']

        # One-hot encode area_type
        area_type_Built_up = 1 if area_type == "Built_up_Area" else 0
        area_type_Carpet = 1 if area_type == "Carpet_Area" else 0
        area_type_Plot = 1 if area_type == "Plot_Area" else 0
        area_type_Super_built_up = 1 if area_type == "Super_built_up_Area" else 0

        # One-hot encode balcony count
        balcony_0 = 1 if balcony_count == 0 else 0
        balcony_1 = 1 if balcony_count == 1 else 0
        balcony_2 = 1 if balcony_count == 2 else 0
        balcony_3 = 1 if balcony_count == 3 else 0

        # Derived features
        price_per_sqft = 0  # Default, can be improved by adding input or median
        bath_per_size = bath / size if size != 0 else 0

        # Prepare features array (14 features)
        features = np.array([
            size,
            total_sqft,
            bath,
            area_type_Built_up,
            area_type_Carpet,
            area_type_Plot,
            area_type_Super_built_up,
            balcony_0,
            balcony_1,
            balcony_2,
            balcony_3,
            balcony_count,
            price_per_sqft,
            bath_per_size
        ]).reshape(1, -1)

        # Predict using the model
        prediction = model.predict(features)[0]

        return render_template('index.html', prediction=round(prediction, 2))

    except Exception as e:
        return f"Error during prediction: {str(e)}"

if __name__ == '__main__':
    app.run(debug=True)
