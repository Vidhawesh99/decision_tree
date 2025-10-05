
import joblib
import pandas as pd
from flask import Flask, request, jsonify

app = Flask(__name__)

# Load the trained Decision Tree model
model_path = './models/decision_tree_model.pkl' # Adjust path if necessary
loaded_model = joblib.load(model_path)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        data = request.get_json(force=True)
        # Assuming the input data is a dictionary matching the feature names
        # Convert the input data to a pandas DataFrame
        input_df = pd.DataFrame([data])

        # Ensure the order of columns in input_df matches the training data
        # This is a crucial step for consistent predictions
        # We need the list of columns from the training data X_train or X
        # For simplicity, let's assume X (the full feature set before split) is available
        # If X is not available, you might need to save the column order during training
        # For this placeholder, we'll use a simplified approach.
        # In a real application, you'd need to handle missing columns and order carefully.

        # Placeholder for aligning columns - Replace with actual column alignment logic
        # Example: input_df = input_df.reindex(columns=training_columns, fill_value=0)

        # Make prediction
        prediction = loaded_model.predict(input_df)

        # The target variable 'Amul_Stocking_Intent_Ready' was boolean, so convert prediction
        prediction_result = bool(prediction[0])

        return jsonify({'prediction': prediction_result})

    except Exception as e:
        return jsonify({'error': str(e)})

if __name__ == '__main__':
    # To run locally:
    # Ensure you have Flask installed (`pip install Flask`)
    # Ensure you have joblib installed (`pip install joblib`)
    # Ensure you have pandas installed (`pip install pandas`)
    # Ensure the 'models' directory and 'decision_tree_model.pkl' exist in the same directory as app.py
    # Run this file: `python app.py`
    # Send a POST request to http://127.0.0.1:5000/predict with JSON data

    # Example JSON data structure (replace with actual feature values):
    # {
    #     "Girnar": 0,
    #     "Wagh_Bakri": 0,
    #     "Red_Label": 0,
    #     "Granuel_Beans": 0,
    #     "Other": 0,
    #     "Outlet_Type_MT": 1,
    #     "Location_Urban": 1,
    #     "Tea_Premix_Yes": 1,
    #     "Shelf_Space_Low": 0,
    #     "Shelf_Space_Medium": 1,
    #     "Shelf_Space_No Shelf": 0,
    #     "Stock_Availability_Sometimes": 0,
    #     "Stock_Availability_Unaware": 0,
    #     "Return_Policy_Awareness_Unaware": 0,
    #     "Margin_Low": 0,
    #     "Margin_Moderate": 1,
    #     "Reason_Less Shelf Space": 0,
    #     "Reason_Low Margin": 1,
    #     "Reason_Product Quality": 0,
    #     "Reason_Service issues": 0,
    #     "Reason_Others": 0
    # }

    # Debug mode is useful during development
    # app.run(debug=True)

    # For production, run with a production-ready WSGI server like Gunicorn or uWSGI
    # Example using Gunicorn: `pip install gunicorn` then `gunicorn -w 4 app:app`
    print("App running...")
    app.run(debug=False) # Set debug=True for development
