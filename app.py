from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import os
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)

# Load the trained model
model_path = "./lung_cancer_prediction_model.joblib"
if os.path.exists(model_path):
    model = joblib.load(model_path)
else:
    print("Model file not found. Please run the training script first.")
    model = None

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = request.get_json()
        
        # Create patient data dictionary
        patient_data = {
            'GENDER': data['gender'],
            'AGE': int(data['age']),
            'SMOKING': int(data['smoking']),
            'YELLOW_FINGERS': int(data['yellow_fingers']),
            'ANXIETY': int(data['anxiety']),
            'PEER_PRESSURE': int(data['peer_pressure']),
            'CHRONIC_DISEASE': int(data['chronic_disease']),
            'FATIGUE': int(data['fatigue']),
            'ALLERGY': int(data['allergy']),
            'WHEEZING': int(data['wheezing']),
            'ALCOHOL_CONSUMING': int(data['alcohol_consuming']),
            'COUGHING': int(data['coughing']),
            'SHORTNESS_OF_BREATH': int(data['shortness_of_breath']),
            'SWALLOWING_DIFFICULTY': int(data['swallowing_difficulty']),
            'CHEST_PAIN': int(data['chest_pain'])
        }
        
        # Create DataFrame
        input_df = pd.DataFrame([patient_data])
        
        # Make prediction
        if model is not None:
            prediction = model.predict(input_df)[0]
            probability = model.predict_proba(input_df)[0][1]
            
            result = {
                'prediction': int(prediction),
                'probability': float(probability),
                'risk_level': 'HIGH' if prediction == 1 else 'LOW',
                'message': 'HIGH RISK of lung cancer detected.' if prediction == 1 else 'LOW RISK of lung cancer detected.'
            }
        else:
            result = {
                'error': 'Model not available. Please ensure the model file exists.'
            }
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5001) 