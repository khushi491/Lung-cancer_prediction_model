from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import pandas as pd
import joblib
import os
import warnings
import logging

# Suppress warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Load the trained model
model_path = "./lung_cancer_prediction_model.joblib"
model = None

try:
    if os.path.exists(model_path):
        model = joblib.load(model_path)
        logger.info("Model loaded successfully")
    else:
        logger.error("Model file not found. Please ensure the model file exists.")
except Exception as e:
    logger.error(f"Error loading model: {e}")

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/health')
def health():
    """Health check endpoint for Railway"""
    return jsonify({
        'status': 'healthy',
        'model_loaded': model is not None
    })

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get form data
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
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
        logger.error(f"Prediction error: {e}")
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Get port from environment variable with fallback
    port = int(os.environ.get('PORT', 5001))
    debug = os.environ.get('FLASK_ENV') == 'development'
    
    logger.info(f"Starting Flask app on port {port}")
    app.run(debug=debug, host='0.0.0.0', port=port) 