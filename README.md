# Lung Cancer Risk Assessment Web Application

A modern web-based interface for lung cancer risk assessment using machine learning. This application provides a user-friendly form where patients can input their symptoms and lifestyle factors to get an instant risk assessment.

## Features

- 🎨 **Modern UI**: Beautiful, responsive design with gradient backgrounds and smooth animations
- 📱 **Mobile Friendly**: Works perfectly on desktop, tablet, and mobile devices
- ⚡ **Real-time Prediction**: Instant risk assessment using trained machine learning model
- 📊 **Visual Results**: Progress bars and color-coded risk levels
- 🔄 **Easy Reset**: Start new assessments with one click

## Prerequisites

- Python 3.7 or higher
- The trained model file: `lung_cancer_prediction_model.joblib`

## Installation

1. **Clone or download the project files**

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Ensure the model file exists**:
   - Make sure `lung_cancer_prediction_model.joblib` is in the project root directory
   - If you don't have it, run the training script first:
     ```bash
     python final.py
     ```

## Running the Application

1. **Start the Flask server**:
   ```bash
   python app.py
   ```

2. **Open your web browser** and navigate to:
   ```
   http://localhost:5001
   ```

3. **Fill out the form** with patient information:
   - Personal Information (Gender, Age)
   - Lifestyle Factors (Smoking, Alcohol, Peer Pressure)
   - Physical Symptoms (Yellow Fingers, Wheezing, Cough, etc.)
   - Health Conditions (Anxiety, Chronic Disease, Fatigue, Allergies)

4. **Click "Assess Risk"** to get instant results

## File Structure

```
├── app.py                          # Flask web application
├── final.py                        # Original training script
├── lung_cancer_prediction_model.joblib  # Trained model
├── requirements.txt                # Python dependencies
├── templates/
│   └── index.html                 # Web interface template
├── data/                          # Data files
│   ├── lung_cancer_survey.csv
│   ├── lung_cancer_final_v1.csv
│   └── ...
└── images/                        # Generated charts
```

## How It Works

1. **Data Collection**: The web form collects 15 different patient attributes
2. **Preprocessing**: Data is formatted to match the training dataset structure
3. **Prediction**: The trained Random Forest model analyzes the input
4. **Results**: Risk level (High/Low) and probability percentage are displayed

## Model Information

- **Algorithm**: Random Forest Classifier
- **Features**: 15 patient attributes including symptoms, lifestyle, and demographics
- **Output**: Binary classification (High/Low risk) with probability score
- **Training Data**: Lung cancer survey dataset with medical indicators

## API Endpoints

- `GET /`: Main web interface
- `POST /predict`: Prediction endpoint (accepts JSON data)

## Example API Usage

```python
import requests

data = {
    "gender": "M",
    "age": 45,
    "smoking": 1,
    "yellow_fingers": 0,
    "anxiety": 1,
    "peer_pressure": 0,
    "chronic_disease": 0,
    "fatigue": 1,
    "allergy": 0,
    "wheezing": 1,
    "alcohol_consuming": 0,
    "coughing": 1,
    "shortness_of_breath": 0,
    "swallowing_difficulty": 0,
    "chest_pain": 0
}

response = requests.post('http://localhost:5000/predict', json=data)
result = response.json()
print(f"Risk Level: {result['risk_level']}")
print(f"Probability: {result['probability']:.1%}")
```

## Security Notes

- This is a demonstration application
- Always consult healthcare professionals for medical decisions
- The model is for educational/research purposes only
- Patient data is not stored or logged

## Troubleshooting

**Model not found error**:
- Ensure `lung_cancer_prediction_model.joblib` exists in the project root
- Run `python final.py` to generate the model if needed

**Port already in use**:
- Change the port in `app.py`: `app.run(debug=True, host='0.0.0.0', port=5001)`

**Dependencies issues**:
- Create a virtual environment: `python -m venv venv`
- Activate it: `source venv/bin/activate` (Linux/Mac) or `venv\Scripts\activate` (Windows)
- Install dependencies: `pip install -r requirements.txt`

## Contributing

Feel free to enhance the application with additional features like:
- Patient data storage (with proper security)
- Multiple model comparison
- Detailed risk factor explanations
- Export functionality for medical records 