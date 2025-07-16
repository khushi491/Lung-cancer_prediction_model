#!/usr/bin/env python3
"""
Test script for the Lung Cancer Risk Assessment API
"""

import requests
import json

def test_prediction_api():
    """Test the prediction endpoint with sample data"""
    
    # Sample patient data
    test_data = {
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
    
    try:
        # Make API request
        response = requests.post('http://localhost:5001/predict', json=test_data)
        
        if response.status_code == 200:
            result = response.json()
            print("✅ API Test Successful!")
            print(f"Risk Level: {result['risk_level']}")
            print(f"Probability: {result['probability']:.1%}")
            print(f"Prediction: {result['prediction']}")
            print(f"Message: {result['message']}")
        else:
            print(f"❌ API Test Failed with status code: {response.status_code}")
            print(f"Response: {response.text}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Make sure the Flask app is running on port 5001")
    except Exception as e:
        print(f"❌ Error: {e}")

def test_web_interface():
    """Test if the web interface is accessible"""
    
    try:
        response = requests.get('http://localhost:5001/')
        
        if response.status_code == 200:
            print("✅ Web Interface is accessible!")
            print("You can now open http://localhost:5001 in your browser")
        else:
            print(f"❌ Web Interface not accessible. Status code: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print("❌ Connection Error: Make sure the Flask app is running on port 5001")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("🧪 Testing Lung Cancer Risk Assessment API...")
    print("=" * 50)
    
    test_web_interface()
    print()
    test_prediction_api()
    
    print("\n" + "=" * 50)
    print("🎉 Testing complete!") 