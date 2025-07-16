# Lung Cancer Prediction Model 🫁

A machine learning project that predicts lung cancer risk using patient survey data. Includes data preprocessing, model training, evaluation, and model serialization.

---

## 🚀 Table of Contents

- [Project Overview](#project-overview)  
- [Dataset](#dataset)  
- [Prerequisites](#prerequisites)  
- [Usage](#usage)  
- [Model Evaluation](#model-evaluation)  
- [Model Persistence](#model-persistence)  
- [License & Disclaimer](#license--disclaimer)

---

## 📌 Project Overview

This project builds a classifier (e.g., Logistic Regression) to predict lung cancer presence. You’ll go through:

1. Loading and cleaning the raw dataset  
2. Encoding categorical and binary features  
3. Splitting into training/testing sets  
4. Scaling features  
5. Training multiple models and evaluating their performance  
6. Serializing the best-performing model via `joblib`

---

## 📂 Dataset

The data file `Lung Cancer Data.csv` contains features like:

- **AGE** – Patient’s age  
- **SMOKING**, **YELLOW_FINGERS**, **ANXIETY**, etc. (0 = No, 1 = Yes)  
- **LUNG_CANCER** – Target (0 = No, 1 = Yes)

This mirrors public datasets used for basic lung cancer prediction :contentReference[oaicite:1]{index=1}.

---

## 🛠️ Prerequisites

Ensure you have the following Python packages installed:

```bash
pip install pandas numpy scikit-learn joblib matplotlib
