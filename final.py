"""
phase3_master.py
----------------
**Single‑file pipeline covering CRISP‑DM Phase 3 tasks
+ detailed exploratory charts (merged from project_1_eda.py).**

Tasks & Deliverables
--------------------
Task 3.1 : Selecting Data…… -> Data Rationale Report (printed log)
Task 3.2 : Cleaning Data……  -> Data Cleansing Report (printed log)
Task 3.3 : Constructing Data -> Data Attribute & Generation Reports (printed log + CSV)
Task 3.4 : Integrating Data… -> Merged Data Set (CSV)
Task 3.5 : Formatting Data…  -> Final Formatted Dataset (CSV)

Extras
------
• Saves all charts (class balance, age distribution, feature vs cancer barplots, correlation matrix, pairplot) in ./images/, 300 dpi.
• Verbose `log()` timestamps show progress.
• Requires: pandas, matplotlib, seaborn, numpy (all standard in most envs).
"""

import os, datetime, uuid, sys, importlib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend for script execution
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV
import joblib

# ---------------- Configuration ----------------
RAW_PATH    = "./data/lung_cancer_survey.csv"
DATA_DIR    = "./data"
IMAGE_DIR   = "./images"
SCORES_PATH = os.path.join(DATA_DIR, "lung_scores.csv")
MERGED_PATH = os.path.join(DATA_DIR, "lung_cancer_merged.csv")
FINAL_PATH  = os.path.join(DATA_DIR, "lung_cancer_final_v1.csv")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(IMAGE_DIR, exist_ok=True)

def log(msg):
    ts = datetime.datetime.now().strftime("%H:%M:%S")
    print(f"[{ts}] {msg}")

# ---------------- Task 3.1 – Selecting Data ----------------
log("Task 3.1  Selecting Data – loading raw survey.")
df_raw = pd.read_csv(RAW_PATH)
log(f"Raw dataset shape: {df_raw.shape}")
log("Data Rationale: entire survey retained; already domain‑filtered to potential lung‑cancer predictors.")

# ---------------- Task 3.2 – Cleaning Data ----------------
log("Task 3.2  Cleaning Data – duplicates, header normalisation, binary mapping.")
df = df_raw.drop_duplicates().reset_index(drop=True)
df.columns = [c.strip().upper().replace(' ', '_') for c in df.columns]
df.insert(0, "PATIENT_ID", df.index + 1)

# Map 1/2 codes -> 0/1 where appropriate
binary_cols = [c for c in df.select_dtypes("int").columns if c not in ("PATIENT_ID","AGE")]
for col in binary_cols:
    if set(df[col].unique()) <= {1,2}:
        df[col] = df[col].map({1:0, 2:1})

# Map YES/NO to 1/0 for all columns with those values
for col in df.columns:
    if set(df[col].dropna().unique()) <= {"YES", "NO"}:
        df[col] = df[col].map({"YES": 1, "NO": 0})

log(f"Binary columns mapped: {len(binary_cols)}; Post‑clean shape: {df.shape}")

# ---------------- Task 3.3 – Constructing Data ----------------
log("Task 3.3  Constructing Data – engineering composite scores.")
df['RESPIRATORY_SCORE'] = df['WHEEZING'] + df['SHORTNESS_OF_BREATH'] + df['COUGHING']
df['LIFESTYLE_SCORE']   = df['SMOKING'] + df['ALCOHOL_CONSUMING'] + df['PEER_PRESSURE']
df['SYMPTOM_SCORE']     = (
    df['YELLOW_FINGERS'] + df['ANXIETY'] + df['FATIGUE']
    + df['SWALLOWING_DIFFICULTY'] + df['CHEST_PAIN']
)
scores_df = df[['PATIENT_ID','RESPIRATORY_SCORE','LIFESTYLE_SCORE','SYMPTOM_SCORE']]
scores_df.to_csv(SCORES_PATH, index=False)
log(f"Scores CSV saved → {SCORES_PATH}")

# ---------------- Task 3.4 – Integrating Data ----------------
log("Task 3.4  Integrating Data – merging scores (already in df).")
merged_df = df.copy()
merged_df.to_csv(MERGED_PATH, index=False)
log(f"Merged dataset saved → {MERGED_PATH}")

# ---------------- Task 3.5 – Formatting Data ----------------
log("Task 3.5  Formatting Data – ordering & final dataset.")
ordered_cols = [
    "PATIENT_ID", "GENDER", "AGE",
    "SMOKING", "ALCOHOL_CONSUMING", "PEER_PRESSURE", "LIFESTYLE_SCORE",
    "WHEEZING", "SHORTNESS_OF_BREATH", "COUGHING", "RESPIRATORY_SCORE",
    "YELLOW_FINGERS", "ANXIETY", "FATIGUE", "SWALLOWING_DIFFICULTY", "CHEST_PAIN", "SYMPTOM_SCORE",
    "CHRONIC_DISEASE", "ALLERGY", "LUNG_CANCER"
]
ordered_cols = [c for c in ordered_cols if c in merged_df.columns]
final_df = merged_df[ordered_cols]
final_df.to_csv(FINAL_PATH, index=False)
log(f"Final formatted dataset saved → {FINAL_PATH}  (shape={final_df.shape})")

# ---------------- Exploratory Charts (project_1_eda equivalent) ----------------
log("Generating exploratory charts (class balance, age distribution, feature barplots, correlation, pairplot).")

# 1. Class Distribution
sns.countplot(x='LUNG_CANCER', data=final_df)
plt.title("Class Distribution: Lung Cancer")
plt.xlabel("Lung Cancer (1=Yes, 0=No)")
plt.ylabel("Count")
class_plot = os.path.join(IMAGE_DIR, "class_distribution.png")
plt.tight_layout()
plt.savefig(class_plot, dpi=300)
plt.close(); log(f"Saved {class_plot}")

# 2. Age Distribution
sns.histplot(final_df['AGE'], bins=20, kde=True)
plt.title("Age Distribution of Patients")
plt.xlabel("Age")
plt.ylabel("Count")
age_plot = os.path.join(IMAGE_DIR, "age_distribution.png")
plt.tight_layout()
plt.savefig(age_plot, dpi=300)
plt.close(); log(f"Saved {age_plot}")

# 3. Feature vs Lung Cancer Probability (labels and varied chart types)
binary_features = [c for c in binary_cols if c not in ['LUNG_CANCER']]
for i, col in enumerate(binary_features):
    plt.figure(figsize=(4,3))
    unique_vals = final_df[col].nunique()
    label_map = {0: "No", 1: "Yes"}
    if unique_vals == 2:
        # Barplot for binary features, with labels
        prob_df = final_df.groupby(col)['LUNG_CANCER'].mean().reset_index()
        prob_df[col] = prob_df[col].map(label_map)
        sns.barplot(x=col, y='LUNG_CANCER', data=prob_df)
        plt.title(f"{col} vs Lung Cancer Probability")
        plt.xlabel(f"{col} (No/Yes)")
        plt.ylabel("Probability")
    elif i % 2 == 0:
        # Boxplot for variety, with labels
        final_df[col + "_LABEL"] = final_df[col].map(label_map)
        sns.boxplot(x=col + "_LABEL", y='LUNG_CANCER', data=final_df)
        plt.title(f"{col} vs Lung Cancer Distribution")
        plt.xlabel(f"{col} (No/Yes)")
        plt.ylabel("Lung Cancer")
    else:
        # Violinplot for variety, with labels
        final_df[col + "_LABEL"] = final_df[col].map(label_map)
        sns.violinplot(x=col + "_LABEL", y='LUNG_CANCER', data=final_df, inner="quartile")
        plt.title(f"{col} vs Lung Cancer Distribution")
        plt.xlabel(f"{col} (No/Yes)")
        plt.ylabel("Lung Cancer")
    fname = os.path.join(IMAGE_DIR, f"{col.lower()}_vs_cancer.png")
    plt.tight_layout()
    plt.savefig(fname, dpi=300)
    plt.close()
    log(f"Saved {fname}")

# 4. Correlation Matrix
plt.figure(figsize=(10,8))
numeric_df = final_df.select_dtypes(include=[np.number])  # Only numeric columns
sns.heatmap(numeric_df.corr(), annot=False, cmap='coolwarm', fmt=".2f")
plt.title("Correlation Matrix")
corr_plot = os.path.join(IMAGE_DIR, "correlation_matrix.png")
plt.tight_layout()
plt.savefig(corr_plot, dpi=300)
plt.close(); log(f"Saved {corr_plot}")

# 5. Pairplot (selected cols)
selected_cols = [
    'AGE','SMOKING','YELLOW_FINGERS','ANXIETY','PEER_PRESSURE',
    'CHRONIC_DISEASE','FATIGUE','ALLERGY','WHEEZING',
    'ALCOHOL_CONSUMING','COUGHING','SHORTNESS_OF_BREATH',
    'SWALLOWING_DIFFICULTY','CHEST_PAIN','LUNG_CANCER'
]
sns.pairplot(final_df[selected_cols], hue='LUNG_CANCER', corner=True)
pair_plot = os.path.join(IMAGE_DIR, "pairplot_selected.png")
plt.savefig(pair_plot, dpi=300)
plt.close(); log(f"Saved {pair_plot}")

# 6. Histograms of Composite Scores
for col in ['RESPIRATORY_SCORE','LIFESTYLE_SCORE','SYMPTOM_SCORE']:
    plt.figure()
    plt.hist(final_df[col])
    plt.title(f"Distribution of {col}")
    plt.xlabel(col)
    plt.ylabel("Frequency")
    plt.tight_layout()
    fname = os.path.join(IMAGE_DIR, f"{col.lower()}_hist.png")
    plt.savefig(fname, dpi=300)
    plt.close()
    log(f"Saved {fname}")

log("All tasks and charts completed successfully")
# --- Step 1: Data Preparation ---

# Define features (X) and target (y)
X = df.drop(columns=['LUNG_CANCER', 'PATIENT_ID', 'RESPIRATORY_SCORE', 'LIFESTYLE_SCORE', 'SYMPTOM_SCORE'])
y = df['LUNG_CANCER']

# Identify categorical and numerical features for preprocessing
categorical_features = ['GENDER']
numerical_features = ['AGE']
# All other features are already binary (0/1) and don't need scaling
binary_features = [col for col in X.columns if col not in categorical_features + numerical_features]


# Create a preprocessing pipeline
# OneHotEncoder for 'GENDER', StandardScaler for 'AGE', and passthrough for binary features
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_features),
        ('cat', OneHotEncoder(), categorical_features),
        ('bin', 'passthrough', binary_features) # Keep binary columns as they are
    ],
    remainder='drop' # Drop any columns not specified
)

# Split data into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training set shape: {X_train.shape}")
print(f"Testing set shape: {X_test.shape}")

# --- Step 2: Model Training & Evaluation ---

# Create a pipeline with preprocessing and the Logistic Regression model
lr_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', LogisticRegression(random_state=42))])

# Train the model
lr_pipeline.fit(X_train, y_train)


# Make predictions on the test data
y_pred_lr = lr_pipeline.predict(X_test)

# Evaluate the model
print("--- Logistic Regression Evaluation ---")
print(classification_report(y_test, y_pred_lr))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_lr))
print("AUC Score:", roc_auc_score(y_test, lr_pipeline.predict_proba(X_test)[:, 1]))

# Now, do the same for a Random Forest Classifier
rf_pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                              ('classifier', RandomForestClassifier(random_state=42))])

rf_pipeline.fit(X_train, y_train)
y_pred_rf = rf_pipeline.predict(X_test)

print("\n--- Random Forest Evaluation ---")
print(classification_report(y_test, y_pred_rf))
print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred_rf))
print("AUC Score:", roc_auc_score(y_test, rf_pipeline.predict_proba(X_test)[:, 1]))

# --- Step 3: Hyperparameter Tuning ---

# Define a parameter grid to search through
# These are some common parameters to tune for a RandomForest
param_grid = {
    'classifier__n_estimators': [100, 200, 300],         # Number of trees in the forest
    'classifier__max_depth': [None, 10, 20],           # Maximum depth of the trees
    'classifier__min_samples_leaf': [1, 2, 4]          # Minimum samples required at a leaf node
}

# Note: 'classifier__' prefix is needed to tell the pipeline which step to apply the parameters to.

# Create the Grid Search object
grid_search = GridSearchCV(rf_pipeline, param_grid, cv=5, n_jobs=-1, scoring='roc_auc')

# Fit it to the training data
grid_search.fit(X_train, y_train)

# Print the best parameters found
print("\n--- Hyperparameter Tuning Results ---")
print("Best parameters found:", grid_search.best_params_)
print("Best cross-validation AUC score:", grid_search.best_score_)

# --- Step 4: Finalize and Save Model ---

# The grid_search object is already the best model, trained on the full training data
best_rf_model = grid_search.best_estimator_

# Evaluate this final, tuned model on the test set
y_pred_final = best_rf_model.predict(X_test)

print("\n--- Final Tuned Model Evaluation ---")
print(classification_report(y_test, y_pred_final))
print("Final AUC Score:", roc_auc_score(y_test, best_rf_model.predict_proba(X_test)[:, 1]))

# Save the final model pipeline to a file
model_path = "./lung_cancer_prediction_model.joblib"
joblib.dump(best_rf_model, model_path)

print(f"\n Model saved successfully to {model_path}")

# To load and use it later:
# loaded_model = joblib.load(model_path)
# new_prediction = loaded_model.predict(new_patient_data)

loaded_model = joblib.load(model_path)

import warnings

# Suppress a specific pandas warning that is not critical here
warnings.filterwarnings(
    "ignore",
    message="A value is trying to be set on a copy of a slice from a DataFrame.*",
    category=FutureWarning
)

def get_validated_input(prompt, validation_type):
    """A helper function to get and validate user input."""
    while True:
        user_input = input(prompt).lower().strip()
        if validation_type == 'yes_no':
            if user_input in ['yes', 'y']:
                return 1
            elif user_input in ['no', 'n']:
                return 0
            else:
                print("Invalid input. Please enter 'yes' or 'no'.")

        # --- THIS SECTION IS UPDATED ---
        elif validation_type == 'gender':
            if user_input in ['m']:
                return 'M'
            elif user_input in ['f']:
                return 'F'
            else:
                print("Invalid input. Please enter 'm' for male or 'f' for female.")
        # --- END OF UPDATE ---

        elif validation_type == 'age':
            try:
                age = int(user_input)
                if 18 <= age <= 100:
                    return age
                else:
                    print("Invalid input. Please enter an age between 18 and 100.")
            except ValueError:
                print("Invalid input. Please enter a valid number for age.")

def predict_lung_cancer_interactive(model_pipeline, feature_names):
    """
    Interactively asks a user questions to predict lung cancer risk.

    Args:
        model_pipeline: The trained scikit-learn pipeline (preprocessor + model).
        feature_names: A list of the feature names the model was trained on.
    """
    print("🩺 Please answer the following questions to predict lung cancer risk.")
    print("--- (Enter 'yes' or 'no', unless specified otherwise) ---")

    # --- Gather User Data ---
    patient_data = {
        'GENDER': get_validated_input("What is your gender? (m/f): ", 'gender'),
        'AGE': get_validated_input("What is your age?: ", 'age'),
        'SMOKING': get_validated_input("Do you smoke?: ", 'yes_no'),
        'YELLOW_FINGERS': get_validated_input("Do you have yellow fingers?: ", 'yes_no'),
        'ANXIETY': get_validated_input("Do you experience significant anxiety?: ", 'yes_no'),
        'PEER_PRESSURE': get_validated_input("Do you feel peer pressure to smoke?: ", 'yes_no'),
        'CHRONIC_DISEASE': get_validated_input("Do you have a chronic disease?: ", 'yes_no'),
        'FATIGUE': get_validated_input("Do you experience unusual fatigue?: ", 'yes_no'),
        'ALLERGY': get_validated_input("Do you have any severe allergies?: ", 'yes_no'),
        'WHEEZING': get_validated_input("Do you experience wheezing?: ", 'yes_no'),
        'ALCOHOL_CONSUMING': get_validated_input("Do you consume alcohol frequently?: ", 'yes_no'),
        'COUGHING': get_validated_input("Do you have a persistent cough?: ", 'yes_no'),
        'SHORTNESS_OF_BREATH': get_validated_input("Do you experience shortness of breath?: ", 'yes_no'),
        'SWALLOWING_DIFFICULTY': get_validated_input("Do you have difficulty swallowing?: ", 'yes_no'),
        'CHEST_PAIN': get_validated_input("Do you experience chest pain?: ", 'yes_no'),
    }

    # --- Make Prediction ---
    # Create a DataFrame with the exact same columns as the training data
    input_df = pd.DataFrame([patient_data], columns=feature_names)

    # Get prediction and probability
    prediction = model_pipeline.predict(input_df)[0]
    probability = model_pipeline.predict_proba(input_df)[0][1] # Probability of class '1' (cancer)

    # --- Display Result ---
    print("\n" + "="*40)
    print("          PREDICTION RESULT")
    print("="*40)

    if prediction == 1:
        print(f"⚠️ The model predicts a HIGH RISK of lung cancer.")
    else:
        print(f"✅ The model predicts a LOW RISK of lung cancer.")

    print(f"\nConfidence Score (Risk Probability): {probability:.0%}")
    print("="*40)



predict_lung_cancer_interactive(loaded_model, X_train.columns)