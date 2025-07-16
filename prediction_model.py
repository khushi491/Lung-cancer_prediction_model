from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import GridSearchCV
import joblib

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

print(f"\n✅ Model saved successfully to {model_path}")

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
