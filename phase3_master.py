
import os, datetime, uuid, sys, importlib
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # headless backend for script execution
import matplotlib.pyplot as plt
import seaborn as sns

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
