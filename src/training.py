import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from joblib import dump
import json
from datetime import datetime
import os

# =========================
# Parameters
# =========================
TRAIN_DATASET = "../data/adult/adult_train.csv"
TEST_DATASET = "../data/adult/adult_test.csv"
TARGET_COLUMN = "income"
ARTIFACTS_DIR = "artifacts/"
RANDOM_STATE = 42
VALIDATION_FRAC = 0.15  # fraction of training data to use as validation

os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# =========================
# Load train dataset
# =========================
df_train = pd.read_csv(TRAIN_DATASET)
features = [c for c in df_train.columns if c != TARGET_COLUMN]

# Split train into train + validation
X_full = df_train[features]
y_full = df_train[TARGET_COLUMN]

# Clean potential trailing dots
y_full = y_full.str.strip().str.replace(".", "", regex=False)

X_train, X_val, y_train, y_val = train_test_split(
    X_full, y_full, test_size=VALIDATION_FRAC, stratify=y_full, random_state=RANDOM_STATE
)

# =========================
# YData Profiling
# =========================
from ydata_profiling import ProfileReport

profile = ProfileReport(df_train, 
                        title="Adult Dataset Profiling Report",
                        explorative=True)
profile_path = os.path.join(ARTIFACTS_DIR, "adult_dataset_profile.html")
profile.to_file(profile_path)
print(f"Profiling report saved to {profile_path}")

# =========================
# Load test dataset
# =========================
df_test = pd.read_csv(TEST_DATASET)
X_test = df_test[features]
y_test = df_test[TARGET_COLUMN]

# Clean potential trailing dots in test set
y_test = y_test.str.strip().str.replace(".", "", regex=False)

# =========================
# Identify categorical columns
# =========================
categorical_features = X_train.select_dtypes(
    include=["object", "string", "category"]
).columns.tolist()

# =========================
# Preprocessing
# =========================
preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
    ],
    remainder="passthrough"
)

# =========================
# Pipeline
# =========================
pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(
        random_state=RANDOM_STATE,
        max_depth=10
    ))
])

# =========================
# Training
# =========================
pipeline.fit(X_train, y_train)

# =========================
# Evaluation on test set
# =========================
y_test_pred = pipeline.predict(X_test)
accuracy = accuracy_score(y_test, y_test_pred)
f1 = f1_score(y_test, y_test_pred, pos_label=">50K")

# =========================
# Save model
# =========================
model_path = os.path.join(ARTIFACTS_DIR, "model.joblib")
dump(pipeline, model_path)

# =========================
# Save metrics
# =========================
metrics = {
    "train_dataset": TRAIN_DATASET,
    "test_dataset": TEST_DATASET,
    "target": TARGET_COLUMN,
    "metrics": {
        "accuracy": accuracy,
        "f1_score": f1
    },
    "hyperparameters": {
        "classifier": "DecisionTreeClassifier",
        "max_depth": 10
    },
    "timestamp": datetime.now().isoformat()
}

metrics_path = os.path.join(ARTIFACTS_DIR, "metrics.json")
with open(metrics_path, "w") as f:
    json.dump(metrics, f, indent=4)

# =========================
# Save feature schema
# =========================
feature_schema = {col: str(X_train[col].dtype) for col in features}
feature_schema_path = os.path.join(ARTIFACTS_DIR, "feature_schema.json")
with open(feature_schema_path, "w") as f:
    json.dump(feature_schema, f, indent=4)

# =========================
# Save dataset info JSON
# =========================
total_rows = X_train.shape[0] + X_val.shape[0] + X_test.shape[0]
dataset_info = {
    "dataset_name": TRAIN_DATASET,
    "dataset_shape": {
        "rows": total_rows,
        "columns": X_train.shape[1] + 1  # +1 for target
    },
    "target_column": TARGET_COLUMN,
    "features_used": features,
    "split_fractions": {
        "train": X_train.shape[0] / total_rows,
        "validation": X_val.shape[0] / total_rows,
        "test": X_test.shape[0] / total_rows
    },
    "random_state": RANDOM_STATE
}

dataset_info_path = os.path.join(ARTIFACTS_DIR, "dataset_info.json")
with open(dataset_info_path, "w") as f:
    json.dump(dataset_info, f, indent=4)

# =========================
# Final Info
# =========================
print(f"Train size: {X_train.shape[0]}")
print(f"Validation size: {X_val.shape[0]}")
print(f"Test size: {X_test.shape[0]}")
print(f"Accuracy (test): {accuracy:.4f}")
print(f"F1-score (test): {f1:.4f}")
print(f"Income distribution (train):\n{y_train.value_counts()}")
print(f"Dataset info saved to {dataset_info_path}")
