import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay, RocCurveDisplay
from joblib import dump
import json
from datetime import datetime
import os
import mlflow
import mlflow.sklearn
import matplotlib.pyplot as plt

# =========================
# Parameters
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

TRAIN_DATASET = os.path.join(BASE_DIR, "data/adult/adult_train.csv")
TEST_DATASET = os.path.join(BASE_DIR, "data/adult/adult_test.csv")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "src/artifacts")

TARGET_COLUMN = "income"
RANDOM_STATE = 42
VALIDATION_FRAC = 0.15

# MLflow config
DB_PATH = os.path.join(BASE_DIR, "mlflow.db")
mlflow.set_tracking_uri(f"sqlite:///{DB_PATH}")
experiment = mlflow.set_experiment("adult_income_classification")

print("Tracking URI:", mlflow.get_tracking_uri())

os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# =========================
# Load train dataset
# =========================
df_train = pd.read_csv(TRAIN_DATASET)
features = [c for c in df_train.columns if c != TARGET_COLUMN]

X_full = df_train[features]
y_full = df_train[TARGET_COLUMN]
y_full = y_full.str.strip().str.replace(".", "", regex=False)

X_train, X_val, y_train, y_val = train_test_split(
    X_full, y_full, test_size=VALIDATION_FRAC,
    stratify=y_full, random_state=RANDOM_STATE
)

# =========================
# Profiling
# =========================
from ydata_profiling import ProfileReport

profile = ProfileReport(df_train, title="Adult Dataset Profiling Report", explorative=True)
profile_path = os.path.join(ARTIFACTS_DIR, "adult_dataset_profile.html")
profile.to_file(profile_path)

# =========================
# Load test dataset
# =========================
df_test = pd.read_csv(TEST_DATASET)
X_test = df_test[features]
y_test = df_test[TARGET_COLUMN]
y_test = y_test.str.strip().str.replace(".", "", regex=False)

# =========================
# Preprocessing
# =========================
categorical_features = X_train.select_dtypes(
    include=["object", "string", "category"]
).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)
    ],
    remainder="passthrough"
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(
        random_state=RANDOM_STATE,
        max_depth=10
    ))
])

# =========================
# MLflow run
# =========================
with mlflow.start_run(experiment_id=experiment.experiment_id) as run:

    print("Run ID:", run.info.run_id)

    # Params
    mlflow.log_param("model_type", "DecisionTreeClassifier")
    mlflow.log_param("max_depth", 10)
    mlflow.log_param("random_state", RANDOM_STATE)
    mlflow.log_param("validation_frac", VALIDATION_FRAC)

    # Training
    pipeline.fit(X_train, y_train)

    # Evaluation
    y_test_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred, pos_label=">50K")

    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("f1_score", f1)

    # Take a small sample
    sample_df = X_test.copy()
    sample_df["y_true"] = y_test
    sample_df["y_pred"] = y_test_pred

    sample_path = os.path.join(ARTIFACTS_DIR, "prediction_samples.csv")
    sample_df.head(50).to_csv(sample_path, index=False)

    mlflow.log_artifact(sample_path)

    mlflow.log_table(
    data=sample_df.head(50),
    artifact_file="prediction_samples.json"
)

    # =========================
    # NEW: Confusion Matrix
    # =========================
    cm_path = os.path.join(ARTIFACTS_DIR, "confusion_matrix.png")
    ConfusionMatrixDisplay.from_predictions(y_test, y_test_pred)
    plt.savefig(cm_path)
    plt.close()

    # =========================
    # NEW: ROC Curve
    # =========================
    roc_path = os.path.join(ARTIFACTS_DIR, "roc_curve.png")
    RocCurveDisplay.from_estimator(pipeline, X_test, y_test)
    plt.savefig(roc_path)
    plt.close()

    # =========================
    # Save model
    # =========================
    model_path = os.path.join(ARTIFACTS_DIR, "model.joblib")
    dump(pipeline, model_path)

    mlflow.sklearn.log_model(pipeline, "model")

    # =========================
    # Save metadata files
    # =========================
    metrics = {
        "train_dataset": TRAIN_DATASET,
        "test_dataset": TEST_DATASET,
        "target": TARGET_COLUMN,
        "metrics": {"accuracy": accuracy, "f1_score": f1},
        "timestamp": datetime.now().isoformat()
    }

    metrics_path = os.path.join(ARTIFACTS_DIR, "metrics.json")
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    feature_schema = {col: str(X_train[col].dtype) for col in features}
    feature_schema_path = os.path.join(ARTIFACTS_DIR, "feature_schema.json")
    with open(feature_schema_path, "w") as f:
        json.dump(feature_schema, f, indent=4)

    total_rows = X_train.shape[0] + X_val.shape[0] + X_test.shape[0]
    dataset_info = {
        "dataset_name": TRAIN_DATASET,
        "dataset_shape": {"rows": total_rows, "columns": X_train.shape[1] + 1},
        "target_column": TARGET_COLUMN
    }

    dataset_info_path = os.path.join(ARTIFACTS_DIR, "dataset_info.json")
    with open(dataset_info_path, "w") as f:
        json.dump(dataset_info, f, indent=4)

    # =========================
    # Log ALL artifacts at once
    # =========================
    mlflow.log_artifacts(ARTIFACTS_DIR)

# =========================
# Final prints
# =========================
print(f"Accuracy: {accuracy:.4f}")
print(f"F1-score: {f1:.4f}")