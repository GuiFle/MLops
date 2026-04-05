import os
import json
from datetime import datetime

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay, RocCurveDisplay
from joblib import dump

import mlflow
import mlflow.sklearn
from ydata_profiling import ProfileReport

# =========================
# Paths and parameters
# =========================
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
TRAIN_DATASET = os.path.join(BASE_DIR, "data/adult/adult_train.csv")
TEST_DATASET = os.path.join(BASE_DIR, "data/adult/adult_test.csv")
ARTIFACTS_DIR = os.path.join(BASE_DIR, "src/artifacts")

TARGET_COLUMN = "income"
RANDOM_STATE = 42
VALIDATION_FRAC = 0.15

# Create artifacts dir
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# =========================
# MLflow setup (safe for CI)
# =========================
MLFLOW_TRACKING_DIR = os.path.join(BASE_DIR, "mlruns")
os.makedirs(MLFLOW_TRACKING_DIR, exist_ok=True)
mlflow.set_tracking_uri(f"file://{MLFLOW_TRACKING_DIR}")
experiment = mlflow.set_experiment("adult_income_classification")
print("Tracking URI:", mlflow.get_tracking_uri())

# =========================
# Load train dataset
# =========================
df_train = pd.read_csv(TRAIN_DATASET)
features = [c for c in df_train.columns if c != TARGET_COLUMN]

X_full = df_train[features]
y_full = df_train[TARGET_COLUMN].str.strip().str.replace(".", "", regex=False)

X_train, X_val, y_train, y_val = train_test_split(
    X_full, y_full, test_size=VALIDATION_FRAC,
    stratify=y_full, random_state=RANDOM_STATE
)

# =========================
# Profiling
# =========================
profile = ProfileReport(df_train, title="Adult Dataset Profiling Report", explorative=True)
profile_path = os.path.join(ARTIFACTS_DIR, "adult_dataset_profile.html")
profile.to_file(profile_path)

# =========================
# Load test dataset
# =========================
df_test = pd.read_csv(TEST_DATASET)
X_test = df_test[features]
y_test = df_test[TARGET_COLUMN].str.strip().str.replace(".", "", regex=False)

# =========================
# Preprocessing
# =========================
categorical_features = X_train.select_dtypes(include=["object", "string", "category"]).columns.tolist()
preprocessor = ColumnTransformer(
    transformers=[("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), categorical_features)],
    remainder="passthrough"
)

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("classifier", DecisionTreeClassifier(random_state=RANDOM_STATE, max_depth=10))
])

# =========================
# MLflow run
# =========================
with mlflow.start_run(experiment_id=experiment.experiment_id) as run:
    print("Run ID:", run.info.run_id)

    # Log params
    mlflow.log_param("model_type", "DecisionTreeClassifier")
    mlflow.log_param("max_depth", 10)
    mlflow.log_param("random_state", RANDOM_STATE)
    mlflow.log_param("validation_frac", VALIDATION_FRAC)

    # Train
    pipeline.fit(X_train, y_train)

    # =========================
    # Evaluation
    # =========================
    y_test_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_test_pred)
    f1 = f1_score(y_test, y_test_pred, pos_label=">50K")

    mlflow.log_metrics({"accuracy": float(accuracy), "f1_score": float(f1)})

    # Save small prediction sample
    sample_df = X_test.copy()
    sample_df["y_true"] = y_test
    sample_df["y_pred"] = y_test_pred
    sample_path = os.path.join(ARTIFACTS_DIR, "prediction_samples.csv")
    sample_df.head(50).to_csv(sample_path, index=False)
    mlflow.log_artifact(sample_path)
    mlflow.log_table(data=sample_df.head(50), artifact_file="prediction_samples.json")

    # =========================
    # Confusion matrix
    # =========================
    cm_path = os.path.join(ARTIFACTS_DIR, "confusion_matrix.png")
    ConfusionMatrixDisplay.from_predictions(y_test, y_test_pred)
    plt.savefig(cm_path)
    plt.close()

    # =========================
    # ROC Curve
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
    # Metadata files
    # =========================
    metrics = {
        "train_dataset": TRAIN_DATASET,
        "test_dataset": TEST_DATASET,
        "target": TARGET_COLUMN,
        "metrics": {"accuracy": accuracy, "f1_score": f1},
        "timestamp": datetime.now().isoformat()
    }
    with open(os.path.join(ARTIFACTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=4)

    feature_schema = {col: str(X_train[col].dtype) for col in features}
    with open(os.path.join(ARTIFACTS_DIR, "feature_schema.json"), "w") as f:
        json.dump(feature_schema, f, indent=4)

    dataset_info = {
        "dataset_name": TRAIN_DATASET,
        "dataset_shape": {"rows": X_train.shape[0] + X_val.shape[0] + X_test.shape[0], "columns": X_train.shape[1] + 1},
        "target_column": TARGET_COLUMN
    }
    with open(os.path.join(ARTIFACTS_DIR, "dataset_info.json"), "w") as f:
        json.dump(dataset_info, f, indent=4)

    # Log all artifacts
    mlflow.log_artifacts(ARTIFACTS_DIR)

    # =========================
    # GenAI evaluation (optional)
    # =========================
    from mlflow.genai.scorers import Correctness

    @mlflow.trace(name="genai_evaluation_prediction")
    def predict_fn(**inputs):
        if "row_dict" in inputs:
            inputs = inputs["row_dict"]
        df = pd.DataFrame([inputs])
        return pipeline.predict(df)[0]

    eval_dataset = []
    for i in range(min(50, X_test.shape[0])):
        row = X_test.iloc[i].to_dict()
        eval_dataset.append({"inputs": row, "expectations": {"expected_response": y_test.iloc[i]}})

    mlflow.genai.evaluate(data=eval_dataset, predict_fn=predict_fn, scorers=[Correctness()])

    # =========================
    # Final prints
    # =========================
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-score: {f1:.4f}")