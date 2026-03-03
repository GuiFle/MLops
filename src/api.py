# src/mlops_tp/api.py
from fastapi import FastAPI, Body
from pydantic import BaseModel
from typing import Union, List
import pandas as pd
from joblib import load
import json
import os
import time

# =========================
# Parameters
# =========================
ARTIFACTS_DIR = "artifacts/"
MODEL_FILE = os.path.join(ARTIFACTS_DIR, "model.joblib")
SCHEMA_FILE = os.path.join(ARTIFACTS_DIR, "feature_schema.json")
DEFAULT_CSV = os.path.join("..", "data", "adult", "adult_test.csv")
MODEL_VERSION = "1.0"
TASK_TYPE = "classification"
CLASS_MAPPING = {"<=50K": "<=50K", ">50K": ">50K"}

# =========================
# Load model and feature schema
# =========================
model = load(MODEL_FILE)
with open(SCHEMA_FILE, "r") as f:
    feature_schema = json.load(f)
EXPECTED_FEATURES = list(feature_schema.keys())

# =========================
# Request model for POST /predict
# =========================
class PredictRequest(BaseModel):
    features: Union[dict, List[dict]]  # Accept single dict or list of dicts

# =========================
# FastAPI app
# =========================
app = FastAPI(title="Adult Income Prediction API", version=MODEL_VERSION)

# =========================
# GET /health
# =========================
@app.get("/health")
def health():
    return {"status": "alive"}

# =========================
# GET /metadata
# =========================
@app.get("/metadata")
def metadata():
    return {
        "model_version": MODEL_VERSION,
        "task_type": TASK_TYPE,
        "expected_features": EXPECTED_FEATURES
    }

# =========================
# POST /predict
# =========================
@app.post("/predict")
def predict(request: PredictRequest = Body(...)):
    start_time = time.time()
    try:
        # Determine if batch or single
        if isinstance(request.features, list):
            df = pd.DataFrame(request.features)  # batch
        elif isinstance(request.features, dict):
            df = pd.DataFrame([request.features])  # single
        else:
            return {"error": "Invalid features format"}

        # Check for missing columns
        missing_cols = set(EXPECTED_FEATURES) - set(df.columns)
        if missing_cols:
            return {"error": f"Missing columns: {missing_cols}"}

        # Prediction
        y_pred = model.predict(df)
        try:
            y_proba_array = model.predict_proba(df)
        except AttributeError:
            y_proba_array = None

        results = []
        for i, pred in enumerate(y_pred):
            proba_dict = {}
            if y_proba_array is not None:
                for idx, cls in enumerate(model.classes_):
                    proba_dict[CLASS_MAPPING.get(cls, str(cls))] = float(y_proba_array[i][idx])
            results.append({
                "prediction": CLASS_MAPPING.get(pred, str(pred)),
                "proba": proba_dict,
                "task": TASK_TYPE,
                "model_version": MODEL_VERSION,
                "latency_ms": round((time.time() - start_time) * 1000, 2)
            })

        return {"results": results}

    except Exception as e:
        return {"error": str(e)}