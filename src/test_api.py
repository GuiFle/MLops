# src/test_api.py
import pytest
from fastapi.testclient import TestClient
from api import app, CLASS_MAPPING  # Assure-toi que ton API correspond au dataset Adult

client = TestClient(app)

# -----------------------------
# GET /health
# -----------------------------
def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "alive"}

# -----------------------------
# GET /metadata
# -----------------------------
def test_metadata():
    response = client.get("/metadata")
    assert response.status_code == 200
    data = response.json()
    assert "model_version" in data
    assert "task_type" in data
    assert "expected_features" in data
    assert isinstance(data["expected_features"], list)

# -----------------------------
# POST /predict with valid features
# -----------------------------
def test_predict_valid():
    valid_input = {
        "features": {
            "age": 37,
            "workclass": "Private",
            "fnlwgt": 284582,
            "education": "Bachelors",
            "education_num": 13,
            "marital_status": "Never-married",
            "occupation": "Prof-specialty",
            "relationship": "Not-in-family",
            "race": "White",
            "sex": "Male",
            "capital_gain": 0,
            "capital_loss": 0,
            "hours_per_week": 40,
            "native_country": "United-States"
        }
    }
    response = client.post("/predict", json=valid_input)
    assert response.status_code == 200
    data = response.json()
    assert "results" in data

# -----------------------------
# POST /predict with missing features
# -----------------------------
def test_predict_missing_features():
    features = {
        "age": 37,
        "workclass": "Private"
        # missing all other features
    }
    response = client.post("/predict", json={"features": features})
    assert response.status_code == 200  # endpoint returns 200 but error in JSON
    data = response.json()
    assert "error" in data
    assert "Missing columns" in data["error"]

# -----------------------------
# POST /predict with default input (was failing)
# -----------------------------
def test_predict_default_csv():
    test_input = {
        "features": { 
            "age": 37,
            "workclass": "Private",
            "fnlwgt": 284582,
            "education": "Bachelors",
            "education_num": 13,
            "marital_status": "Married-civ-spouse",
            "occupation": "Exec-managerial",
            "relationship": "Husband",
            "race": "White",
            "sex": "Male",
            "capital_gain": 0,
            "capital_loss": 0,
            "hours_per_week": 40,
            "native_country": "United-States"
        }
    }

    response = client.post("/predict", json=test_input)
    print(response.json())  # pour debug si besoin
    assert response.status_code == 200

    results = response.json().get("results", [])
    assert len(results) == 1
    assert "prediction" in results[0]
    assert "proba" in results[0]
    assert results[0]["task"] == "classification"