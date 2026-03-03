import pytest
from fastapi.testclient import TestClient
from api import app, CLASS_MAPPING  # make sure your API uses Adult dataset

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
    features = {
        "age": 37,
        "workclass": "Private",
        "fnlwgt": 284582,
        "education": "Bachelors",
        "education_num": 13,          # changed
        "marital_status": "Married-civ-spouse",  # changed
        "occupation": "Exec-managerial",
        "relationship": "Husband",
        "race": "White",
        "sex": "Male",
        "capital_gain": 0,            # changed
        "capital_loss": 0,            # changed
        "hours_per_week": 40,         # changed
        "native_country": "United-States"  # changed
    }
    response = client.post("/predict", json={"features": features})
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
# POST /predict with no body (default CSV)
# -----------------------------
def test_predict_default_csv():
    response = client.post("/predict", json={})
    assert response.status_code == 200
    data = response.json()
    assert "results" in data
    # each prediction has a latency_ms
    for res in data["results"]:
        assert "latency_ms" in res
    # at least one prediction
    assert len(data["results"]) > 0
    for res in data["results"][:3]:
        assert res["prediction"] in CLASS_MAPPING.values()