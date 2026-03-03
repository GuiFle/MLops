import pandas as pd
from joblib import load

ARTIFACTS_DIR = "artifacts/"
MODEL_FILE = ARTIFACTS_DIR + "model.joblib"
def test_inference_predict():
    # Load the trained model
    model = load(MODEL_FILE)
    
    # Create a mini dataset for testing predictions
    test_data = pd.DataFrame({
        "age": [37, 50],
        "workclass": ["Private", "Self-emp-not-inc"],
        "fnlwgt": [284582, 83311],
        "education": ["Bachelors", "HS-grad"],
        "education_num": [13, 9],            # changed
        "marital_status": ["Married-civ-spouse", "Never-married"],  # changed
        "occupation": ["Exec-managerial", "Adm-clerical"],
        "relationship": ["Husband", "Not-in-family"],
        "race": ["White", "Black"],
        "sex": ["Male", "Female"],
        "capital_gain": [0, 2174],           # changed
        "capital_loss": [0, 0],              # changed
        "hours_per_week": [40, 40],          # changed
        "native_country": ["United-States", "United-States"]  # changed
    })
    
    # Predict
    y_pred = model.predict(test_data)
    assert all(c in ["<=50K", ">50K"] for c in y_pred), f"Unknown classes predicted: {y_pred}"
    
    # Predict probabilities
    y_prob = model.predict_proba(test_data)
    assert (y_prob >= 0).all() and (y_prob <= 1).all(), "predict_proba contains values outside [0,1]"
    
    print("✅ test_inference passed: predict and predict_proba OK")