import os
import subprocess

ARTIFACTS_DIR = "artifacts/"
MODEL_FILE = os.path.join(ARTIFACTS_DIR, "model.joblib")

def test_training_creates_model():
    # Remove old model if it exists
    if os.path.exists(MODEL_FILE):
        os.remove(MODEL_FILE)
    
    # Run the training script for the Adult dataset
    result = subprocess.run(
        ["python3", "mlops.py"],  # make sure mlops.py now trains on adult_train.csv
        capture_output=True,
        text=True
    )
    
    # Check that the script finished successfully
    assert result.returncode == 0, f"Training script failed: {result.stderr}"
    
    # Check that model.joblib was created
    assert os.path.exists(MODEL_FILE), "The file model.joblib was not created"

    print("✅ test_training passed: model.joblib successfully created")