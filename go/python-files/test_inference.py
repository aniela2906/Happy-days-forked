import os
import sys
import pandas as pd
import joblib
import numpy as np
from sklearn.metrics import accuracy_score 
from sklearn.linear_model import LogisticRegression # Needed for joblib compatibility

# --- CONSTANTS ---
# This is the name of the folder where the GitHub Action downloads artifacts
ARTIFACT_FOLDER = "model" 

# Paths to files inside the downloaded artifact folder
X_TEST_PATH = os.path.join(ARTIFACT_FOLDER, "X_test.csv")
Y_TEST_PATH = os.path.join(ARTIFACT_FOLDER, "y_test.csv")
LR_MODEL_PATH = os.path.join(ARTIFACT_FOLDER, "lead_model_lr.pkl")

# --- CORE INFERENCE TEST LOGIC ---
def run_inference_test():
    print("\n--- Running Artifact Integrity and Inference Test (Job 2) ---")
    
    # 1. Load Model and Test Data
    try:
        # Load the artifacts produced by the train job
        X_test = pd.read_csv(X_TEST_PATH)
        # Flatten the y_test column into a simple array
        y_test = pd.read_csv(Y_TEST_PATH).values.flatten() 
        # Load the fitted Logistic Regression model
        lr_model = joblib.load(LR_MODEL_PATH)
    except Exception as e:
        print(f"FATAL ERROR: Missing or corrupted artifact for inference: {e}", file=sys.stderr)
        sys.exit(1)

    # 2. Perform Inference on the first 5 samples
    y_pred = lr_model.predict(X_test.head(5))
    y_true = y_test[:5]
    
    # 3. Validation Check
    # The expected output from the reproducible training run:
    expected_pred = np.array([0., 1., 0., 1., 0.])
    expected_y = np.array([0., 1., 0., 1., 0.]) 

    # Check if the model's prediction AND the actual label values are correct
    if not np.array_equal(y_pred, expected_pred) or not np.array_equal(y_true, expected_y):
        print("INFERENCE TEST FAILED: Outputs do not match hardcoded expectations!", file=sys.stderr)
        print("Model is non-reproducible. Check training script.", file=sys.stderr)
        sys.exit(1) # Fail the job if the test fails

    print("✅ INFERENCE TEST PASSED: Outputs match expectations.")
    
    # Optional performance check
    test_accuracy = accuracy_score(y_test, lr_model.predict(X_test))
    print(f"✅ INFO: Full Test Set Accuracy: {test_accuracy:.4f}")
    
    return True

if __name__ == "__main__":
    try:
        run_inference_test()
        print("\n✅ Inference Test workflow finished successfully (Exit Code 0).")
    except Exception as e:
        print(f"\n❌ An unrecoverable error occurred during testing: {e}", file=sys.stderr)
        sys.exit(1)