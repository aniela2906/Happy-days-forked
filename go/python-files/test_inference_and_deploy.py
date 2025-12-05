import os
import sys
import json
import time
from datetime import datetime

# --- LIBRARIES FOR INFERENCE AND DEPLOYMENT ---
import mlflow
import pandas as pd
import joblib
from sklearn.metrics import accuracy_score 
from mlflow.tracking.client import MlflowClient
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
from xgboost import XGBRFClassifier # Needed to load XGBoost model
import numpy as np

# --- CONSTANTS ---
ARTIFACT_FOLDER = "model" # The name of the folder created by the 'Download Artifact' step
MODEL_RESULTS_PATH = os.path.join(ARTIFACT_FOLDER, "model_results.json")
MODEL_NAME = "lead_model"
ARTIFACT_PATH = "model" 
# Assumes experiment name is the current date from the training run
EXPERIMENT_NAME = datetime.now().strftime("%Y_%B_%d") 

# --- HELPER FUNCTIONS (Required for Deployment Logic) ---
def wait_until_ready(model_name, model_version):
    """Waits for an MLflow registered model version to be ready."""
    client = MlflowClient()
    for _ in range(10):
        model_version_details = client.get_model_version(name=model_name, version=model_version)
        status = ModelVersionStatus.from_string(model_version_details.status)
        if status == ModelVersionStatus.READY:
            print(f"Model version {model_version} is ready.")
            break
        time.sleep(1)

def wait_for_deployment(model_name, model_version, stage='Staging'):
    """Waits for an MLflow model version to transition to the target stage."""
    client = MlflowClient()
    status = False
    while not status:
        model_version_details = dict(client.get_model_version(name=model_name,version=model_version))
        if model_version_details['current_stage'] == stage:
            print(f'Transition completed to {stage}')
            status = True
            break
        else:
            time.sleep(2)
    return status

# --- CORE INFERENCE TEST LOGIC ---

def run_inference_test(expected_accuracy_threshold=0.75):
    """
    Loads the model artifact and performs a functional inference test using the saved test set.
    If the model cannot load or the accuracy is too low, the script exits (fails the job).
    """
    print("\n--- Running Artifact Integrity and Inference Test ---")

    # Paths to the downloaded test set and model artifact (from the model/ folder)
    X_TEST_PATH = os.path.join(ARTIFACT_FOLDER, "X_test.csv")
    Y_TEST_PATH = os.path.join(ARTIFACT_FOLDER, "y_test.csv")
    LR_MODEL_PATH = os.path.join(ARTIFACT_FOLDER, "lead_model_lr.pkl")
    
    # 1. Load Model and Test Data
    try:
        # Load test data saved by train_model.py
        X_test = pd.read_csv(X_TEST_PATH)
        y_test = pd.read_csv(Y_TEST_PATH)
        # Load the Logistic Regression model (the model promoted by pyfunc logging)
        lr_model = joblib.load(LR_MODEL_PATH)
        
    except FileNotFoundError as e:
        print(f"ERROR: Missing required artifact file for inference: {e}")
        sys.exit(1)

    # 2. Perform Inference
    print("Executing model prediction on test set...")
    y_pred = lr_model.predict(X_test)
    
    # 3. Validation Check (Functional Test)
    # Calculate the actual accuracy against the truth values
    test_accuracy = accuracy_score(y_test, y_pred)
    
    if test_accuracy >= expected_accuracy_threshold:
        print(f"INFERENCE TEST PASSED: Accuracy ({test_accuracy:.2f}) meets required threshold.")
        return True
    else:
        # If the test fails, the job immediately fails (exit code 1), blocking deployment.
        print(f"INFERENCE TEST FAILED: Accuracy {test_accuracy:.2f} is too low. Promotion blocked.")
        sys.exit(1)

# --- DEPLOYMENT LOGIC (Final Promotion) ---
def finalize_deployment():
    print("\n--- Starting Final MLflow Deployment Stage ---")

    # Search MLflow for the best run that produced the required artifact
    try:
        # MLflow must be configured to point to the correct tracking server here.
        # It relies on the environment variables set by the runner to find the experiment name.
        experiment_ids = [mlflow.get_experiment_by_name(EXPERIMENT_NAME).experiment_id]
        experiment_best = mlflow.search_runs(
            experiment_ids=experiment_ids,
            order_by=["metrics.f1_score DESC"],
            max_results=1
        ).iloc[0]
    except Exception as e:
        print(f"Error accessing MLflow experiment: {e}. Cannot deploy.")
        sys.exit(1)

    # 1. Register Best Model
    run_id = experiment_best["run_id"]
    model_uri = "runs:/{run_id}/{artifact_path}".format(run_id=run_id, artifact_path=ARTIFACT_PATH)
    
    model_details = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
    wait_until_ready(model_details.name, model_details.version)
    
    # 2. Transition to Staging
    model_version = model_details.version
    client = MlflowClient()

    client.transition_model_version_stage(
        name=MODEL_NAME,
        version=model_version,
        stage="Staging", 
        archive_existing_versions=True
    )
    wait_for_deployment(MODEL_NAME, model_version, 'Staging')
    
    print("\Model Promoted to Staging.")

if __name__ == "__main__":
    # The job runs this function. If run_inference_test() fails, the process exits (exit code 1).
    if run_inference_test():
        finalize_deployment()