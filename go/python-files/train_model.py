import os
import sys
import warnings
import json
import datetime
import time

# --- 1. DATA PROCESSING LIBRARIES ---
import pandas as pd
import numpy as np
import shutil
from pprint import pprint
from sklearn.preprocessing import MinMaxScaler
import joblib

# --- 2. MODELING LIBRARIES ---
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from xgboost import XGBRFClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report, f1_score
from scipy.stats import uniform, randint

# --- 3. MLFLOW/DEPLOYMENT LIBRARIES ---
import mlflow
import mlflow.sklearn
import mlflow.pyfunc
from mlflow.tracking.client import MlflowClient
from mlflow.entities.model_registry.model_version_status import ModelVersionStatus
from mlflow.tracking.client import MlflowClient # Imported twice in notebook, kept for consistency

# Set Pandas options and filter warnings (from notebook)
warnings.filterwarnings('ignore')
pd.set_option('display.float_format', lambda x: "%.3f" % x)

# --- 4. CONSTANTS / FILE PATHS ---
# Dagger mounts 'go/python-files' as '/python' inside the container.
INPUT_DIR = "/python" 
OUTPUT_DIR = "/output" 

ARTIFACTS_DIR = os.path.join(INPUT_DIR, "artifacts")
DATA_PATH = os.path.join(ARTIFACTS_DIR, "raw_data.csv")
GOLD_DATA_PATH = os.path.join(ARTIFACTS_DIR, "train_data_gold.csv")
MODEL_NAME = "lead_model"
ARTIFACT_PATH = "model" # Artifact path used for MLflow logging

# --- 5. HELPER FUNCTIONS ---

def describe_numeric_col(x):
    """Calculates descriptive stats for a numeric column."""
    return pd.Series(
        [x.count(), x.isnull().count(), x.mean(), x.min(), x.max()],
        index=["Count", "Missing", "Mean", "Min", "Max"]
    )

def impute_missing_values(x, method="mean"):
    """Imputes the mean/median for numeric columns or the mode for others."""
    if (x.dtype == "float64") | (x.dtype == "int64"):
        x = x.fillna(x.mean()) if method=="mean" else x.fillna(x.median())
    else:
        # Use mode imputation for categorical/object types
        x = x.fillna(x.mode()[0])
    return x

def create_dummy_cols(df, col):
    """Create one-hot encoding columns in the data."""
    df_dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
    new_df = pd.concat([df, df_dummies], axis=1)
    new_df = new_df.drop(col, axis=1)
    return new_df

def wait_until_ready(model_name, model_version):
    """Waits for an MLflow registered model version to be ready."""
    client = MlflowClient()
    for _ in range(10):
        model_version_details = client.get_model_version(name=model_name, version=model_version)
        status = ModelVersionStatus.from_string(model_version_details.status)
        if status == ModelVersionStatus.READY:
            break
        time.sleep(1)

def wait_for_deployment(model_name, model_version, stage='Staging'):
    """Waits for an MLflow model version to transition to the target stage."""
    client = MlflowClient()
    status = False
    while not status:
        model_version_details = dict(client.get_model_version(name=model_name,version=model_version))
        if model_version_details['current_stage'] == stage:
            status = True
            break
        else:
            time.sleep(2)
    return status

class lr_wrapper(mlflow.pyfunc.PythonModel):
    """MLflow wrapper for Logistic Regression to predict probability."""
    def __init__(self, model):
        self.model = model
    
    def predict(self, context, model_input):
        return self.model.predict_proba(model_input)[:, 1]

# --- 6. DATA PROCESSING STAGE ---
def process_data(data_path, artifacts_dir):
    print("--- Starting Data Processing Stage ---")

    # The notebook creates the artifacts dir; we ensure it exists and skip dvc pull (handled by GA)
    os.makedirs(artifacts_dir, exist_ok=True)
    
    data = pd.read_csv(data_path)
    
    # Simulate date filtering logic (using hardcoded dates from notebook for consistency)
    min_date_str = "2024-01-01"
    max_date_str = "2024-01-31"
    max_date = pd.to_datetime(max_date_str).date()
    min_date = pd.to_datetime(min_date_str).date()
    
    data["date_part"] = pd.to_datetime(data["date_part"]).dt.date
    data = data[(data["date_part"] >= min_date) & (data["date_part"] <= max_date)]
    
    # Save date limits artifact
    date_limits = {"min_date": str(data["date_part"].min()), "max_date": str(data["date_part"].max())}
    with open(os.path.join(artifacts_dir, "date_limits.json"), "w") as f:
        json.dump(date_limits, f)
    
    # 2. Feature Selection (Removing first set of columns)
    data = data.drop(
        ["is_active", "marketing_consent", "first_booking", "existing_customer", "last_seen"],
        axis=1
    )
    # Feature Selection (Removing second set of columns)
    data = data.drop(
        ["domain", "country", "visited_learn_more_before_booking", "visited_faq"],
        axis=1
    )
    
    # 3. Data Cleaning (Handling empty strings and missing target)
    data["lead_indicator"].replace("", np.nan, inplace=True)
    data["lead_id"].replace("", np.nan, inplace=True)
    data["customer_code"].replace("", np.nan, inplace=True)
    data = data.dropna(axis=0, subset=["lead_indicator"])
    data = data.dropna(axis=0, subset=["lead_id"])
    data = data[data.source == "signup"] # Filter by source
    
    # 4. Data Type Conversion (Categorical)
    vars_to_convert = ["lead_id", "lead_indicator", "customer_group", "onboarding", "source", "customer_code"]
    for col in vars_to_convert:
        data[col] = data[col].astype("object")

    # 5. Separate Continuous and Categorical
    cont_vars = data.loc[:, ((data.dtypes=="float64")|(data.dtypes=="int64"))]
    cat_vars = data.loc[:, (data.dtypes=="object")]
    
    # 6. Outlier Handling (Clipping)
    cont_vars = cont_vars.apply(lambda x: x.clip(lower = (x.mean()-2*x.std()),
                                                 upper = (x.mean()+2*x.std())))
    outlier_summary = cont_vars.apply(describe_numeric_col).T
    outlier_summary.to_csv(os.path.join(artifacts_dir, 'outlier_summary.csv'))
    
    # 7. Impute Missing Data
    cat_missing_impute = cat_vars.mode(numeric_only=False, dropna=True)
    cat_missing_impute.to_csv(os.path.join(artifacts_dir, "cat_missing_impute.csv"))
    
    # Impute continuous variables
    cont_vars = cont_vars.apply(impute_missing_values)
    
    # Specific categorical imputation (as done in notebook)
    cat_vars.loc[cat_vars['customer_code'].isna(),'customer_code'] = 'None'
    cat_vars = cat_vars.apply(impute_missing_values) # Impute remaining cat NaNs with mode

    # 8. Data Standardization (MinMaxScaler)
    scaler = MinMaxScaler()
    scaler.fit(cont_vars)
    joblib.dump(value=scaler, filename=os.path.join(artifacts_dir, "scaler.pkl"))
    cont_vars = pd.DataFrame(scaler.transform(cont_vars), columns=cont_vars.columns)

    # 9. Combine Data
    cont_vars = cont_vars.reset_index(drop=True)
    cat_vars = cat_vars.reset_index(drop=True)
    data = pd.concat([cat_vars, cont_vars], axis=1)
    
    # 10. Data Drift Artifact
    data_columns = list(data.columns)
    with open(os.path.join(artifacts_dir, 'columns_drift.json'), 'w+') as f:           
        json.dump(data_columns, f)
        data.to_csv(os.path.join(artifacts_dir, 'training_data.csv'), index=False)

    # 11. Binning (Source column transformation)
    mapping = {'li' : 'socials', 'fb' : 'socials', 'organic': 'group1', 'signup': 'group1'}
    data['bin_source'] = data['source'].map(mapping)
    
    # 12. Save Final Gold Medallion Dataset
    data.to_csv(GOLD_DATA_PATH, index=False)
    
    print("--- Data Processing Stage Complete ---")
    return data

# --- 7. MODEL TRAINING AND SELECTION STAGE ---

def train_and_select_model(data, artifacts_dir):
    print("--- Starting Model Training Stage ---")
    
    # MLflow Setup 
    current_date = datetime.datetime.now().strftime("%Y_%B_%d")
    experiment_name = current_date
    os.makedirs("mlruns", exist_ok=True)
    os.makedirs("mlruns/.trash", exist_ok=True)
    mlflow.set_experiment(experiment_name)
    mlflow.sklearn.autolog(log_input_examples=True, log_models=False)
    
    # 1. Data Splitting and Feature Engineering
    data = data.drop(["lead_id", "customer_code", "date_part"], axis=1)
    cat_cols = ["customer_group", "onboarding", "bin_source", "source"]
    cat_vars = data[cat_cols]
    other_vars = data.drop(cat_cols, axis=1)
    
    # Dummy variables and type conversion
    for col in cat_vars:
        cat_vars[col] = cat_vars[col].astype("category")
        cat_vars = create_dummy_cols(cat_vars, col)
    
    data = pd.concat([other_vars, cat_vars], axis=1)
    
    # Final type conversion to float
    for col in data:
        data[col] = data[col].astype("float64")
        
    y = data["lead_indicator"]
    X = data.drop(["lead_indicator"], axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, test_size=0.15, stratify=y
    )
    
    model_results = {}
    
    # --- XGBoost Training ---
    print("\n--- Training XGBoost ---")
    model = XGBRFClassifier(random_state=42)
    params_xgb = {
        "learning_rate": uniform(1e-2, 3e-1),
        "min_split_loss": uniform(0, 10),
        "max_depth": randint(3, 10),
        "subsample": uniform(0, 1),
        "objective": ["binary:logistic"], # Classification objective
        "eval_metric": ["aucpr", "error"]
    }
    model_grid_xgb = RandomizedSearchCV(model, param_distributions=params_xgb, n_jobs=-1, verbose=0, n_iter=10, cv=10)
    model_grid_xgb.fit(X_train, y_train)
    
    y_pred_train_xgb = model_grid_xgb.predict(X_train)
    xgboost_model = model_grid_xgb.best_estimator_
    xgboost_model_path = os.path.join(artifacts_dir, "lead_model_xgboost.json")
    xgboost_model.save_model(xgboost_model_path)
    model_results[xgboost_model_path] = classification_report(y_train, y_pred_train_xgb, output_dict=True)

    # --- Logistic Regression (LR) Training and MLflow Logging ---
    print("\n--- Training Logistic Regression (MLflow) ---")
    experiment_id = mlflow.get_experiment_by_name(experiment_name).experiment_id
    
    with mlflow.start_run(experiment_id=experiment_id) as run:
        model_lr = LogisticRegression()
        lr_model_path = os.path.join(artifacts_dir, "lead_model_lr.pkl")

        params_lr = {
                  'solver': ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
                  'penalty':  ["l2", "none", "l1", "elasticnet"], 
                  'C' : [100, 10, 1.0, 0.1, 0.01]
        }
        model_grid_lr = RandomizedSearchCV(model_lr, param_distributions=params_lr, verbose=0, n_iter=10, cv=3)
        model_grid_lr.fit(X_train, y_train)

        best_model_lr = model_grid_lr.best_estimator_
        y_pred_test_lr = model_grid_lr.predict(X_test)
        
        # Log artifacts (including the models and data)
        mlflow.log_metric('f1_score', f1_score(y_test, y_pred_test_lr))
        mlflow.log_artifacts(artifacts_dir, artifact_path=ARTIFACT_PATH) # Log all generated artifacts
        mlflow.log_param("data_version", "00000")
        
        # Store model for model interpretability (as in notebook)
        joblib.dump(value=model_lr, filename=lr_model_path)
            
        # Custom python model for predicting probability (for MLflow model registry)
        mlflow.pyfunc.log_model(ARTIFACT_PATH, python_model=lr_wrapper(best_model_lr))
        
        model_classification_report = classification_report(y_test, y_pred_test_lr, output_dict=True)
        model_results[lr_model_path] = model_classification_report
        
        # Save columns and model results artifacts
        with open(os.path.join(artifacts_dir, 'columns_list.json'), 'w+') as columns_file:
            json.dump({'column_names': list(X_train.columns)}, columns_file)
            
        with open(os.path.join(artifacts_dir, "model_results.json"), 'w+') as results_file:
            json.dump(model_results, results_file)
        
        print("--- Model Training Stage Complete ---")
        return experiment_name, run.info.run_id

def model_selection_and_deploy(experiment_name):
    print("\n--- Starting Model Selection and Deployment Stage ---")
    
    # 1. Get Best Run
    experiment_ids = [mlflow.get_experiment_by_name(experiment_name).experiment_id]
    experiment_best = mlflow.search_runs(
        experiment_ids=experiment_ids,
        order_by=["metrics.f1_score DESC"],
        max_results=1
    ).iloc[0]
    
    # 2. Determine Best Model from artifacts
    with open(os.path.join(ARTIFACTS_DIR, "model_results.json"), "r") as f:
        model_results = json.load(f)
        
    results_df = pd.DataFrame({model: val["weighted avg"] for model, val in model_results.items()}).T
    best_model_artifact_path = results_df.sort_values("f1-score", ascending=False).iloc[0].name
    
    # 3. Register Best Model
    run_id = experiment_best["run_id"]
    model_uri = "runs:/{run_id}/{artifact_path}".format(run_id=run_id, artifact_path=ARTIFACT_PATH)
    
    # Register the model saved from the best run (which was the LR model)
    model_details = mlflow.register_model(model_uri=model_uri, name=MODEL_NAME)
    wait_until_ready(model_details.name, model_details.version)
    
    # 4. Transition to Staging
    model_version = model_details.version
    client = MlflowClient()

    if dict(client.get_model_version(name=MODEL_NAME, version=model_version))['current_stage'] != 'Staging':
        client.transition_model_version_stage(
            name=MODEL_NAME,
            version=model_version,
            stage="Staging", 
            archive_existing_versions=True
        )
        wait_for_deployment(MODEL_NAME, model_version, 'Staging')
    else:
        print('Model already in staging')
        
    print("--- Model Selection and Deployment Stage Complete ---")

# --- 8. MAIN EXECUTION ---
if __name__ == "__main__":
    try:
        # 1. Data Processing
        data_frame = process_data(DATA_PATH, ARTIFACTS_DIR)
        
        # 2. Model Training and MLflow Logging
        experiment_name, run_id = train_and_select_model(data_frame, ARTIFACTS_DIR)
        
        # 3. Model Selection and Deployment
        model_selection_and_deploy(experiment_name)
        
        print(" Dagger Pipeline (train_model.py) finished successfully.")

    except Exception as e:
        print(f"An unrecoverable error occurred: {e}", file=sys.stderr)
        sys.exit(1)