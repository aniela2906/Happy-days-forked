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
from sklearn.metrics import classification_report, f1_score
from scipy.stats import uniform, randint

# --- 3. MLFLOW/DEPLOYMENT LIBRARIES ---
import mlflow
import mlflow.sklearn
import mlflow.pyfunc
from mlflow.tracking.client import MlflowClient
# NOTE: Removed unused entities imports to maintain clean code

# Set Pandas options and filter warnings (from notebook)
warnings.filterwarnings('ignore')
pd.set_option('display.float_format', lambda x: "%.3f" % x)

# --- 4. CONSTANTS / FILE PATHS ---
INPUT_DIR = "/python" 
ARTIFACTS_DIR = os.path.join(INPUT_DIR, "artifacts")
DATA_PATH = os.path.join(ARTIFACTS_DIR, "raw_data.csv")
GOLD_DATA_PATH = os.path.join(ARTIFACTS_DIR, "train_data_gold.csv")
MODEL_NAME = "lead_model"
ARTIFACT_PATH = "model" 

# --- 5. HELPER FUNCTIONS (Only core data helpers remain) ---
# NOTE: Deployment helpers (wait_until_ready, wait_for_deployment) are removed 
# as they belong to the test_inference_and_deploy.py file.

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
        x = x.fillna(x.mode()[0])
    return x

def create_dummy_cols(df, col):
    """Create one-hot encoding columns in the data."""
    df_dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
    new_df = pd.concat([df, df_dummies], axis=1)
    new_df = new_df.drop(col, axis=1)
    return new_df

class lr_wrapper(mlflow.pyfunc.PythonModel):
    """MLflow wrapper for Logistic Regression to predict probability."""
    def __init__(self, model):
        self.model = model
    
    def predict(self, context, model_input):
        return self.model.predict_proba(model_input)[:, 1]

# --- 6. DATA PROCESSING STAGE ---
def process_data(data_path, artifacts_dir):
    print("--- Starting Data Processing Stage ---")

    print("--- Starting Data Processing Stage ---")

    # --- ADDED DEBUGGING BLOCK ---
    print(f"DEBUG: INPUT_DIR set to: {INPUT_DIR}")
    print(f"DEBUG: ARTIFACTS_DIR set to: {ARTIFACTS_DIR}")
    print(f"DEBUG: Looking for data at: {DATA_PATH}")
    
    # Critical File Check
    if not os.path.exists(DATA_PATH):
        # This will print the exact reason for the failure inside the Dagger log
        print(f"FATAL ERROR: Data file not found at expected path: {DATA_PATH}")
        # Reraise a specific error or exit to ensure the job fails clearly
        raise FileNotFoundError(f"Required data file is missing: {DATA_PATH}")
    # --- END DEBUGGING BLOCK ---

    os.makedirs(artifacts_dir, exist_ok=True)
    data = pd.read_csv(data_path)
    
    min_date_str = "2024-01-01"
    max_date_str = "2024-01-31"
    max_date = pd.to_datetime(max_date_str).date()
    min_date = pd.to_datetime(min_date_str).date()
    
    data["date_part"] = pd.to_datetime(data["date_part"]).dt.date
    data = data[(data["date_part"] >= min_date) & (data["date_part"] <= max_date)]
    
    date_limits = {"min_date": str(data["date_part"].min()), "max_date": str(data["date_part"].max())}
    with open(os.path.join(artifacts_dir, "date_limits.json"), "w") as f:
        json.dump(date_limits, f)
    
    data = data.drop(
        ["is_active", "marketing_consent", "first_booking", "existing_customer", "last_seen"],
        axis=1
    )
    data = data.drop(
        ["domain", "country", "visited_learn_more_before_booking", "visited_faq"],
        axis=1
    )
    
    data["lead_indicator"].replace("", np.nan, inplace=True)
    data["lead_id"].replace("", np.nan, inplace=True)
    data["customer_code"].replace("", np.nan, inplace=True)
    data = data.dropna(axis=0, subset=["lead_indicator"])
    data = data.dropna(axis=0, subset=["lead_id"])
    data = data[data.source == "signup"] 
    
    vars_to_convert = ["lead_id", "lead_indicator", "customer_group", "onboarding", "source", "customer_code"]
    for col in vars_to_convert:
        data[col] = data[col].astype("object")

    cont_vars = data.loc[:, ((data.dtypes=="float64")|(data.dtypes=="int64"))]
    cat_vars = data.loc[:, (data.dtypes=="object")]
    
    cont_vars = cont_vars.apply(lambda x: x.clip(lower = (x.mean()-2*x.std()),
                                                 upper = (x.mean()+2*x.std())))
    outlier_summary = cont_vars.apply(describe_numeric_col).T
    outlier_summary.to_csv(os.path.join(artifacts_dir, 'outlier_summary.csv'))
    
    cat_missing_impute = cat_vars.mode(numeric_only=False, dropna=True)
    cat_missing_impute.to_csv(os.path.join(artifacts_dir, "cat_missing_impute.csv"))
    
    cont_vars = cont_vars.apply(impute_missing_values)
    
    cat_vars.loc[cat_vars['customer_code'].isna(),'customer_code'] = 'None'
    cat_vars = cat_vars.apply(impute_missing_values) 

    scaler = MinMaxScaler()
    scaler.fit(cont_vars)
    joblib.dump(value=scaler, filename=os.path.join(artifacts_dir, "scaler.pkl"))
    cont_vars = pd.DataFrame(scaler.transform(cont_vars), columns=cont_vars.columns)

    cont_vars = cont_vars.reset_index(drop=True)
    cat_vars = cat_vars.reset_index(drop=True)
    data = pd.concat([cat_vars, cont_vars], axis=1)
    
    data_columns = list(data.columns)
    with open(os.path.join(artifacts_dir, 'columns_drift.json'), 'w+') as f:           
        json.dump(data_columns, f)
        data.to_csv(os.path.join(artifacts_dir, 'training_data.csv'), index=False)

    mapping = {'li' : 'socials', 'fb' : 'socials', 'organic': 'group1', 'signup': 'group1'}
    data['bin_source'] = data['source'].map(mapping)
    
    data.to_csv(GOLD_DATA_PATH, index=False)
    
    print("--- Data Processing Stage Complete ---")
    return data

# --- 7. MODEL TRAINING AND ARTIFACT CREATION STAGE ---
def train_and_select_model(data, artifacts_dir):
    print("--- Starting Model Training Stage ---")
    
    current_date = datetime.datetime.now().strftime("%Y_%B_%d")
    experiment_name = current_date
    os.makedirs("mlruns", exist_ok=True)
    os.makedirs("mlruns/.trash", exist_ok=True)
    mlflow.set_experiment(experiment_name)
    mlflow.sklearn.autolog(log_input_examples=True, log_models=False)
    
    data = data.drop(["lead_id", "customer_code", "date_part"], axis=1)
    cat_cols = ["customer_group", "onboarding", "bin_source", "source"]
    cat_vars = data[cat_cols]
    other_vars = data.drop(cat_cols, axis=1)
    
    for col in cat_vars:
        cat_vars[col] = cat_vars[col].astype("category")
        cat_vars = create_dummy_cols(cat_vars, col)
    
    data = pd.concat([other_vars, cat_vars], axis=1)
    
    for col in data:
        data[col] = data[col].astype("float64")
        
    y = data["lead_indicator"]
    X = data.drop(["lead_indicator"], axis=1)
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, random_state=42, test_size=0.15, stratify=y
    )
    
    model_results = {}
    
    # --- CRITICAL: SAVE X_TEST and Y_TEST for the SEPARATE INFERENCE JOB ---
    X_test.to_csv(os.path.join(artifacts_dir, "X_test.csv"), index=False)
    y_test.to_csv(os.path.join(artifacts_dir, "y_test.csv"), index=False)
    # ------------------------------------------------------------------
    
    # --- XGBoost Training ---
    print("\n--- Training XGBoost ---")
    model = XGBRFClassifier(random_state=42)
    params_xgb = {
        "learning_rate": uniform(1e-2, 3e-1),
        "min_split_loss": uniform(0, 10),
        "max_depth": randint(3, 10),
        "subsample": uniform(0, 1),
        "objective": ["binary:logistic"],
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
        
        input_example = X_train.head(5) 
        
        mlflow.log_metric('f1_score', f1_score(y_test, y_pred_test_lr))
        mlflow.log_artifacts(artifacts_dir, artifact_path=ARTIFACT_PATH) 
        mlflow.log_param("data_version", "00000")
        
        joblib.dump(value=best_model_lr, filename=lr_model_path)
            
        mlflow.pyfunc.log_model(
            ARTIFACT_PATH, 
            python_model=lr_wrapper(best_model_lr),
            input_example=input_example
        )
        
        model_classification_report = classification_report(y_test, y_pred_test_lr, output_dict=True)
        model_results[lr_model_path] = model_classification_report
        
        with open(os.path.join(artifacts_dir, 'columns_list.json'), 'w+') as columns_file:
            json.dump({'column_names': list(X_train.columns)}, columns_file)
            
        with open(os.path.join(artifacts_dir, "model_results.json"), 'w+') as results_file:
            json.dump(model_results, results_file)
        
        print("--- Model Training Stage Complete ---")

        #Write Run ID to a file in the artifacts directory
        with open(os.path.join(artifacts_dir, "mlflow_run_id.txt"), "w") as f:
            f.write(run.info.run_id)

        return experiment_name, run.info.run_id

# --- 8. MAIN EXECUTION ---
if __name__ == "__main__":
    try:
        # 1. Data Processing
        data_frame = process_data(DATA_PATH, ARTIFACTS_DIR)
        
        # 2. Model Training and Artifact Creation
        experiment_name, run_id = train_and_select_model(data_frame, ARTIFACTS_DIR)
        
        print("\Dagger Pipeline (train_model.py) finished successfully.")

    except Exception as e:
        print(f"\An unrecoverable error occurred: {e}", file=sys.stderr)
        sys.exit(1)