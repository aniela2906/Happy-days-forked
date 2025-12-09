# train_xgb.py

import json
from pathlib import Path
import mlflow
import mlflow.xgboost

import pandas as pd
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from xgboost import XGBRFClassifier
from scipy.stats import uniform, randint


# ----------------------------
# Load GOLD dataset
# ----------------------------
def load_gold_data(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


# ----------------------------
# preprocessing for model training
# ----------------------------
def create_dummy_cols(df: pd.DataFrame, col: str) -> pd.DataFrame:
    dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
    df = pd.concat([df, dummies], axis=1)
    df = df.drop(columns=[col])
    return df


def prepare_features(df: pd.DataFrame):
    # Remove unused columns
    df = df.drop(columns=["lead_id", "customer_code", "date_part"], errors="ignore")

    # Categorical columns from notebook
    cat_cols = ["customer_group", "onboarding", "bin_source", "source"]

    cat_vars = df[cat_cols].copy()
    other_vars = df.drop(columns=cat_cols)

    # One-hot encode categorical columns
    for col in cat_vars.columns:
        cat_vars[col] = cat_vars[col].astype("category")
        cat_vars = create_dummy_cols(cat_vars, col)

    # Recombine encoded categoricals with numeric columns
    df = pd.concat([other_vars, cat_vars], axis=1)

    # Convert all columns to float
    df = df.astype("float64")

    return df


# ----------------------------
# Training XGBoost + MLflow
# ----------------------------
def train_xgboost_model(df: pd.DataFrame, artifacts_dir="artifacts", experiment_name="xgb_experiment"):

    # MLflow experiment
    mlflow.set_experiment(experiment_name)

    # Split target and features
    y = df["lead_indicator"]
    X = df.drop(columns=["lead_indicator"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    # Model + hyperparameter search
    model = XGBRFClassifier(random_state=42)

    params = {
        "learning_rate": uniform(1e-2, 3e-1),
        "min_split_loss": uniform(0, 10),
        "max_depth": randint(3, 10),
        "subsample": uniform(0, 1),
        "objective": ["reg:squarederror", "binary:logistic", "reg:logistic"],
        "eval_metric": ["aucpr", "error"],
    }

    model_grid = RandomizedSearchCV(
        model,
        param_distributions=params,
        n_iter=10,
        n_jobs=-1,
        verbose=3,
        cv=10,
    )

    # ----------------------------
    # MLflow run
    # ----------------------------
    with mlflow.start_run() as run:

        mlflow.log_param("model_type", "XGBRFClassifier")

        model_grid.fit(X_train, y_train)
        best_model = model_grid.best_estimator_

        # predictions
        y_pred_train = best_model.predict(X_train)
        y_pred_test = best_model.predict(X_test)

        # metrics
        metrics = {
            "accuracy_train": float(accuracy_score(y_train, y_pred_train)),
            "accuracy_test": float(accuracy_score(y_test, y_pred_test)),
        }

        mlflow.log_params(model_grid.best_params_)
        mlflow.log_metrics(metrics)

        # detailed reports
        classifications = {
            "train_report": classification_report(y_train, y_pred_train, output_dict=True),
            "test_report": classification_report(y_test, y_pred_test, output_dict=True),
            "confusion_matrix_test": confusion_matrix(y_test, y_pred_test).tolist(),
        }

        # save metrics JSON
        Path(artifacts_dir).mkdir(exist_ok=True)
        metrics_path = Path(artifacts_dir) / "xgboost_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(classifications, f, indent=4)

        mlflow.log_artifact(str(metrics_path))

        # ----------------------------
        # Log model to MLflow
        # ----------------------------
        mlflow.xgboost.log_model(
            xgb_model=best_model,
            artifact_path="model"
        )

        return best_model, classifications, run.info.run_id


# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":
    df = load_gold_data("artifacts/train_data_gold.csv")
    df = prepare_features(df)

    model, metrics, run_id = train_xgboost_model(df)

    print("XGBoost model training completed.")
    print("MLflow Run ID:", run_id)
