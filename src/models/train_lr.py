# train_logistic_regression.py

import json
import joblib
import mlflow
import mlflow.pyfunc

import pandas as pd
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


# ----------------------------
# MLflow Wrapper
# ----------------------------
class LRWrapper(mlflow.pyfunc.PythonModel):
    """Custom wrapper so MLflow can serve probability predictions."""
    
    def __init__(self, model):
        self.model = model

    def predict(self, context, model_input):
        return self.model.predict_proba(model_input)[:, 1]


# ----------------------------
# Load GOLD dataset
# ----------------------------
def load_gold_data(path: str | Path) -> pd.DataFrame:
    return pd.read_csv(path)


# ----------------------------
# Prepare features 
# ----------------------------
def create_dummy_cols(df: pd.DataFrame, col: str) -> pd.DataFrame:
    dummies = pd.get_dummies(df[col], prefix=col, drop_first=True)
    df = pd.concat([df, dummies], axis=1)
    df = df.drop(columns=[col])
    return df


def prepare_features(df: pd.DataFrame) -> pd.DataFrame:

    df = df.drop(columns=["lead_id", "customer_code", "date_part"], errors="ignore")

    cat_cols = ["customer_group", "onboarding", "bin_source", "source"]

    cat_vars = df[cat_cols].copy()
    other_vars = df.drop(columns=cat_cols)

    for col in cat_vars.columns:
        cat_vars[col] = cat_vars[col].astype("category")
        cat_vars = create_dummy_cols(cat_vars, col)

    df = pd.concat([other_vars, cat_vars], axis=1)

    # convert all to float
    df = df.astype("float64")

    return df


# ----------------------------
# Train Logistic Regression + MLflow
# ----------------------------
def train_logistic_regression(df: pd.DataFrame, artifacts_dir="artifacts", experiment_name="lr_experiment"):

    mlflow.set_experiment(experiment_name)

    # Split features
    y = df["lead_indicator"]
    X = df.drop(columns=["lead_indicator"])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )

    params = {
        "solver": ["newton-cg", "lbfgs", "liblinear", "sag", "saga"],
        "penalty": ["none", "l1", "l2", "elasticnet"],
        "C": [100, 10, 1.0, 0.1, 0.01]
    }

    model = LogisticRegression(max_iter=500)

    search = RandomizedSearchCV(
        model,
        param_distributions=params,
        n_iter=10,
        verbose=3,
        cv=3
    )

    # ----------------------------
    # MLflow run
    # ----------------------------
    with mlflow.start_run() as run:

        mlflow.log_param("model_type", "LogisticRegression")

        search.fit(X_train, y_train)
        best_model = search.best_estimator_

        # predictions
        y_pred_train = best_model.predict(X_train)
        y_pred_test = best_model.predict(X_test)

        # metrics
        metrics = {
            "accuracy_train": float(accuracy_score(y_train, y_pred_train)),
            "accuracy_test": float(accuracy_score(y_test, y_pred_test)),
        }

        mlflow.log_params(search.best_params_)
        mlflow.log_metrics(metrics)

        # detailed reports
        classifications = {
            "train_report": classification_report(y_train, y_pred_train, output_dict=True),
            "test_report": classification_report(y_test, y_pred_test, output_dict=True),
            "confusion_matrix_test": confusion_matrix(y_test, y_pred_test).tolist(),
        }

        # save metrics JSON
        Path(artifacts_dir).mkdir(exist_ok=True)
        metrics_path = Path(artifacts_dir) / "lr_metrics.json"
        with open(metrics_path, "w") as f:
            json.dump(classifications, f, indent=4)

        mlflow.log_artifact(str(metrics_path))

        # save pickled model
        model_path = Path(artifacts_dir) / "lead_model_lr.pkl"
        joblib.dump(best_model, model_path)

        # log MLflow pyfunc model
        mlflow.pyfunc.log_model(
            artifact_path="model",
            python_model=LRWrapper(best_model)
        )

        return best_model, classifications, run.info.run_id


# ----------------------------
# MAIN
# ----------------------------
if __name__ == "__main__":

    df = load_gold_data("artifacts/train_data_gold.csv")
    df = prepare_features(df)

    model, metrics, run_id = train_logistic_regression(df)

    print("\nLogistic Regression training complete.")
    print("MLflow Run ID:", run_id)
