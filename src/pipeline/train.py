import json
from pathlib import Path

from src.data.load_data import load_raw_data
from src.data.preprocess import preprocess_data
from src.models.train_xgb import train_xgboost_model, prepare_features as prepare_xgb_features
from src.models.train_lr import train_logistic_regression, prepare_features as prepare_lr_features

ARTIFACTS_DIR = Path("artifacts")
RAW_DATA_PATH = Path("notebooks/artifacts/raw_data.csv")
GOLD_DATA_PATH = ARTIFACTS_DIR / "train_data_gold.csv"


def main():

    # ============================================================
    # 1. LOAD RAW DATA
    # ============================================================
    print("Loading raw dataset...")
    df_raw = load_raw_data(RAW_DATA_PATH)

    # ============================================================
    # 2. PREPROCESS → generate GOLD dataset
    # ============================================================
    print("Preprocessing data...")
    df_gold = preprocess_data(df_raw)
    df_gold.to_csv(GOLD_DATA_PATH, index=False)

    # ============================================================
    # 3. PREPARE DATA FOR BOTH MODELS
    #    (each one has its own encoding rules)
    # ============================================================
    print("Preparing feature matrices...")

    df_xgb = prepare_xgb_features(df_gold.copy())
    df_lr = prepare_lr_features(df_gold.copy())

    # ============================================================
    # 4. TRAIN BOTH MODELS
    # ============================================================
    print("\nTraining XGBoost...")
    xgb_model, xgb_metrics, xgb_run_id = train_xgboost_model(df_xgb)

    print("\nTraining Logistic Regression...")
    lr_model, lr_metrics, lr_run_id = train_logistic_regression(df_lr)

    # ============================================================
    # 5. MODEL SELECTION BASED ON F1-SCORE
    # ============================================================
    xgb_f1 = xgb_metrics["test_report"]["weighted avg"]["f1-score"]
    lr_f1 = lr_metrics["test_report"]["weighted avg"]["f1-score"]


    best_model_name = None

    if xgb_f1 >= lr_f1:
        best_model_name = "xgboost"
        best_f1 = xgb_f1
        best_run = xgb_run_id
    else:
        best_model_name = "logistic_regression"
        best_f1 = lr_f1
        best_run = lr_run_id

    print("\n=========================================")
    print(f" BEST MODEL SELECTED: {best_model_name.upper()}")
    print(f" F1-SCORE: {best_f1:.4f}")
    print(f" MLflow Run ID: {best_run}")
    print("=========================================\n")

    # ============================================================
    # 6. SAVE METADATA
    # ============================================================
    metadata = {
        "best_model": best_model_name,
        "best_f1": best_f1,
        "best_run_id": best_run,
        "xgb_f1": xgb_f1,
        "lr_f1": lr_f1
    }

    metadata_path = ARTIFACTS_DIR / "model_selection.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print("Saved model selection metadata to artifacts/model_selection.json")

    print("\nPipeline completed successfully.")


if __name__ == "__main__":
    main()
