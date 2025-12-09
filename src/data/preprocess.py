import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler
import joblib


# Feature selection lists (copied from notebook logic)

DROP_COLS_1 = [
    "is_active", 
    "marketing_consent", 
    "first_booking", 
    "existing_customer", 
    "last_seen"
]

DROP_COLS_2 = [
    "domain", 
    "country", 
    "visited_learn_more_before_booking", 
    "visited_faq"
]

CAT_COLS = [
    "lead_id", 
    "lead_indicator", 
    "customer_group",
    "onboarding", 
    "source", 
    "customer_code"
]


# =====================================================================
# Main preprocessing function
# =====================================================================

def preprocess_data(df: pd.DataFrame, artifacts_dir: str = "./artifacts") -> pd.DataFrame:

    # 1) FEATURE SELECTION — drop irrelevant columns

    df = df.drop(columns=DROP_COLS_1, errors="ignore")
    df = df.drop(columns=DROP_COLS_2, errors="ignore")

    # 2) CLEANING: replace blank strings with NaN

    df["lead_indicator"].replace("", np.nan, inplace=True)
    df["lead_id"].replace("", np.nan, inplace=True)
    df["customer_code"].replace("", np.nan, inplace=True)

    # 3) REMOVE ROWS WITH MISSING TARGET

    df = df.dropna(subset=["lead_indicator"])

    # 4) KEEP ONLY SIGNUP SOURCE

    df = df[df["source"] == "signup"]

    # 5) CREATE CATEGORICAL COLUMNS (convert dtype)

    for col in CAT_COLS:
        if col in df.columns:
            df[col] = df[col].astype("object")

    # 6) SEPARATE CONTINUOUS vs CATEGORICAL COLUMNS

    cont_vars = df.loc[:, (df.dtypes == "float64") | (df.dtypes == "int64")]
    cat_vars = df.loc[:, df.dtypes == "object"]

    # 7) HANDLE OUTLIERS (Z-score clipping ±2 std)

    cont_vars = cont_vars.apply(
        lambda x: x.clip(lower=x.mean() - 2 * x.std(),
                         upper=x.mean() + 2 * x.std())
    )

    # 8) IMPUTE MISSING VALUES
    #       - continuous → mean
    #       - categorical → mode


    # continuous
    cont_vars = cont_vars.apply(lambda x: x.fillna(x.mean()))

    # categorical
    cat_vars = cat_vars.fillna(cat_vars.mode().iloc[0])

    # special rule for customer_code
    if "customer_code" in cat_vars.columns:
        cat_vars.loc[cat_vars["customer_code"].isna(), "customer_code"] = "None"

    # 9) STANDARDIZATION (MinMaxScaler) — continuous only

    scaler_path = Path(artifacts_dir) / "scaler.pkl"
    scaler = MinMaxScaler()
    scaler.fit(cont_vars)

    joblib.dump(scaler, scaler_path)

    cont_vars = pd.DataFrame(
        scaler.transform(cont_vars),
        columns=cont_vars.columns
    )

    # 10) COMBINE BACK cat + cont dataframes

    cont_vars = cont_vars.reset_index(drop=True)
    cat_vars = cat_vars.reset_index(drop=True)
    df = pd.concat([cat_vars, cont_vars], axis=1)

    # 11) BINNING OBJECT COLUMNS (bin_source)

    if "source" in df.columns:
        values_list = ["li", "organic", "signup", "fb"]

        df["bin_source"] = df["source"]
        df.loc[~df["source"].isin(values_list), "bin_source"] = "Others"

        mapping = {
            "li": "socials",
            "fb": "socials",
            "organic": "group1",
            "signup": "group1"
        }

        df["bin_source"] = df["source"].map(mapping)

    # 12) SAVE FINAL GOLD DATASET

    gold_path = Path(artifacts_dir) / "train_data_gold.csv"
    df.to_csv(gold_path, index=False)

    # also save column list for drift detection if needed
    column_meta = {"columns": list(df.columns)}
    with open(Path(artifacts_dir) / "columns_list.json", "w") as f:
        json.dump(column_meta, f)

    return df
