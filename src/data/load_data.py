import pandas as pd
import json
import datetime
from pathlib import Path
import os


def load_raw_data(path: str | Path, min_date=None, max_date=None) -> pd.DataFrame:
    #load raw CSV, convert dates, filter by range, save metadata.

    df = pd.read_csv(path)

    # convert date column
    df["date_part"] = pd.to_datetime(df["date_part"]).dt.date

    # determine bounds
    if max_date:
        max_date = pd.to_datetime(max_date).date()
    else:
        max_date = datetime.date.today()

    min_date = pd.to_datetime(min_date).date() if min_date else df["date_part"].min()

    # apply filtering
    df = df[(df["date_part"] >= min_date) & (df["date_part"] <= max_date)]

    # metadata output
    
    os.makedirs("artifacts", exist_ok=True)
    
    date_limits = {"min_date": str(min_date), "max_date": str(max_date)}
    with open("artifacts/date_limits.json", "w") as f:
        json.dump(date_limits, f)

    return df
