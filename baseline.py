# baseline_weather_anomaly.py
# Deterministic baseline for: per-numeric-column stats + z-score + anomaly flag
# OUTPUT: RESULT dict (JSON-serializable)

import json
from datetime import datetime
import numpy as np
import pandas as pd


def baseline_weather_analysis(file_path: str) -> dict:
    df = pd.read_csv(file_path)

    # Pick the "today"/latest row
    latest_date_str = None
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce")
        latest_ts = df["date"].max()
        if pd.isna(latest_ts):
            today_row = df.tail(1)
        else:
            latest_date_str = str(latest_ts.date())
            today_row = df.loc[df["date"] == latest_ts].tail(1)
            if today_row.empty:
                today_row = df.tail(1)
    else:
        today_row = df.tail(1)

    # Numeric columns (exclude obvious id column)
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c != "id"]

    results = {}

    for col in numeric_cols:
        series = pd.to_numeric(df[col], errors="coerce")  # ensure numeric with NaNs
        mean_val = series.mean()
        median_val = series.median()
        std_val = series.std(ddof=1)  # pandas default
        min_val = series.min()
        max_val = series.max()

        today_val = pd.to_numeric(today_row[col], errors="coerce").iloc[0]

        z = np.nan
        if pd.notna(std_val) and std_val > 0 and pd.notna(today_val):
            z = (today_val - mean_val) / std_val

        anomaly = None
        if pd.notna(z):
            if z >= 2:
                anomaly = "high"
            elif z <= -2:
                anomaly = "low"

        # Make JSON-friendly (convert numpy/pandas scalars -> Python)
        def py(x):
            if pd.isna(x):
                return None
            if isinstance(x, (np.generic,)):
                return x.item()
            return float(x) if isinstance(x, (np.floating, float)) else x

        results[col] = {
            "mean": py(mean_val),
            "median": py(median_val),
            "std": py(std_val),
            "min": py(min_val),
            "max": py(max_val),
            "today": py(today_val),
            "z_score": py(z) if z is not None else None,
            "anomaly": anomaly,
        }

        if latest_date_str is not None:
            results[col]["latest_date"] = latest_date_str

    return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("file_path", help="Path to CSV")
    args = parser.parse_args()

    RESULT = baseline_weather_analysis(args.file_path)
    print(json.dumps(RESULT, indent=2, sort_keys=True))
