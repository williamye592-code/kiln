
"""
Train baseline models for future CO-deviation risk scoring.

Current v1:
- Random Forest Regressor predicts risk_target.
- Output dataset contains an additional rf_score column.
- The generated dataset can be passed to the warning policy evaluator.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


DEFAULT_FEATURE_COLUMNS = [
    "co_raw",
    "co_baseline",
    "co_relative",
    "drum_level_1",
    "drum_level_2",
    "drum_level_3",
    "main_motor_current",
    "south_kiln_main_motor_freq",
    "whrb_drum_pressure",
    "whrb_outlet_pressure",
    "esp_inlet_pressure",
    "esp_outlet_pressure",
    "fan_outlet_pressure",
    "disc_feeder_freq",
    "kiln_tail_temp",
    "kiln_head_temp",
    "whrb_inlet_temp",
    "esp_inlet_temp",
    "esp_outlet_temp",
    "fan_outlet_temp",
]


def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    ts = pd.to_datetime(df["timestamp"])

    seconds = (
        ts.dt.hour * 3600
        + ts.dt.minute * 60
        + ts.dt.second
    ).astype(float)

    day_seconds = 24.0 * 3600.0

    df["time_sin"] = np.sin(2.0 * np.pi * seconds / day_seconds)
    df["time_cos"] = np.cos(2.0 * np.pi * seconds / day_seconds)

    return df


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)

    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return {
            "n": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "r2": np.nan,
        }

    mse = mean_squared_error(y_true, y_pred)

    return {
        "n": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        default="outputs/processed/risk_dataset.csv",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="outputs/predictions/risk_dataset_with_scores.csv",
    )
    parser.add_argument(
        "--metrics_path",
        type=str,
        default="outputs/metrics/rf_baseline_metrics.csv",
    )
    parser.add_argument("--n_estimators", type=int, default=200)
    parser.add_argument("--max_depth", type=int, default=12)
    parser.add_argument("--min_samples_leaf", type=int, default=20)
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--n_jobs", type=int, default=-1)
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    metrics_path = Path(args.metrics_path)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path, parse_dates=["timestamp"])
    df = add_time_features(df)

    feature_cols = DEFAULT_FEATURE_COLUMNS + ["time_sin", "time_cos"]

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    valid = df[df["is_valid_sample"] == True].copy()

    train = valid[valid["split"] == "train"].copy()
    val = valid[valid["split"] == "val"].copy()
    test = valid[valid["split"] == "test"].copy()

    train = train.dropna(subset=feature_cols + ["risk_target"])
    val = val.dropna(subset=feature_cols + ["risk_target"])
    test = test.dropna(subset=feature_cols + ["risk_target"])

    print(f"Train rows: {len(train)}")
    print(f"Val rows:   {len(val)}")
    print(f"Test rows:  {len(test)}")

    X_train = train[feature_cols].to_numpy()
    y_train = train["risk_target"].to_numpy()

    model = RandomForestRegressor(
        n_estimators=args.n_estimators,
        max_depth=args.max_depth,
        min_samples_leaf=args.min_samples_leaf,
        random_state=args.random_state,
        n_jobs=args.n_jobs,
    )

    print("Training Random Forest...")
    model.fit(X_train, y_train)

    df["rf_score"] = np.nan

    metrics_rows = []

    for split_name, part in [("train", train), ("val", val), ("test", test)]:
        X = part[feature_cols].to_numpy()
        y = part["risk_target"].to_numpy()
        pred = model.predict(X)

        df.loc[part.index, "rf_score"] = pred

        m = regression_metrics(y, pred)
        m["model"] = "random_forest"
        m["split"] = split_name
        metrics_rows.append(m)

    metrics = pd.DataFrame(metrics_rows)

    # Feature importances
    importances = pd.DataFrame(
        {
            "feature": feature_cols,
            "importance": model.feature_importances_,
        }
    ).sort_values("importance", ascending=False)

    importance_path = metrics_path.parent / "rf_feature_importance.csv"

    df.to_csv(output_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    importances.to_csv(importance_path, index=False)

    meta = {
        "model": "RandomForestRegressor",
        "input_path": str(input_path),
        "output_path": str(output_path),
        "metrics_path": str(metrics_path),
        "feature_columns": feature_cols,
        "n_estimators": args.n_estimators,
        "max_depth": args.max_depth,
        "min_samples_leaf": args.min_samples_leaf,
        "random_state": args.random_state,
    }

    meta_path = metrics_path.parent / "rf_baseline_metadata.json"
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print(f"Saved scored dataset: {output_path}")
    print(f"Saved RF metrics: {metrics_path}")
    print(f"Saved feature importance: {importance_path}")
    print()
    print(metrics)
    print()
    print(importances.head(20))


if __name__ == "__main__":
    main()
