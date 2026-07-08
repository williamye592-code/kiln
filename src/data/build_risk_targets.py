
"""
Build relative CO signals and future CO-deviation risk targets.

This script converts raw daily kiln CSV files into a risk-aware dataset:

raw CO
→ past-only baseline
→ relative CO deviation
→ future high CO deviation
→ risk target
→ train / val / test split
→ provisional event labels
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml


COLUMN_NAMES = [
    "row_id",
    "time_of_day",
    "co_raw",
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

SENSOR_COLUMNS = COLUMN_NAMES[2:]


def load_config(config_path: str) -> dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_daily_csv(path: Path, encoding: str) -> pd.DataFrame:
    df = pd.read_csv(path, header=None, encoding=encoding)

    if df.shape[1] != len(COLUMN_NAMES):
        raise ValueError(
            f"{path} has {df.shape[1]} columns, expected {len(COLUMN_NAMES)}."
        )

    df.columns = COLUMN_NAMES

    date_str = path.stem
    df["date"] = date_str
    df["source_file"] = path.name

    df["timestamp"] = pd.to_datetime(
        date_str + " " + df["time_of_day"].astype(str),
        errors="coerce",
    )

    for col in ["row_id"] + SENSOR_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    return df


def infer_base_interval_seconds(df: pd.DataFrame) -> float:
    diffs = df["timestamp"].sort_values().diff().dt.total_seconds()
    diffs = diffs[(diffs > 0) & (diffs <= 60)]
    if len(diffs) == 0:
        return 10.0
    return float(diffs.median())


def add_session_ids(df: pd.DataFrame, session_gap_minutes: float) -> pd.DataFrame:
    df = df.sort_values("timestamp").reset_index(drop=True)
    gap_seconds = df["timestamp"].diff().dt.total_seconds()

    new_session = (
        gap_seconds.isna()
        | (gap_seconds < 0)
        | (gap_seconds > session_gap_minutes * 60.0)
        | (df["source_file"] != df["source_file"].shift(1))
    )

    df["gap_seconds"] = gap_seconds
    df["session_id"] = new_session.cumsum().astype(int) - 1
    return df


def add_past_baseline(
    df: pd.DataFrame,
    past_baseline_minutes: float,
    min_past_minutes: float,
    base_interval_seconds: float,
) -> pd.DataFrame:
    window = f"{int(past_baseline_minutes)}min"
    min_periods = max(
        1,
        int(round((min_past_minutes * 60.0) / base_interval_seconds)),
    )

    parts = []

    for _, g in df.groupby("session_id", sort=False):
        g = g.sort_values("timestamp").copy()
        gi = g.set_index("timestamp")

        shifted_co = gi["co_raw"].shift(1)
        baseline = shifted_co.rolling(
            window=window,
            min_periods=min_periods,
        ).median()

        g["co_baseline"] = baseline.to_numpy()
        g["co_relative"] = g["co_raw"] - g["co_baseline"]
        parts.append(g)

    return pd.concat(parts, axis=0).sort_values("timestamp").reset_index(drop=True)


def future_window_quantile(
    values: np.ndarray,
    start_steps: int,
    end_steps: int,
    q: float,
) -> np.ndarray:
    values = values.astype(float)
    n = len(values)
    out = np.full(n, np.nan, dtype=float)

    if n <= start_steps:
        return out

    end_steps = max(start_steps, end_steps)

    padded = np.concatenate([values, np.full(end_steps, np.nan)])
    windows = np.lib.stride_tricks.sliding_window_view(
        padded,
        end_steps + 1,
    )[:n]

    future_band = windows[:, start_steps : end_steps + 1]
    valid = np.sum(~np.isnan(future_band), axis=1) > 0

    if valid.any():
        out[valid] = np.nanquantile(future_band[valid], q, axis=1)

    return out


def add_future_risk_target(
    df: pd.DataFrame,
    future_start_minutes: float,
    future_end_minutes: float,
    future_quantile: float,
    base_interval_seconds: float,
) -> pd.DataFrame:
    start_steps = max(
        1,
        int(round((future_start_minutes * 60.0) / base_interval_seconds)),
    )
    end_steps = max(
        start_steps,
        int(round((future_end_minutes * 60.0) / base_interval_seconds)),
    )

    parts = []

    for _, g in df.groupby("session_id", sort=False):
        g = g.sort_values("timestamp").copy()

        future_high_raw = future_window_quantile(
            g["co_raw"].to_numpy(),
            start_steps=start_steps,
            end_steps=end_steps,
            q=future_quantile,
        )

        g["future_high_raw"] = future_high_raw
        g["future_high_relative"] = g["future_high_raw"] - g["co_baseline"]
        g["risk_target"] = g["future_high_relative"]

        parts.append(g)

    out = pd.concat(parts, axis=0).sort_values("timestamp").reset_index(drop=True)

    out["future_start_steps"] = start_steps
    out["future_end_steps"] = end_steps
    out["future_start_minutes"] = future_start_minutes
    out["future_end_minutes"] = future_end_minutes
    out["future_quantile"] = future_quantile

    return out


def assign_split(
    df: pd.DataFrame,
    train_end_date: str,
    val_end_date: str,
    test_start_date: str,
) -> pd.DataFrame:
    date = pd.to_datetime(df["date"])

    train_end = pd.to_datetime(train_end_date)
    val_end = pd.to_datetime(val_end_date)
    test_start = pd.to_datetime(test_start_date)

    split = np.full(len(df), "unused", dtype=object)

    split[date <= train_end] = "train"
    split[(date > train_end) & (date <= val_end)] = "val"
    split[date >= test_start] = "test"

    df["split"] = split
    return df


def add_event_ids(
    df: pd.DataFrame,
    event_threshold: float,
    event_merge_gap_minutes: float,
    min_event_duration_minutes: float,
) -> pd.DataFrame:
    df = df.sort_values("timestamp").copy()
    df["is_event_point"] = df["co_relative"] >= event_threshold

    event_id = np.full(len(df), -1, dtype=int)
    current_event = -1
    last_event_time = None

    for i, row in enumerate(df.itertuples(index=False)):
        if not bool(row.is_event_point):
            continue

        ts = row.timestamp

        if last_event_time is None:
            current_event += 1
        else:
            gap_min = (ts - last_event_time).total_seconds() / 60.0
            if gap_min > event_merge_gap_minutes:
                current_event += 1

        event_id[i] = current_event
        last_event_time = ts

    df["event_id"] = event_id

    valid_ids = []

    for eid, g in df[df["event_id"] >= 0].groupby("event_id"):
        duration_min = (
            g["timestamp"].max() - g["timestamp"].min()
        ).total_seconds() / 60.0

        if duration_min >= min_event_duration_minutes:
            valid_ids.append(eid)

    df.loc[~df["event_id"].isin(valid_ids), "event_id"] = -1
    df["is_event_point"] = df["event_id"] >= 0

    return df


def make_summary(df: pd.DataFrame, base_interval_seconds: float) -> pd.DataFrame:
    rows = []

    for split, g in df.groupby("split"):
        if len(g) == 0:
            continue

        event_count = g.loc[g["event_id"] >= 0, "event_id"].nunique()

        rows.append(
            {
                "split": split,
                "n_rows": len(g),
                "n_sessions": g["session_id"].nunique(),
                "start_time": g["timestamp"].min(),
                "end_time": g["timestamp"].max(),
                "n_valid_samples": int(g["is_valid_sample"].sum()),
                "co_raw_mean": float(g["co_raw"].mean()),
                "co_raw_std": float(g["co_raw"].std()),
                "co_relative_mean": float(g["co_relative"].mean()),
                "co_relative_std": float(g["co_relative"].std()),
                "risk_target_mean": float(g["risk_target"].mean()),
                "risk_target_std": float(g["risk_target"].std()),
                "n_events": int(event_count),
                "base_interval_seconds": float(base_interval_seconds),
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/data.yaml")
    args = parser.parse_args()

    cfg = load_config(args.config)

    raw_data_dir = Path(cfg["raw_data_dir"])
    processed_data_dir = Path(cfg["processed_data_dir"])
    processed_data_dir.mkdir(parents=True, exist_ok=True)

    csv_files = sorted(raw_data_dir.glob("*.csv"))

    if not csv_files:
        raise FileNotFoundError(f"No CSV files found in {raw_data_dir}")

    print(f"Found {len(csv_files)} daily CSV files.")

    frames = [
        read_daily_csv(path, encoding=cfg.get("encoding", "utf-8-sig"))
        for path in csv_files
    ]

    df = pd.concat(frames, axis=0, ignore_index=True)
    df = df.dropna(subset=["timestamp"]).sort_values("timestamp").reset_index(drop=True)

    base_interval_seconds = infer_base_interval_seconds(df)
    print(f"Inferred base interval: {base_interval_seconds:.3f} seconds.")

    df = add_session_ids(
        df,
        session_gap_minutes=cfg["session_gap_minutes"],
    )

    df = add_past_baseline(
        df,
        past_baseline_minutes=cfg["past_baseline_minutes"],
        min_past_minutes=cfg["min_past_minutes"],
        base_interval_seconds=base_interval_seconds,
    )

    df = add_future_risk_target(
        df,
        future_start_minutes=cfg["future_start_minutes"],
        future_end_minutes=cfg["future_end_minutes"],
        future_quantile=cfg["future_quantile"],
        base_interval_seconds=base_interval_seconds,
    )

    df = assign_split(
        df,
        train_end_date=cfg["train_end_date"],
        val_end_date=cfg["val_end_date"],
        test_start_date=cfg["test_start_date"],
    )

    df = add_event_ids(
        df,
        event_threshold=cfg["event_threshold"],
        event_merge_gap_minutes=cfg["event_merge_gap_minutes"],
        min_event_duration_minutes=cfg["min_event_duration_minutes"],
    )

    df["is_valid_sample"] = (
        df["co_baseline"].notna()
        & df["co_relative"].notna()
        & df["risk_target"].notna()
        & df["split"].isin(["train", "val", "test"])
    )

    summary = make_summary(df, base_interval_seconds)

    csv_path = processed_data_dir / "risk_dataset.csv"
    summary_path = processed_data_dir / "data_summary.csv"
    meta_path = processed_data_dir / "risk_dataset_metadata.json"

    df.to_csv(csv_path, index=False)
    summary.to_csv(summary_path, index=False)

    metadata = {
        "n_raw_files": len(csv_files),
        "n_rows": int(len(df)),
        "base_interval_seconds": float(base_interval_seconds),
        "column_names": COLUMN_NAMES,
        "sensor_columns": SENSOR_COLUMNS,
        "config": cfg,
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2, default=str)

    print(f"Saved risk dataset: {csv_path}")
    print(f"Saved data summary: {summary_path}")
    print(f"Saved metadata: {meta_path}")
    print()
    print(summary)


if __name__ == "__main__":
    main()
