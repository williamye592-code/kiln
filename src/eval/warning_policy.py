
"""
Event-level warning policy evaluator.

This script converts risk scores into actionable warning windows and evaluates
industrial early-warning metrics for EAAI-style decision-level analysis.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import yaml


def load_config(config_path: str) -> Dict:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def infer_sampling_seconds(df: pd.DataFrame, time_col: str) -> float:
    diffs = df[time_col].sort_values().diff().dt.total_seconds()
    diffs = diffs[(diffs > 0) & (diffs <= 60)]
    if len(diffs) == 0:
        return 10.0
    return float(diffs.median())


def make_warning_windows(
    df: pd.DataFrame,
    score_col: str,
    threshold: float,
    time_col: str,
    merge_gap_minutes: float,
    cooldown_minutes: float,
    max_window_minutes: float,
) -> pd.DataFrame:
    """Convert threshold exceedances into merged and capped warning windows."""
    g = df.sort_values(time_col).copy()
    g = g[g[score_col].notna()].copy()

    if g.empty:
        return pd.DataFrame(columns=["window_start", "window_end", "max_score"])

    sampling_seconds = infer_sampling_seconds(g, time_col)
    above = g[g[score_col] >= threshold].copy()

    if above.empty:
        return pd.DataFrame(columns=["window_start", "window_end", "max_score"])

    raw_windows = []

    current_start = None
    current_end = None
    current_max = None

    for row in above.itertuples(index=False):
        ts = getattr(row, time_col)
        score = getattr(row, score_col)

        if current_start is None:
            current_start = ts
            current_end = ts
            current_max = score
            continue

        gap_min = (ts - current_end).total_seconds() / 60.0

        if gap_min <= merge_gap_minutes:
            current_end = ts
            current_max = max(current_max, score)
        else:
            raw_windows.append((current_start, current_end, current_max))
            current_start = ts
            current_end = ts
            current_max = score

    raw_windows.append((current_start, current_end, current_max))

    # Add one sampling interval so single-point windows have nonzero duration.
    sample_delta = pd.Timedelta(seconds=sampling_seconds)

    capped = []
    for start, end, max_score in raw_windows:
        end = end + sample_delta
        cap_end = start + pd.Timedelta(minutes=max_window_minutes)
        end = min(end, cap_end)
        capped.append((start, end, max_score))

    # Apply cooldown between emitted warning windows.
    emitted = []
    last_end = None

    for start, end, max_score in capped:
        if last_end is not None:
            if start <= last_end + pd.Timedelta(minutes=cooldown_minutes):
                continue

        emitted.append((start, end, max_score))
        last_end = end

    return pd.DataFrame(
        emitted,
        columns=["window_start", "window_end", "max_score"],
    )


def extract_events(
    df: pd.DataFrame,
    time_col: str,
    event_id_col: str,
) -> pd.DataFrame:
    """Extract event start/end times from event_id labels."""
    event_df = df[df[event_id_col] >= 0].copy()

    if event_df.empty:
        return pd.DataFrame(columns=["event_id", "event_start", "event_end", "event_duration_minutes"])

    rows = []

    for event_id, g in event_df.groupby(event_id_col):
        start = g[time_col].min()
        end = g[time_col].max()
        duration = (end - start).total_seconds() / 60.0

        rows.append(
            {
                "event_id": int(event_id),
                "event_start": start,
                "event_end": end,
                "event_duration_minutes": duration,
            }
        )

    return pd.DataFrame(rows)


def evaluate_windows_against_events(
    windows: pd.DataFrame,
    events: pd.DataFrame,
    lead_horizon_minutes: float,
    min_lead_minutes: float,
) -> Dict:
    """Evaluate warning windows against event-level detection targets."""
    n_windows = len(windows)
    n_events = len(events)

    if n_events == 0:
        return {
            "n_events": 0,
            "n_detected_events": 0,
            "event_detection_rate": np.nan,
            "median_lead_minutes": np.nan,
            "lead_q25_minutes": np.nan,
            "lead_q75_minutes": np.nan,
            "n_true_warning_windows": 0,
            "n_false_warning_windows": n_windows,
            "warning_precision": 0.0 if n_windows > 0 else np.nan,
        }

    detected = []
    lead_times = []

    true_window_flags = np.zeros(n_windows, dtype=bool)

    for _, event in events.iterrows():
        event_start = event["event_start"]

        horizon_start = event_start - pd.Timedelta(minutes=lead_horizon_minutes)
        horizon_end = event_start - pd.Timedelta(minutes=min_lead_minutes)

        event_detected = False
        event_leads = []

        for i, window in windows.iterrows():
            ws = window["window_start"]
            we = window["window_end"]

            overlaps_actionable_horizon = (we >= horizon_start) and (ws <= horizon_end)

            if overlaps_actionable_horizon:
                true_window_flags[i] = True
                event_detected = True

                effective_warning_time = max(ws, horizon_start)
                lead = (event_start - effective_warning_time).total_seconds() / 60.0
                lead = max(0.0, min(float(lead), float(lead_horizon_minutes)))
                event_leads.append(lead)

        detected.append(event_detected)

        if event_leads:
            lead_times.append(max(event_leads))

    n_detected = int(np.sum(detected))
    n_true_windows = int(true_window_flags.sum())
    n_false_windows = int(n_windows - n_true_windows)

    if lead_times:
        median_lead = float(np.median(lead_times))
        q25 = float(np.quantile(lead_times, 0.25))
        q75 = float(np.quantile(lead_times, 0.75))
    else:
        median_lead = np.nan
        q25 = np.nan
        q75 = np.nan

    precision = n_true_windows / n_windows if n_windows > 0 else np.nan

    return {
        "n_events": int(n_events),
        "n_detected_events": n_detected,
        "event_detection_rate": n_detected / n_events if n_events > 0 else np.nan,
        "median_lead_minutes": median_lead,
        "lead_q25_minutes": q25,
        "lead_q75_minutes": q75,
        "n_true_warning_windows": n_true_windows,
        "n_false_warning_windows": n_false_windows,
        "warning_precision": precision,
    }


def evaluate_one_setting(
    df: pd.DataFrame,
    score_col: str,
    split: str,
    period_type: str,
    threshold: float,
    merge_gap_minutes: float,
    cooldown_minutes: float,
    max_window_minutes: float,
    cfg: Dict,
) -> Tuple[Dict, pd.DataFrame]:
    time_col = cfg["time_col"]
    split_col = cfg["split_col"]
    period_type_col = cfg["period_type_col"]
    event_id_col = cfg["event_id_col"]

    sub = df[df[split_col] == split].copy()

    if period_type != "all":
        sub = sub[sub[period_type_col] == period_type].copy()

    sub = sub[sub["is_valid_sample"] == True].copy()

    if sub.empty:
        row = {
            "score_col": score_col,
            "split": split,
            "period_type": period_type,
            "threshold": threshold,
            "merge_gap_minutes": merge_gap_minutes,
            "cooldown_minutes": cooldown_minutes,
            "max_window_minutes": max_window_minutes,
            "n_points": 0,
            "n_days": 0,
            "n_warning_windows": 0,
            "warning_windows_per_day": np.nan,
            "active_minutes": 0.0,
            "active_minutes_per_day": np.nan,
            "false_windows_per_day": np.nan,
        }
        return row, pd.DataFrame()

    n_days = sub["date"].nunique()

    windows = make_warning_windows(
        sub,
        score_col=score_col,
        threshold=threshold,
        time_col=time_col,
        merge_gap_minutes=merge_gap_minutes,
        cooldown_minutes=cooldown_minutes,
        max_window_minutes=max_window_minutes,
    )

    if not windows.empty:
        windows["score_col"] = score_col
        windows["split"] = split
        windows["period_type"] = period_type
        windows["threshold"] = threshold
        windows["merge_gap_minutes"] = merge_gap_minutes
        windows["cooldown_minutes"] = cooldown_minutes
        windows["max_window_minutes"] = max_window_minutes
        windows["duration_minutes"] = (
            windows["window_end"] - windows["window_start"]
        ).dt.total_seconds() / 60.0
    else:
        windows["duration_minutes"] = []

    events = extract_events(
        sub,
        time_col=time_col,
        event_id_col=event_id_col,
    )

    event_metrics = evaluate_windows_against_events(
        windows,
        events,
        lead_horizon_minutes=cfg["lead_horizon_minutes"],
        min_lead_minutes=cfg["min_lead_minutes"],
    )

    n_warning_windows = len(windows)
    active_minutes = float(windows["duration_minutes"].sum()) if n_warning_windows > 0 else 0.0

    row = {
        "score_col": score_col,
        "split": split,
        "period_type": period_type,
        "threshold": threshold,
        "merge_gap_minutes": merge_gap_minutes,
        "cooldown_minutes": cooldown_minutes,
        "max_window_minutes": max_window_minutes,
        "n_points": len(sub),
        "n_days": n_days,
        "n_warning_windows": n_warning_windows,
        "warning_windows_per_day": n_warning_windows / n_days if n_days > 0 else np.nan,
        "active_minutes": active_minutes,
        "active_minutes_per_day": active_minutes / n_days if n_days > 0 else np.nan,
    }

    row.update(event_metrics)

    row["false_windows_per_day"] = (
        row["n_false_warning_windows"] / n_days if n_days > 0 else np.nan
    )

    return row, windows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_config", type=str, default="configs/warning_policy.yaml")
    args = parser.parse_args()

    cfg = load_config(args.policy_config)

    input_path = Path(cfg["input_path"])
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path, parse_dates=[cfg["time_col"]])

    all_rows = []
    all_windows = []

    for score_col in cfg["score_columns"]:
        if score_col not in df.columns:
            print(f"Skipping missing score column: {score_col}")
            continue

        for split in cfg["splits"]:
            for period_type in cfg["period_types"]:
                for threshold in cfg["thresholds"]:
                    for merge_gap in cfg["merge_gap_minutes"]:
                        for cooldown in cfg["cooldown_minutes"]:
                            for max_window in cfg["max_window_minutes"]:
                                row, windows = evaluate_one_setting(
                                    df=df,
                                    score_col=score_col,
                                    split=split,
                                    period_type=period_type,
                                    threshold=float(threshold),
                                    merge_gap_minutes=float(merge_gap),
                                    cooldown_minutes=float(cooldown),
                                    max_window_minutes=float(max_window),
                                    cfg=cfg,
                                )

                                all_rows.append(row)

                                if windows is not None and not windows.empty:
                                    all_windows.append(windows)

    results = pd.DataFrame(all_rows)

    if all_windows:
        window_df = pd.concat(all_windows, axis=0, ignore_index=True)
    else:
        window_df = pd.DataFrame()

    results_path = output_dir / "policy_eval_results.csv"
    windows_path = output_dir / "warning_windows.csv"

    results.to_csv(results_path, index=False)
    window_df.to_csv(windows_path, index=False)

    print(f"Saved policy results: {results_path}")
    print(f"Saved warning windows: {windows_path}")
    print()
    print(results.head(20))


if __name__ == "__main__":
    main()
