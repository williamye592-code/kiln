
"""
Create paper-ready main result table for kiln early-warning framework.

This table compares representative deployable strategies on the held-out test period:
- current relative CO score
- RF baseline
- HGNN-risk
- CO-gated HGNN fusion
- oracle future-risk upper bound

The output is a compact table with event-rich detection metrics and
normal-operation false-alarm metrics side by side.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


REPRESENTATIVE_POLICIES = [
    {
        "method": "Current relative CO",
        "score_col": "co_relative",
        "threshold": 0.05,
        "role": "current-state baseline",
    },
    {
        "method": "Random forest risk score",
        "score_col": "rf_score",
        "threshold": 0.30,
        "role": "flat ML baseline",
    },
    {
        "method": "HGNN-risk score",
        "score_col": "hgnn_score",
        "threshold": 0.20,
        "role": "mechanism-guided learned score",
    },
    {
        "method": "CO-gated HGNN fusion",
        "score_col": "calibrated_fusion_score",
        "threshold": 0.005,
        "role": "proposed robust fusion",
    },
    {
        "method": "Oracle future risk",
        "score_col": "risk_target",
        "threshold": 0.20,
        "role": "non-deployable upper bound",
    },
]


def get_row(res: pd.DataFrame, score_col: str, threshold: float, period_type: str) -> pd.Series:
    sub = res[
        (res["split"] == "test")
        & (res["score_col"] == score_col)
        & (res["period_type"] == period_type)
        & (res["threshold"].round(6) == round(threshold, 6))
    ]

    if sub.empty:
        raise ValueError(
            f"Missing result for score_col={score_col}, "
            f"threshold={threshold}, period_type={period_type}"
        )

    return sub.iloc[0]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        default="outputs/policy_eval/policy_eval_results.csv",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="paper_assets/tables/main_result_table.csv",
    )
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    res = pd.read_csv(input_path)

    rows = []

    for policy in REPRESENTATIVE_POLICIES:
        event_row = get_row(
            res,
            score_col=policy["score_col"],
            threshold=policy["threshold"],
            period_type="event_rich",
        )

        normal_row = get_row(
            res,
            score_col=policy["score_col"],
            threshold=policy["threshold"],
            period_type="normal_operation",
        )

        rows.append(
            {
                "method": policy["method"],
                "score_col": policy["score_col"],
                "threshold": policy["threshold"],
                "role": policy["role"],

                "event_rich_days": event_row["n_days"],
                "event_rich_events": event_row["n_events"],
                "event_rich_warning_windows": event_row["n_warning_windows"],
                "event_detection_rate": event_row["event_detection_rate"],
                "median_lead_minutes": event_row["median_lead_minutes"],
                "warning_precision_event_rich": event_row["warning_precision"],
                "false_windows_per_event_rich_day": event_row["false_windows_per_day"],
                "active_minutes_per_event_rich_day": event_row["active_minutes_per_day"],

                "normal_operation_days": normal_row["n_days"],
                "normal_warning_windows": normal_row["n_warning_windows"],
                "false_windows_per_normal_day": normal_row["false_windows_per_day"],
                "active_minutes_per_normal_day": normal_row["active_minutes_per_day"],
            }
        )

    table = pd.DataFrame(rows)

    numeric_cols = [
        "event_detection_rate",
        "median_lead_minutes",
        "warning_precision_event_rich",
        "false_windows_per_event_rich_day",
        "active_minutes_per_event_rich_day",
        "false_windows_per_normal_day",
        "active_minutes_per_normal_day",
    ]

    for c in numeric_cols:
        table[c] = table[c].astype(float).round(4)

    table.to_csv(output_path, index=False)

    print(f"Saved main result table: {output_path}")
    print()
    print(table)


if __name__ == "__main__":
    main()
