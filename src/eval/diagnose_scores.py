
"""
Diagnose score distributions across event-rich and normal-operation periods.

This script is used to explain why a score works or fails at the
decision level. In particular, it can reveal whether a learned model assigns
inflated risk scores to normal-operation periods.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input_path",
        type=str,
        default="outputs/predictions/risk_dataset_with_scores.csv",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="outputs/metrics",
    )
    parser.add_argument(
        "--score_columns",
        nargs="+",
        default=["co_relative", "risk_target", "rf_score"],
    )
    parser.add_argument(
        "--thresholds",
        nargs="+",
        type=float,
        default=[0.05, 0.10, 0.15, 0.20, 0.30],
    )
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path, parse_dates=["timestamp"])
    valid = df[df["is_valid_sample"] == True].copy()

    score_rows = []

    for split in ["train", "val", "test"]:
        for period_type in ["event_rich", "normal_operation"]:
            sub = valid[
                (valid["split"] == split)
                & (valid["period_type"] == period_type)
            ].copy()

            if sub.empty:
                continue

            for score in args.score_columns:
                if score not in sub.columns:
                    continue

                score_rows.append(
                    {
                        "split": split,
                        "period_type": period_type,
                        "score_col": score,
                        "n_rows": len(sub),
                        "n_days": sub["date"].nunique(),
                        "mean": sub[score].mean(),
                        "std": sub[score].std(),
                        "p50": sub[score].quantile(0.50),
                        "p90": sub[score].quantile(0.90),
                        "p95": sub[score].quantile(0.95),
                        "p99": sub[score].quantile(0.99),
                        "max": sub[score].max(),
                    }
                )

    score_audit = pd.DataFrame(score_rows)
    score_audit_path = output_dir / "score_distribution_audit.csv"
    score_audit.to_csv(score_audit_path, index=False)

    threshold_rows = []

    for score in args.score_columns:
        if score not in valid.columns:
            continue

        for split in ["train", "val", "test"]:
            for period_type in ["event_rich", "normal_operation"]:
                sub = valid[
                    (valid["split"] == split)
                    & (valid["period_type"] == period_type)
                ].copy()

                if sub.empty:
                    continue

                for thr in args.thresholds:
                    threshold_rows.append(
                        {
                            "score_col": score,
                            "split": split,
                            "period_type": period_type,
                            "threshold": thr,
                            "n_rows": len(sub),
                            "n_days": sub["date"].nunique(),
                            "n_above_threshold": int((sub[score] >= thr).sum()),
                            "frac_above_threshold": float((sub[score] >= thr).mean()),
                            "score_p95": float(sub[score].quantile(0.95)),
                            "score_p99": float(sub[score].quantile(0.99)),
                            "score_max": float(sub[score].max()),
                        }
                    )

    threshold_audit = pd.DataFrame(threshold_rows)
    threshold_audit_path = output_dir / "threshold_distribution_audit.csv"
    threshold_audit.to_csv(threshold_audit_path, index=False)

    print(f"Saved score audit: {score_audit_path}")
    print(f"Saved threshold audit: {threshold_audit_path}")
    print()
    print("Score distribution audit:")
    print(score_audit.sort_values(["score_col", "split", "period_type"]))
    print()
    print("Test threshold audit:")
    print(
        threshold_audit[
            threshold_audit["split"] == "test"
        ].sort_values(["score_col", "threshold", "period_type"])
    )


if __name__ == "__main__":
    main()
