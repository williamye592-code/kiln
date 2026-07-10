
"""
Build fusion risk score for robust actionable early warning.

The fusion score combines:
1. learned future-risk score from HGNN-risk, and
2. current positive CO deviation as a normal-operation gate.

fusion_score = hgnn_score * max(co_relative, 0)

This suppresses learned-model false alarms during normal-operation periods
where current CO deviation is near zero.
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
        default="outputs/predictions/risk_dataset_with_all_scores.csv",
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default="outputs/predictions/risk_dataset_with_all_scores.csv",
    )
    parser.add_argument(
        "--learned_score_col",
        type=str,
        default="hgnn_score",
    )
    parser.add_argument(
        "--gate_col",
        type=str,
        default="co_relative",
    )
    parser.add_argument(
        "--output_score_col",
        type=str,
        default="fusion_score",
    )
    args = parser.parse_args()

    input_path = Path(args.input_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path, parse_dates=["timestamp"])

    if args.learned_score_col not in df.columns:
        raise ValueError(f"Missing learned score column: {args.learned_score_col}")

    if args.gate_col not in df.columns:
        raise ValueError(f"Missing gate column: {args.gate_col}")

    gate = df[args.gate_col].clip(lower=0)
    df[args.output_score_col] = df[args.learned_score_col] * gate

    df.to_csv(output_path, index=False)

    print(f"Saved fusion score dataset: {output_path}")
    print()
    print(
        df[
            [args.gate_col, args.learned_score_col, args.output_score_col]
        ].describe()
    )


if __name__ == "__main__":
    main()
