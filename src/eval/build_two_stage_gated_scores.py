from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def gate_tag(x: float) -> str:
    return str(x).replace(".", "p").replace("-", "m")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="outputs/predictions/risk_dataset_with_policy_scores.csv")
    parser.add_argument("--output_path", type=str, default="outputs/predictions/risk_dataset_with_two_stage_scores.csv")
    parser.add_argument(
        "--model_cols",
        nargs="+",
        default=[
            "rf_score",
            "hgnn_score",
            "hgnn_policy_score",
            "calibrated_fusion_score",
        ],
    )
    parser.add_argument(
        "--co_gates",
        nargs="+",
        type=float,
        default=[
            0.001, 0.002, 0.005, 0.008,
            0.010, 0.012, 0.014, 0.016, 0.018,
            0.020, 0.025, 0.030, 0.040, 0.050,
        ],
    )
    args = parser.parse_args()

    in_path = Path(args.input_path)
    if not in_path.exists():
        fallback = Path("outputs/predictions/risk_dataset_with_all_scores.csv")
        print(f"{in_path} not found. Falling back to {fallback}")
        in_path = fallback

    df = pd.read_csv(in_path)

    if "co_relative" not in df.columns:
        raise ValueError("Missing co_relative column.")

    co = pd.to_numeric(df["co_relative"], errors="coerce").fillna(-1e9)

    created = []

    for model_col in args.model_cols:
        if model_col not in df.columns:
            print(f"Skip missing model column: {model_col}")
            continue

        model_score = pd.to_numeric(df[model_col], errors="coerce").fillna(0.0)

        for gate in args.co_gates:
            name = f"{model_col}_co_gate_{gate_tag(gate)}"
            df[name] = np.where(co >= gate, model_score, 0.0)
            created.append(name)

    out_path = Path(args.output_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"Loaded: {in_path}")
    print(f"Saved: {out_path}")
    print(f"Created {len(created)} two-stage gated score columns.")
    print("Preview created columns:")
    for c in created[:20]:
        print(" ", c)
    if len(created) > 20:
        print(" ...")


if __name__ == "__main__":
    main()
