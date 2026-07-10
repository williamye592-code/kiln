from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -60, 60)
    return 1.0 / (1.0 + np.exp(-x))


def find_input_path(path: str) -> Path:
    p = Path(path)
    if p.exists():
        return p

    candidates = [
        Path("outputs/predictions/risk_dataset_with_all_scores.csv"),
        Path("outputs/predictions/risk_dataset_with_scores.csv"),
        Path("outputs/processed/risk_dataset.csv"),
    ]
    for c in candidates:
        if c.exists():
            print(f"Input path not found: {p}. Falling back to {c}")
            return c

    raise FileNotFoundError(f"No usable input file found. Requested: {p}")


def numeric_series(df: pd.DataFrame, col: str, default: float = 0.0) -> pd.Series:
    if col not in df.columns:
        return pd.Series(default, index=df.index, dtype=float)
    return pd.to_numeric(df[col], errors="coerce")


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    co = numeric_series(df, "co_relative")
    h = numeric_series(df, "hgnn_score")

    co_pos = co.clip(lower=0)
    h_pos = h.clip(lower=0)

    out = pd.DataFrame(
        {
            "co_relative": co,
            "hgnn_score": h,
            "co_pos": co_pos,
            "hgnn_pos": h_pos,
            "co_abs": co.abs(),
            "hgnn_abs": h.abs(),
            "co_x_hgnn": co_pos * h_pos,
            "co_x_hgnn_signed": co * h,
        },
        index=df.index,
    )
    return out.replace([np.inf, -np.inf], np.nan)


def score_audit(df: pd.DataFrame, score_cols: List[str]) -> pd.DataFrame:
    rows = []
    for score_col in score_cols:
        if score_col not in df.columns:
            continue

        for split in sorted(df["split"].dropna().unique()):
            for period_type in sorted(df["period_type"].dropna().unique()):
                sub = df[(df["split"] == split) & (df["period_type"] == period_type)]
                if sub.empty:
                    continue

                s = pd.to_numeric(sub[score_col], errors="coerce").dropna()
                if s.empty:
                    rows.append(
                        {
                            "split": split,
                            "period_type": period_type,
                            "score_col": score_col,
                            "n_rows": len(sub),
                            "n_days": sub["date"].nunique() if "date" in sub.columns else np.nan,
                            "mean": np.nan,
                            "std": np.nan,
                            "p50": np.nan,
                            "p90": np.nan,
                            "p95": np.nan,
                            "p99": np.nan,
                            "max": np.nan,
                        }
                    )
                    continue

                rows.append(
                    {
                        "split": split,
                        "period_type": period_type,
                        "score_col": score_col,
                        "n_rows": len(sub),
                        "n_days": sub["date"].nunique() if "date" in sub.columns else np.nan,
                        "mean": float(s.mean()),
                        "std": float(s.std()),
                        "p50": float(s.quantile(0.50)),
                        "p90": float(s.quantile(0.90)),
                        "p95": float(s.quantile(0.95)),
                        "p99": float(s.quantile(0.99)),
                        "max": float(s.max()),
                    }
                )

    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_path", type=str, default="outputs/predictions/risk_dataset_with_all_scores.csv")
    parser.add_argument("--output_path", type=str, default="outputs/predictions/risk_dataset_with_policy_scores.csv")
    parser.add_argument("--metrics_path", type=str, default="outputs/metrics/policy_calibrated_score_audit.csv")
    parser.add_argument("--label_threshold", type=float, default=0.20)
    parser.add_argument("--learned_score_col", type=str, default="hgnn_policy_score")
    parser.add_argument("--fusion_score_col", type=str, default="calibrated_fusion_score")
    args = parser.parse_args()

    input_path = find_input_path(args.input_path)
    df = pd.read_csv(input_path)

    required = ["split", "period_type", "risk_target", "co_relative", "hgnn_score"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}. Available columns: {list(df.columns)}")

    train_mask = df["split"].eq("train")
    train_df = df[train_mask].copy()

    y_train = (numeric_series(train_df, "risk_target").fillna(0.0) >= args.label_threshold).astype(int)

    if y_train.nunique() < 2:
        print("Warning: train split has only one class. Falling back to train+val for calibrator fitting.")
        dev_mask = df["split"].isin(["train", "val"])
        train_df = df[dev_mask].copy()
        y_train = (numeric_series(train_df, "risk_target").fillna(0.0) >= args.label_threshold).astype(int)

    if y_train.nunique() < 2:
        raise ValueError("Cannot train calibrator: development labels still contain only one class.")

    X_train = make_features(train_df)
    medians = X_train.median(numeric_only=True).fillna(0.0)
    X_train = X_train.fillna(medians)

    model = Pipeline(
        steps=[
            ("scaler", StandardScaler()),
            (
                "clf",
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    C=1.0,
                    solver="lbfgs",
                ),
            ),
        ]
    )
    model.fit(X_train, y_train)

    X_all = make_features(df).fillna(medians)
    learned_prob = model.predict_proba(X_all)[:, 1]
    df[args.learned_score_col] = learned_prob

    # Conservative CO gate: choose gate center from train normal-operation p99 when available.
    train_normal = df[df["split"].eq("train") & df["period_type"].eq("normal_operation")]
    if len(train_normal) > 0:
        gate_center = float(pd.to_numeric(train_normal["co_relative"], errors="coerce").quantile(0.99))
    else:
        gate_center = float(pd.to_numeric(df.loc[train_mask, "co_relative"], errors="coerce").quantile(0.95))

    if not np.isfinite(gate_center):
        gate_center = 0.0

    gate_scale = 80.0
    co = numeric_series(df, "co_relative").fillna(0.0).to_numpy()
    gate = sigmoid(gate_scale * (co - gate_center))

    df["co_gate"] = gate
    df[args.fusion_score_col] = df[args.learned_score_col].to_numpy() * gate

    Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.output_path, index=False)

    score_cols = [
        "co_relative",
        "hgnn_score",
        args.learned_score_col,
        args.fusion_score_col,
        "rf_score",
        "risk_target",
    ]
    audit = score_audit(df, score_cols)
    Path(args.metrics_path).parent.mkdir(parents=True, exist_ok=True)
    audit.to_csv(args.metrics_path, index=False)

    clf = model.named_steps["clf"]
    feature_names = list(X_train.columns)
    coef = pd.DataFrame(
        {
            "feature": feature_names,
            "coef": clf.coef_[0],
        }
    ).sort_values("coef", ascending=False)

    print(f"Loaded input: {input_path}")
    print(f"Saved scored dataset: {args.output_path}")
    print(f"Saved score audit: {args.metrics_path}")
    print(f"Label threshold: risk_target >= {args.label_threshold}")
    print(f"Training rows: {len(train_df)}")
    print(f"Positive label rate: {float(y_train.mean()):.4f}")
    print(f"CO gate center: {gate_center:.6f}")
    print("\nCalibrator coefficients:")
    print(coef)

    print("\nScore audit preview:")
    show_cols = ["split", "period_type", "score_col", "n_rows", "n_days", "mean", "p90", "p95", "p99", "max"]
    print(audit[show_cols].sort_values(["score_col", "split", "period_type"]).to_string(index=False))


if __name__ == "__main__":
    main()
