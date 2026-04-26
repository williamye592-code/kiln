import argparse
import json
from pathlib import Path

import numpy as np

from data_pipeline import read_all_csv, split_file_arrays, build_samples_per_file
from hgnn_config import COL_INDEX
from train_hgnn_temporal import infer_base_interval_seconds


def metrics(y_true, y_pred):
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    mse = float(np.mean((y_pred - y_true) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(y_pred - y_true)))
    mape = float(np.mean(np.abs((y_pred - y_true) / (y_true + 1e-8))) * 100.0)

    sst = float(np.sum((y_true - np.mean(y_true)) ** 2))
    sse = float(np.sum((y_true - y_pred) ** 2))
    r2 = 1.0 - sse / (sst + 1e-12)

    return {
        "rmse": rmse,
        "mae": mae,
        "mape_percent": mape,
        "r2": r2,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="data/huizhuanyao_data")
    parser.add_argument("--seq_len", type=int, default=60)
    parser.add_argument("--forecast_seconds", type=float, default=600.0)
    parser.add_argument("--train_ratio", type=float, default=0.7)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--output_dir", type=str, default="outputs_baselines")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    file_arrays = read_all_csv(Path(args.data_dir))
    base_interval_seconds = infer_base_interval_seconds(file_arrays)
    horizon_steps = max(1, int(round(args.forecast_seconds / base_interval_seconds)))

    file_splits = split_file_arrays(
        file_arrays,
        train_ratio=args.train_ratio,
        val_ratio=args.val_ratio,
    )

    X_train, y_train = build_samples_per_file(
        file_splits["train"],
        seq_len=args.seq_len,
        horizon=horizon_steps,
    )
    X_val, y_val = build_samples_per_file(
        file_splits["val"],
        seq_len=args.seq_len,
        horizon=horizon_steps,
    )
    X_test, y_test = build_samples_per_file(
        file_splits["test"],
        seq_len=args.seq_len,
        horizon=horizon_steps,
    )

    co_idx = COL_INDEX["co"]

    # Baseline 1: last observed CO in the input window
    y_test_last = X_test[:, -1, co_idx]

    # Baseline 2: training-set target mean
    y_train_mean = float(np.mean(y_train))
    y_test_mean = np.full_like(y_test, fill_value=y_train_mean, dtype=np.float32)

    results = {
        "args": vars(args),
        "base_interval_seconds": base_interval_seconds,
        "horizon_steps": horizon_steps,
        "effective_forecast_seconds": horizon_steps * base_interval_seconds,
        "n_train_samples": int(len(y_train)),
        "n_val_samples": int(len(y_val)),
        "n_test_samples": int(len(y_test)),
        "last_value_baseline": metrics(y_test, y_test_last),
        "train_mean_baseline": metrics(y_test, y_test_mean),
    }

    with open(output_dir / "baseline_summary.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)

    np.savez(
        output_dir / "baseline_predictions_test.npz",
        y_true=y_test,
        y_last_value=y_test_last,
        y_train_mean=y_test_mean,
    )

    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
