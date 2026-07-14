#!/bin/bash
set -e

echo "Training baseline models..."

python -m src.train.train_baselines \
  --input_path outputs/processed/risk_dataset.csv \
  --output_path outputs/predictions/risk_dataset_with_scores.csv \
  --metrics_path outputs/metrics/rf_baseline_metrics.csv \
  --n_estimators 200 \
  --max_depth 12 \
  --min_samples_leaf 20
