#!/bin/bash
set -e

echo "Diagnosing score distributions..."

python -m src.eval.diagnose_scores \
  --input_path outputs/predictions/risk_dataset_with_scores.csv \
  --output_dir outputs/metrics \
  --score_columns co_relative risk_target rf_score \
  --thresholds 0.05 0.10 0.15 0.20 0.30
