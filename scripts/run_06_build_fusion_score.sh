#!/bin/bash
set -e

echo "Building fusion score..."

python -m src.eval.build_fusion_score \
  --input_path outputs/predictions/risk_dataset_with_all_scores.csv \
  --output_path outputs/predictions/risk_dataset_with_all_scores.csv \
  --learned_score_col hgnn_score \
  --gate_col co_relative \
  --output_score_col fusion_score
