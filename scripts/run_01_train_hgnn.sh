#!/bin/bash
set -e

echo "Training HGNN temporal risk model..."
python -m src.train.train_hgnn_risk \
  --data_config configs/data.yaml \
  --graph_config configs/graph_hypergraph.yaml \
  --model_config configs/model_hgnn_tt.yaml
