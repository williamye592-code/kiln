#!/bin/bash
set -e

echo "Training baseline models..."
python -m src.train.train_baselines \
  --data_config configs/data.yaml \
  --experiment_config configs/experiments_eaai.yaml
