#!/bin/bash
set -e

echo "Running preprocessing pipeline..."
python -m src.data.build_risk_targets --config configs/data.yaml
