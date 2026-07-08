#!/bin/bash
set -e

echo "Evaluating event-level warning policy..."
python -m src.eval.warning_policy \
  --policy_config configs/warning_policy.yaml
