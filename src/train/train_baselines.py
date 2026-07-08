"""
Train baseline models for CO-deviation risk scoring.
"""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_config", type=str, default="configs/data.yaml")
    parser.add_argument("--experiment_config", type=str, default="configs/experiments_eaai.yaml")
    args = parser.parse_args()

    print("TODO: train baselines.")
    print(f"Data config: {args.data_config}")
    print(f"Experiment config: {args.experiment_config}")


if __name__ == "__main__":
    main()
