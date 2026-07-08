"""
Build relative CO signals and future CO-deviation risk targets.

This script will convert raw kiln sensor data into model-ready samples for
risk-aware early warning.
"""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/data.yaml")
    args = parser.parse_args()

    print(f"TODO: build relative CO signals and risk targets using {args.config}")


if __name__ == "__main__":
    main()
