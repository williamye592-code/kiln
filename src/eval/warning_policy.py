"""
Convert model risk scores into actionable warning windows and evaluate event-level metrics.
"""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy_config", type=str, default="configs/warning_policy.yaml")
    args = parser.parse_args()

    print(f"TODO: evaluate warning policy using {args.policy_config}")


if __name__ == "__main__":
    main()
