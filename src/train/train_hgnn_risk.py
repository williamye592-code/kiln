"""
Train the mechanism-guided HGNN temporal model for future CO-deviation risk scoring.
"""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_config", type=str, default="configs/data.yaml")
    parser.add_argument("--graph_config", type=str, default="configs/graph_hypergraph.yaml")
    parser.add_argument("--model_config", type=str, default="configs/model_hgnn_tt.yaml")
    args = parser.parse_args()

    print("TODO: train HGNN temporal risk model.")
    print(f"Data config: {args.data_config}")
    print(f"Graph config: {args.graph_config}")
    print(f"Model config: {args.model_config}")


if __name__ == "__main__":
    main()
