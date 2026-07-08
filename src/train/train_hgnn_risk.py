
"""
Train HGNN-risk model for future CO-deviation risk scoring.

Input:
    risk_dataset_with_scores.csv

Output:
    risk_dataset_with_all_scores.csv with hgnn_score column
    hgnn_risk_metrics.csv
    hgnn_risk_training_history.csv

Model idea:
    raw sensors -> process nodes -> hypergraph message passing -> temporal GRU -> risk score
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def load_yaml(path: str) -> Dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_bool_series(s: pd.Series) -> pd.Series:
    if s.dtype == bool:
        return s
    return s.astype(str).str.lower().isin(["true", "1", "yes"])


class StandardScaler:
    def __init__(self) -> None:
        self.mean_ = None
        self.std_ = None

    def fit(self, x: np.ndarray) -> "StandardScaler":
        self.mean_ = np.nanmean(x, axis=0)
        self.std_ = np.nanstd(x, axis=0)
        self.std_[self.std_ < 1e-8] = 1.0
        return self

    def transform(self, x: np.ndarray) -> np.ndarray:
        return (x - self.mean_) / self.std_

    def inverse_transform(self, x: np.ndarray) -> np.ndarray:
        return x * self.std_ + self.mean_


def collect_feature_columns(nodes: Dict[str, List[str]]) -> List[str]:
    cols = []
    for _, node_cols in nodes.items():
        for c in node_cols:
            if c not in cols:
                cols.append(c)
    return cols


def fill_features_by_session(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    out = df[feature_cols].copy()
    out = out.replace([np.inf, -np.inf], np.nan)

    # Fill within each session first.
    filled_parts = []
    for _, idx in df.groupby("session_id").groups.items():
        part = out.loc[idx].copy()
        part = part.ffill().bfill()
        filled_parts.append(part)

    out = pd.concat(filled_parts, axis=0).sort_index()

    # Global median fallback.
    med = out.median(numeric_only=True)
    out = out.fillna(med).fillna(0.0)

    return out


def build_sample_indices(
    df: pd.DataFrame,
    split_name: str,
    seq_len: int,
    feature_matrix: np.ndarray,
    target_array: np.ndarray,
) -> np.ndarray:
    indices = []

    valid_mask = to_bool_series(df["is_valid_sample"]).to_numpy()
    split_values = df["split"].to_numpy()

    for _, g in df.groupby("session_id", sort=False):
        pos = g.index.to_numpy()

        for local_i in range(seq_len - 1, len(pos)):
            end_idx = int(pos[local_i])

            if not valid_mask[end_idx]:
                continue

            if split_values[end_idx] != split_name:
                continue

            if not np.isfinite(target_array[end_idx]):
                continue

            start_idx = int(pos[local_i - seq_len + 1])
            seq = feature_matrix[start_idx : end_idx + 1]

            if seq.shape[0] != seq_len:
                continue

            if not np.isfinite(seq).all():
                continue

            indices.append(end_idx)

    return np.asarray(indices, dtype=np.int64)


class SequenceRiskDataset(Dataset):
    def __init__(
        self,
        feature_matrix: np.ndarray,
        target_array: np.ndarray,
        end_indices: np.ndarray,
        seq_len: int,
    ) -> None:
        self.feature_matrix = feature_matrix.astype(np.float32)
        self.target_array = target_array.astype(np.float32)
        self.end_indices = end_indices.astype(np.int64)
        self.seq_len = int(seq_len)

    def __len__(self) -> int:
        return len(self.end_indices)

    def __getitem__(self, idx: int):
        end_idx = int(self.end_indices[idx])
        start_idx = end_idx - self.seq_len + 1

        x = self.feature_matrix[start_idx : end_idx + 1]
        y = self.target_array[end_idx]

        return torch.from_numpy(x), torch.tensor(y, dtype=torch.float32), end_idx


class HGNNRiskModel(nn.Module):
    def __init__(
        self,
        node_feature_indices: List[List[int]],
        hyperedges: List[List[int]],
        node_embed_dim: int,
        hidden_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()

        self.node_feature_indices = node_feature_indices
        self.n_nodes = len(node_feature_indices)
        self.node_embed_dim = node_embed_dim

        self.node_encoders = nn.ModuleList([
            nn.Sequential(
                nn.Linear(len(indices), node_embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            )
            for indices in node_feature_indices
        ])

        incidence = torch.zeros(self.n_nodes, len(hyperedges), dtype=torch.float32)

        for e_idx, edge in enumerate(hyperedges):
            for node_idx in edge:
                incidence[node_idx, e_idx] = 1.0

        edge_sizes = incidence.sum(dim=0).clamp_min(1.0)
        node_degrees = incidence.sum(dim=1).clamp_min(1.0)

        edge_norm = incidence / edge_sizes.unsqueeze(0)
        node_norm = incidence / node_degrees.unsqueeze(1)

        self.register_buffer("edge_norm", edge_norm)
        self.register_buffer("node_norm", node_norm)

        self.hyper_update = nn.Sequential(
            nn.Linear(node_embed_dim, node_embed_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        self.node_layer_norm = nn.LayerNorm(node_embed_dim)

        self.temporal = nn.GRU(
            input_size=node_embed_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
            dropout=0.0,
        )

        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, T, F]
        node_embs = []

        for indices, encoder in zip(self.node_feature_indices, self.node_encoders):
            node_x = x[:, :, indices]
            node_embs.append(encoder(node_x))

        # [B, T, N, D]
        node_emb = torch.stack(node_embs, dim=2)

        # Hyperedge aggregation: node -> edge -> node.
        # edge_emb: [B, T, E, D]
        edge_emb = torch.einsum("ne,btnd->bted", self.edge_norm, node_emb)

        # node_msg: [B, T, N, D]
        node_msg = torch.einsum("ne,bted->btnd", self.node_norm, edge_emb)

        node_updated = self.node_layer_norm(
            node_emb + self.hyper_update(node_msg)
        )

        # Graph-level sequence: [B, T, D]
        graph_seq = node_updated.mean(dim=2)

        _, h = self.temporal(graph_seq)
        last = h[-1]

        pred = self.head(last).squeeze(-1)
        return pred


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict:
    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y_true = y_true[mask]
    y_pred = y_pred[mask]

    if len(y_true) == 0:
        return {
            "n": 0,
            "mae": np.nan,
            "rmse": np.nan,
            "r2": np.nan,
        }

    mse = mean_squared_error(y_true, y_pred)

    return {
        "n": int(len(y_true)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(np.sqrt(mse)),
        "r2": float(r2_score(y_true, y_pred)),
    }


def evaluate_loss(model, loader, device, criterion) -> float:
    model.eval()
    losses = []

    with torch.no_grad():
        for x, y, _ in loader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x)
            loss = criterion(pred, y)
            losses.append(loss.item())

    return float(np.mean(losses)) if losses else np.nan


def collect_predictions(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()

    preds = []
    end_indices = []

    with torch.no_grad():
        for x, _, idx in loader:
            x = x.to(device)
            pred = model(x).detach().cpu().numpy()
            preds.append(pred)
            end_indices.append(idx.numpy())

    if not preds:
        return np.asarray([]), np.asarray([])

    return np.concatenate(preds), np.concatenate(end_indices)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/model_hgnn_risk.yaml")
    args = parser.parse_args()

    cfg = load_yaml(args.config)
    set_seed(int(cfg["seed"]))

    input_path = Path(cfg["input_path"])
    output_path = Path(cfg["output_path"])
    metrics_path = Path(cfg["metrics_path"])

    output_path.parent.mkdir(parents=True, exist_ok=True)
    metrics_path.parent.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path, parse_dates=[cfg["time_col"]])
    df = df.sort_values(cfg["time_col"]).reset_index(drop=True)

    nodes = cfg["nodes"]
    node_names = list(nodes.keys())
    feature_cols = collect_feature_columns(nodes)

    missing = [c for c in feature_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    feature_df = fill_features_by_session(df, feature_cols)

    train_mask = (
        to_bool_series(df["is_valid_sample"])
        & (df[cfg["split_col"]] == "train")
        & df[cfg["target_col"]].notna()
    )

    feature_scaler = StandardScaler().fit(feature_df.loc[train_mask].to_numpy())
    target_scaler = StandardScaler().fit(df.loc[train_mask, cfg["target_col"]].to_numpy().reshape(-1, 1))

    X_all = feature_scaler.transform(feature_df.to_numpy()).astype(np.float32)
    y_all_raw = df[cfg["target_col"]].to_numpy(dtype=float)
    y_all = target_scaler.transform(y_all_raw.reshape(-1, 1)).reshape(-1).astype(np.float32)

    seq_len = int(cfg["seq_len"])

    train_indices = build_sample_indices(df, "train", seq_len, X_all, y_all)
    val_indices = build_sample_indices(df, "val", seq_len, X_all, y_all)
    test_indices = build_sample_indices(df, "test", seq_len, X_all, y_all)

    print(f"Train samples: {len(train_indices)}")
    print(f"Val samples:   {len(val_indices)}")
    print(f"Test samples:  {len(test_indices)}")

    train_ds = SequenceRiskDataset(X_all, y_all, train_indices, seq_len)
    val_ds = SequenceRiskDataset(X_all, y_all, val_indices, seq_len)
    test_ds = SequenceRiskDataset(X_all, y_all, test_indices, seq_len)

    train_loader = DataLoader(
        train_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=True,
        drop_last=False,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        drop_last=False,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=int(cfg["batch_size"]),
        shuffle=False,
        drop_last=False,
    )

    feature_index = {c: i for i, c in enumerate(feature_cols)}
    node_feature_indices = [
        [feature_index[c] for c in nodes[node_name]]
        for node_name in node_names
    ]

    node_name_to_idx = {name: i for i, name in enumerate(node_names)}
    hyperedges = [
        [node_name_to_idx[name] for name in edge]
        for edge in cfg["hyperedges"]
    ]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = HGNNRiskModel(
        node_feature_indices=node_feature_indices,
        hyperedges=hyperedges,
        node_embed_dim=int(cfg["node_embed_dim"]),
        hidden_dim=int(cfg["hidden_dim"]),
        dropout=float(cfg["dropout"]),
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(cfg["lr"]),
        weight_decay=float(cfg["weight_decay"]),
    )

    criterion = nn.MSELoss()

    best_val = float("inf")
    best_state = None
    patience = int(cfg["patience"])
    bad_epochs = 0
    history = []

    for epoch in range(1, int(cfg["epochs"]) + 1):
        model.train()
        train_losses = []

        for x, y, _ in train_loader:
            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()
            pred = model(x)
            loss = criterion(pred, y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

            train_losses.append(loss.item())

        train_loss = float(np.mean(train_losses))
        val_loss = evaluate_loss(model, val_loader, device, criterion)

        history.append(
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "val_loss": val_loss,
            }
        )

        print(f"Epoch {epoch:03d} | train_loss={train_loss:.6f} | val_loss={val_loss:.6f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = {
                k: v.detach().cpu().clone()
                for k, v in model.state_dict().items()
            }
            bad_epochs = 0
        else:
            bad_epochs += 1

        if bad_epochs >= patience:
            print(f"Early stopping at epoch {epoch}.")
            break

    if best_state is not None:
        model.load_state_dict(best_state)

    df[cfg["score_col"]] = np.nan

    metrics_rows = []

    for split_name, loader, indices in [
        ("train", train_loader, train_indices),
        ("val", val_loader, val_indices),
        ("test", test_loader, test_indices),
    ]:
        pred_scaled, pred_indices = collect_predictions(model, loader, device)
        pred_raw = target_scaler.inverse_transform(pred_scaled.reshape(-1, 1)).reshape(-1)

        df.loc[pred_indices, cfg["score_col"]] = pred_raw

        y_true = df.loc[pred_indices, cfg["target_col"]].to_numpy(dtype=float)
        m = regression_metrics(y_true, pred_raw)
        m["model"] = "hgnn_risk"
        m["split"] = split_name
        metrics_rows.append(m)

    metrics = pd.DataFrame(metrics_rows)
    history_df = pd.DataFrame(history)

    history_path = metrics_path.parent / "hgnn_risk_training_history.csv"
    meta_path = metrics_path.parent / "hgnn_risk_metadata.json"
    ckpt_path = metrics_path.parent / "hgnn_risk_best.pt"

    df.to_csv(output_path, index=False)
    metrics.to_csv(metrics_path, index=False)
    history_df.to_csv(history_path, index=False)

    torch.save(model.state_dict(), ckpt_path)

    meta = {
        "config": cfg,
        "node_names": node_names,
        "feature_columns": feature_cols,
        "train_samples": int(len(train_indices)),
        "val_samples": int(len(val_indices)),
        "test_samples": int(len(test_indices)),
        "best_val_loss": float(best_val),
    }

    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2, default=str)

    print(f"Saved scored dataset: {output_path}")
    print(f"Saved metrics: {metrics_path}")
    print(f"Saved history: {history_path}")
    print(f"Saved checkpoint: {ckpt_path}")
    print()
    print(metrics)


if __name__ == "__main__":
    main()
