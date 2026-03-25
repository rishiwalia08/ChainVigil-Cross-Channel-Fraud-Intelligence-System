"""
ChainVigil — PyTorch Geometric Dataset Converter

Exports the NetworkX Unified Entity Graph into PyTorch Geometric
Data objects for GNN training and inference.
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import networkx as nx
import torch
from torch_geometric.data import Data

from backend.gnn.features import compute_node_features, get_feature_names


def nx_to_pyg(
    G: nx.MultiDiGraph,
    feature_df: Optional[pd.DataFrame] = None,
) -> Tuple[Data, Dict[str, int], List[str]]:
    """
    Convert NetworkX graph to PyTorch Geometric Data object.

    Args:
        G: The full Unified Entity Graph (NetworkX MultiDiGraph)
        feature_df: Pre-computed feature DataFrame (optional)

    Returns:
        data: PyTorch Geometric Data object
        node_mapping: Dict mapping account_id -> node index
        account_ids: Ordered list of account IDs
    """
    # ─── Extract account nodes only ─────────────────────────
    account_ids = [
        n for n, d in G.nodes(data=True)
        if d.get("entity_type") == "Account"
    ]
    account_ids.sort()  # Deterministic ordering

    node_mapping = {acc_id: idx for idx, acc_id in enumerate(account_ids)}
    num_nodes = len(account_ids)

    # ─── Compute features ──────────────────────────────────
    if feature_df is None:
        print("   📊 Computing node features...")
        feature_df = compute_node_features(G, account_ids)

    feature_names = get_feature_names()
    # Ensure all features present, fill missing with 0
    for feat in feature_names:
        if feat not in feature_df.columns:
            feature_df[feat] = 0.0

    # Build feature matrix
    X = feature_df.loc[account_ids][feature_names].values.astype(np.float32)

    # Normalize features (z-score)
    means = X.mean(axis=0)
    stds = X.std(axis=0)
    stds[stds == 0] = 1.0  # Avoid division by zero
    X = (X - means) / stds

    x = torch.tensor(X, dtype=torch.float)

    # ─── Build edge index (account-to-account only) ────────
    edge_src = []
    edge_dst = []
    edge_attr_list = []

    for u, v, data in G.edges(data=True):
        if (data.get("edge_type") == "TRANSFERRED_TO" and
                u in node_mapping and v in node_mapping):
            edge_src.append(node_mapping[u])
            edge_dst.append(node_mapping[v])
            edge_attr_list.append([
                float(data.get("amount", 0)),
                _channel_to_int(data.get("channel_type", "")),
            ])

    if edge_src:
        edge_index = torch.tensor([edge_src, edge_dst], dtype=torch.long)
        edge_attr = torch.tensor(edge_attr_list, dtype=torch.float)
    else:
        edge_index = torch.zeros((2, 0), dtype=torch.long)
        edge_attr = torch.zeros((0, 2), dtype=torch.float)

    # ─── Labels (is_mule) ──────────────────────────────────
    labels = []
    for acc_id in account_ids:
        is_mule = G.nodes[acc_id].get("is_mule", False)
        labels.append(1 if is_mule else 0)

    y = torch.tensor(labels, dtype=torch.long)

    # ─── Build train/val/test masks ────────────────────────
    train_mask, val_mask, test_mask = _create_masks(y, num_nodes)

    # ─── Construct PyG Data ────────────────────────────────
    data = Data(
        x=x,
        edge_index=edge_index,
        edge_attr=edge_attr,
        y=y,
        train_mask=train_mask,
        val_mask=val_mask,
        test_mask=test_mask,
        num_nodes=num_nodes,
    )

    print(f"   ✅ PyG Data: {data.num_nodes} nodes, {data.num_edges} edges, "
          f"{x.shape[1]} features")
    print(f"      Labels: {y.sum().item()} mules / {num_nodes} total")
    print(f"      Train: {train_mask.sum().item()} | Val: {val_mask.sum().item()} | "
          f"Test: {test_mask.sum().item()}")

    return data, node_mapping, account_ids


def _channel_to_int(channel: str) -> float:
    """Map channel type to integer for edge features."""
    mapping = {"UPI": 0.0, "ATM": 1.0, "WEB": 2.0, "MOBILE_APP": 3.0}
    return mapping.get(channel, -1.0)


def _create_masks(
    y: torch.Tensor, num_nodes: int,
    train_ratio: float = 0.6,
    val_ratio: float = 0.2,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create stratified train/val/test masks.
    Ensures both classes are represented proportionally.
    """
    mule_indices = (y == 1).nonzero(as_tuple=True)[0].tolist()
    normal_indices = (y == 0).nonzero(as_tuple=True)[0].tolist()

    np.random.seed(42)
    np.random.shuffle(mule_indices)
    np.random.shuffle(normal_indices)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    for indices in [mule_indices, normal_indices]:
        n = len(indices)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)

        train_mask[indices[:n_train]] = True
        val_mask[indices[n_train:n_train + n_val]] = True
        test_mask[indices[n_train + n_val:]] = True

    return train_mask, val_mask, test_mask
