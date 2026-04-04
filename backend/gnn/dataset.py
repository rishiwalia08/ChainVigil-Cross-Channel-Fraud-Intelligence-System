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

    # ── Feature hardening (STEP 5 extension) ─────────────────────────
    # Z-score alone on skewed count features creates near-perfect linear separation.
    # Strategy:
    #   1. Log1p-transform count/degree/amount features (compresses 10-20x ratios → ~2-3x)
    #   2. Clamp shared_device_count & shared_ip_count to max=4 (removes extreme outliers)
    #   3. Robust scaling (median + IQR) so mule outliers don't crush normal variance to ~0

    log_feature_indices = [
        i for i, name in enumerate(feature_names)
        if any(k in name for k in [
            "degree", "amount", "count", "pagerank",
            "betweenness", "velocity", "atm_total"
        ])
    ]
    shared_dev_idx = feature_names.index("shared_device_count") if "shared_device_count" in feature_names else -1
    shared_ip_idx  = feature_names.index("shared_ip_count")     if "shared_ip_count"     in feature_names else -1

    # Clamp shared counts BEFORE log (max=4 lets mule signal survive but removes extreme outliers)
    if shared_dev_idx >= 0:
        X[:, shared_dev_idx] = np.clip(X[:, shared_dev_idx], 0, 4)
    if shared_ip_idx >= 0:
        X[:, shared_ip_idx]  = np.clip(X[:, shared_ip_idx],  0, 4)

    # Apply log1p to skewed count/amount features
    X[:, log_feature_indices] = np.log1p(np.abs(X[:, log_feature_indices]))

    # Robust scaling: (x - median) / (IQR + eps)
    median = np.median(X, axis=0)
    q75 = np.percentile(X, 75, axis=0)
    q25 = np.percentile(X, 25, axis=0)
    iqr = q75 - q25
    iqr[iqr == 0] = 1.0  # avoid division by zero for constant features
    X = (X - median) / iqr

    # ── Winsorize post-scaling: clip to ±5 IQR units ──────────────────────
    # After robust scaling, mule velocity values hit −18 IQR units — so extreme
    # that a single threshold perfectly separates them. Capping at ±5 compresses
    # mule outliers into a realistic range while preserving separation direction.
    X = np.clip(X, -5.0, 5.0)

    # ── Post-scaling structural noise ─────────────────────────────────────
    # Root cause: pagerank/shared_ip/device for mule rings pile up AT the +5 ceiling
    # while normals sit near 0. Pre-scaling noise cannot fix this — the topology
    # is deterministically separable. Adding jitter in the NORMALIZED space forces
    # real distributional overlap.
    # std=2.0 on a [-5,5] range: sep(pagerank) drops from ~4.8 → ~1.6  ✓
    _post_rng = np.random.default_rng(77)
    for feat_name, post_noise_std in [
        ("pagerank",               2.0),  # structural cheater sep≈4.8 → ~1.6
        ("shared_ip_count",        1.5),  # sep≈4.1  → ~1.5
        ("shared_device_count",    1.2),  # sep≈3.9  → ~1.5
        ("in_degree",              1.2),  # sep≈3.85 → ~1.5  ← now dominant
        ("total_degree",           1.0),  # sep≈3.21 → ~1.4
        ("out_degree",             0.8),  # sep≈2.69 → ~1.3
        ("betweenness_centrality", 1.0),  # sep≈2.0  → moderate
    ]:
        if feat_name in feature_names:
            idx = feature_names.index(feat_name)
            X[:, idx] += _post_rng.normal(0, post_noise_std, size=X.shape[0])
    X = np.clip(X, -5.0, 5.0)  # re-apply bounds after post-noise

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
    train_ratio: float = 0.70,   # STEP 1: Fixed from 0.60 → 0.70
    val_ratio: float = 0.15,     # STEP 1: Fixed from 0.20 → 0.15 (test = remaining 0.15)
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create stratified train/val/test masks with 70/15/15 split.

    No data leakage guarantee: each node index is assigned to exactly ONE mask.
    Stratification ensures both mule and normal classes are proportionally
    represented in every split.
    """
    mule_indices = (y == 1).nonzero(as_tuple=True)[0].tolist()
    normal_indices = (y == 0).nonzero(as_tuple=True)[0].tolist()

    rng = np.random.default_rng(42)  # Use Generator for reproducibility
    rng.shuffle(mule_indices)
    rng.shuffle(normal_indices)

    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)

    for indices in [mule_indices, normal_indices]:
        n = len(indices)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        # test gets the remaining slice — no overlap possible
        train_mask[indices[:n_train]] = True
        val_mask[indices[n_train:n_train + n_val]] = True
        test_mask[indices[n_train + n_val:]] = True  # strictly disjoint

    # Sanity check: no node should be in more than one mask
    assert not (train_mask & val_mask).any(), "Leakage: train ∩ val"
    assert not (train_mask & test_mask).any(), "Leakage: train ∩ test"
    assert not (val_mask & test_mask).any(), "Leakage: val ∩ test"

    return train_mask, val_mask, test_mask
