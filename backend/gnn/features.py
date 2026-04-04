"""
ChainVigil — Feature Engineering for GNN

Computes topological and temporal features for each node in the
Unified Entity Graph:
  - Transaction velocity (Δt between hops)
  - In-degree / Out-degree
  - PageRank centrality
  - Clustering coefficient
  - Channel diversity score
  - Jurisdiction risk weight
  - Shared device count
  - Betweenness centrality
"""

from typing import Dict, List, Optional
from datetime import datetime

import numpy as np
import pandas as pd
import networkx as nx


def compute_node_features(
    G: nx.MultiDiGraph,
    account_ids: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Compute feature vectors for all Account nodes in the graph.

    Returns DataFrame indexed by account_id with feature columns.
    """
    if account_ids is None:
        account_ids = [
            n for n, d in G.nodes(data=True)
            if d.get("entity_type") == "Account"
        ]

    features = []

    # Precompute graph-level metrics on the account-only subgraph
    account_subgraph = _build_account_subgraph(G, account_ids)
    pagerank = nx.pagerank(account_subgraph, alpha=0.85)
    try:
        betweenness = nx.betweenness_centrality(account_subgraph)
    except Exception:
        betweenness = {n: 0.0 for n in account_ids}

    clustering = nx.clustering(account_subgraph.to_undirected())

    for acc_id in account_ids:
        feat = _compute_single_account_features(
            G, acc_id, pagerank, betweenness, clustering
        )
        features.append(feat)

    df = pd.DataFrame(features)
    df.set_index("account_id", inplace=True)

    # ── Pre-noise percentile caps: prevent structural outliers from dominating ─
    # PageRank in a synthetic mule ring is a near-perfect cluster detector.
    # Capping at p95 before adding noise brings mule values into a realistic range.
    for col, pct in [
        ("pagerank",               95),   # sep=7.37 — now dominant after velocity fix
        ("betweenness_centrality", 97),
        ("shared_ip_count",        95),
        ("shared_device_count",    95),
    ]:
        if col in df.columns:
            cap = np.percentile(df[col], pct)
            if cap > 0:
                df[col] = df[col].clip(upper=cap)

    _rng = np.random.default_rng(99)

    # ── Non-negative features: clamp to 0 after noise ─────────────────────

    for col, noise_std_frac in [
        ("in_degree",              0.30),
        ("out_degree",             0.30),
        ("total_degree",           0.30),
        ("shared_device_count",    0.40),   # bumped: sep=4.45 after cap
        ("shared_ip_count",        0.40),   # bumped: sep=4.62 after cap
        ("atm_withdrawal_count",   0.20),
        ("total_in_amount",        0.15),
        ("total_out_amount",       0.15),
    ]:
        if col in df.columns:
            col_std = df[col].std() if df[col].std() > 0 else 1.0
            noise = _rng.normal(0, col_std * noise_std_frac, size=len(df))
            df[col] = (df[col] + noise).clip(lower=0)

    # ── Velocity / centrality: DO NOT clip to 0 ────────────────────────────
    # After robust scaling, mule velocity values are large-negative
    # (rapid transactions → tiny deltas → far below median).
    # Clipping these to 0 would erase the noise entirely for mule nodes.
    for col, noise_std_frac in [
        ("min_velocity_seconds",   0.50),   # sep was 11.1 → now winsorized too
        ("avg_velocity_seconds",   0.40),
        ("max_velocity_seconds",   0.35),
        ("pagerank",               0.50),   # sep=7.37 — extra noise after p95 cap
        ("betweenness_centrality", 0.35),
        ("clustering_coefficient", 0.20),
    ]:
        if col in df.columns:
            col_std = df[col].std() if df[col].std() > 0 else 1.0
            noise = _rng.normal(0, col_std * noise_std_frac, size=len(df))
            df[col] = df[col] + noise   # unrestricted — may remain negative

    return df




def _build_account_subgraph(
    G: nx.MultiDiGraph, account_ids: List[str]
) -> nx.DiGraph:
    """Build a simplified DiGraph with only account-to-account transfer edges."""
    subG = nx.DiGraph()
    subG.add_nodes_from(account_ids)

    for u, v, data in G.edges(data=True):
        if (data.get("edge_type") == "TRANSFERRED_TO" and
                u in account_ids and v in account_ids):
            if subG.has_edge(u, v):
                subG[u][v]["weight"] += 1
                subG[u][v]["total_amount"] += data.get("amount", 0)
            else:
                subG.add_edge(u, v, weight=1, total_amount=data.get("amount", 0))

    return subG


def _compute_single_account_features(
    G: nx.MultiDiGraph,
    acc_id: str,
    pagerank: Dict,
    betweenness: Dict,
    clustering: Dict,
) -> Dict:
    """Compute features for a single account."""
    node_data = G.nodes[acc_id]

    # ─── Degree features ────────────────────────────────────
    in_edges = list(G.in_edges(acc_id, data=True))
    out_edges = list(G.out_edges(acc_id, data=True))

    transfer_in = [e for e in in_edges if e[2].get("edge_type") == "TRANSFERRED_TO"]
    transfer_out = [e for e in out_edges if e[2].get("edge_type") == "TRANSFERRED_TO"]

    in_degree = len(transfer_in)
    out_degree = len(transfer_out)

    # ─── Amount features ────────────────────────────────────
    in_amounts = [e[2].get("amount", 0) for e in transfer_in]
    out_amounts = [e[2].get("amount", 0) for e in transfer_out]

    total_in = sum(in_amounts)
    total_out = sum(out_amounts)
    avg_in = np.mean(in_amounts) if in_amounts else 0
    avg_out = np.mean(out_amounts) if out_amounts else 0

    # ─── Transaction velocity ───────────────────────────────
    all_timestamps = []
    for e in transfer_in + transfer_out:
        ts_str = e[2].get("timestamp", "")
        if ts_str:
            try:
                all_timestamps.append(datetime.fromisoformat(ts_str))
            except (ValueError, TypeError):
                pass

    if len(all_timestamps) >= 2:
        all_timestamps.sort()
        deltas = [
            (all_timestamps[i + 1] - all_timestamps[i]).total_seconds()
            for i in range(len(all_timestamps) - 1)
        ]
        avg_velocity = np.mean(deltas)
        min_velocity = np.min(deltas)
        max_velocity = np.max(deltas)
    else:
        avg_velocity = min_velocity = max_velocity = 0

    # ─── Channel diversity ──────────────────────────────────
    channels = set()
    for e in transfer_in + transfer_out:
        ch = e[2].get("channel_type")
        if ch:
            channels.add(ch)
    channel_diversity = len(channels) / 4.0  # 4 possible channels

    # ─── Shared device count ────────────────────────────────
    device_edges = [e for e in out_edges if e[2].get("edge_type") == "USED_DEVICE"]
    device_ids = [e[1] for e in device_edges]
    shared_device_count = 0
    for dev_id in device_ids:
        # Count how many OTHER accounts share this device
        dev_users = [
            e[0] for e in G.in_edges(dev_id, data=True)
            if e[2].get("edge_type") == "USED_DEVICE" and e[0] != acc_id
        ]
        shared_device_count += len(dev_users)

    # ─── Shared IP count ───────────────────────────────────
    ip_edges = [e for e in out_edges if e[2].get("edge_type") == "LOGGED_FROM"]
    ip_ids = [e[1] for e in ip_edges]
    shared_ip_count = 0
    for ip_id in ip_ids:
        ip_users = [
            e[0] for e in G.in_edges(ip_id, data=True)
            if e[2].get("edge_type") == "LOGGED_FROM" and e[0] != acc_id
        ]
        shared_ip_count += len(ip_users)

    # ─── ATM withdrawal pattern ────────────────────────────
    atm_edges = [e for e in out_edges if e[2].get("edge_type") == "WITHDREW_AT"]
    atm_count = len(atm_edges)
    atm_total = sum(e[2].get("amount", 0) for e in atm_edges)

    return {
        "account_id": acc_id,
        # Degree
        "in_degree": in_degree,
        "out_degree": out_degree,
        "total_degree": in_degree + out_degree,
        # Amount
        "total_in_amount": total_in,
        "total_out_amount": total_out,
        "avg_in_amount": avg_in,
        "avg_out_amount": avg_out,
        "amount_ratio": total_out / max(total_in, 1),
        # Velocity
        "avg_velocity_seconds": avg_velocity,
        "min_velocity_seconds": min_velocity,
        "max_velocity_seconds": max_velocity,
        # Centrality
        "pagerank": pagerank.get(acc_id, 0),
        "betweenness_centrality": betweenness.get(acc_id, 0),
        "clustering_coefficient": clustering.get(acc_id, 0),
        # Diversity
        "channel_diversity": channel_diversity,
        # Shared resources
        "shared_device_count": shared_device_count,
        "shared_ip_count": shared_ip_count,
        # ATM
        "atm_withdrawal_count": atm_count,
        "atm_total_amount": atm_total,
        # Jurisdiction
        "jurisdiction_risk_weight": node_data.get("jurisdiction_risk_weight", 0),
    }


def get_feature_names() -> List[str]:
    """Return ordered list of feature names (matches feature vector dimensions)."""
    return [
        "in_degree", "out_degree", "total_degree",
        "total_in_amount", "total_out_amount",
        "avg_in_amount", "avg_out_amount", "amount_ratio",
        "avg_velocity_seconds", "min_velocity_seconds", "max_velocity_seconds",
        "pagerank", "betweenness_centrality", "clustering_coefficient",
        "channel_diversity",
        "shared_device_count", "shared_ip_count",
        "atm_withdrawal_count", "atm_total_amount",
        "jurisdiction_risk_weight",
    ]


def extract_realtime_features(
    G: nx.MultiDiGraph,
    source_id: str,
    target_id: str,
) -> Dict[str, Dict]:
    """
    Compute feature vectors only for impacted accounts in real-time.
    Returns a dict {account_id: feature_dict}.
    """
    account_ids = []
    for acc_id in [source_id, target_id]:
        if acc_id in G and G.nodes[acc_id].get("entity_type") == "Account":
            account_ids.append(acc_id)

    if not account_ids:
        return {}

    account_subgraph = _build_account_subgraph(G, account_ids)
    pagerank = nx.pagerank(account_subgraph, alpha=0.85) if account_subgraph.number_of_nodes() else {}
    try:
        betweenness = nx.betweenness_centrality(account_subgraph)
    except Exception:
        betweenness = {n: 0.0 for n in account_ids}
    clustering = nx.clustering(account_subgraph.to_undirected()) if account_subgraph.number_of_nodes() else {}

    features = {}
    for acc_id in account_ids:
        features[acc_id] = _compute_single_account_features(
            G,
            acc_id,
            pagerank,
            betweenness,
            clustering,
        )
    return features
