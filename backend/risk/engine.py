"""
ChainVigil — Risk Intelligence Engine

Post-processing layer that:
  - Applies confidence thresholds
  - Aggregates cluster-level risk
  - Computes ring-level velocity metrics
  - Tracks evolving subgraphs over time
  - Generates suspicious account lists
"""

from typing import Dict, List, Optional, Tuple
from collections import defaultdict
from datetime import datetime

import numpy as np
import networkx as nx

from backend.config import RISK_THRESHOLD


class RiskIntelligenceEngine:
    """
    Aggregates GNN scores with topological analysis to produce
    cluster-level risk assessments and mule ring detection.
    """

    def __init__(
        self,
        G: nx.MultiDiGraph,
        risk_scores: List[Dict],
        threshold: float = RISK_THRESHOLD,
    ):
        self.G = G
        self.risk_scores = {r["account_id"]: r for r in risk_scores}
        self.threshold = threshold
        self.clusters = []
        self.flagged_accounts = []

    def analyze(self) -> Dict:
        """Run the full risk analysis pipeline."""
        print("\n🔍 Running Risk Intelligence Engine...")

        # Step 1: Flag high-risk accounts
        self._flag_accounts()

        # Step 2: Detect mule ring clusters
        self._detect_clusters()

        # Step 3: Compute velocity metrics per cluster
        self._compute_cluster_metrics()

        # Step 4: Generate summary
        summary = self._generate_summary()

        print(f"✅ Analysis complete: {len(self.flagged_accounts)} flagged, "
              f"{len(self.clusters)} clusters detected")

        return summary

    def _flag_accounts(self):
        """Identify accounts exceeding the risk threshold."""
        self.flagged_accounts = [
            acc_id for acc_id, data in self.risk_scores.items()
            if data.get("mule_probability", 0) >= self.threshold
        ]

        # Also flag accounts just below threshold with high connectivity to flagged
        borderline = [
            acc_id for acc_id, data in self.risk_scores.items()
            if self.threshold * 0.7 <= data.get("mule_probability", 0) < self.threshold
        ]

        for acc_id in borderline:
            if acc_id not in self.G:
                continue
            neighbors = set(self.G.successors(acc_id)) | set(self.G.predecessors(acc_id))
            flagged_neighbors = neighbors & set(self.flagged_accounts)
            if len(flagged_neighbors) >= 2:
                self.flagged_accounts.append(acc_id)

        print(f"   ⚠️  {len(self.flagged_accounts)} accounts flagged")

    def _detect_clusters(self):
        """
        Detect mule ring clusters using Louvain community detection
        on the subgraph of flagged accounts. Falls back to connected
        components if Louvain is unavailable.
        """
        # Build subgraph of flagged + transfer edges
        flagged_set = set(self.flagged_accounts)
        subgraph = nx.DiGraph()

        for u, v, data in self.G.edges(data=True):
            if (data.get("edge_type") == "TRANSFERRED_TO" and
                    u in flagged_set and v in flagged_set):
                subgraph.add_edge(u, v, **data)

        # Use Louvain community detection for better cluster separation
        # WCC often merges all rings into one blob due to bridge edges
        try:
            undirected = subgraph.to_undirected()
            if undirected.number_of_nodes() > 0:
                communities = nx.community.louvain_communities(
                    undirected, resolution=1.5, seed=42
                )
            else:
                communities = []
        except Exception:
            # Fallback to connected components
            communities = list(nx.weakly_connected_components(subgraph))

        self.clusters = []
        for idx, component in enumerate(communities):
            if len(component) >= 2:  # At least 2 members
                members = list(component)
                cluster_subgraph = subgraph.subgraph(members)

                self.clusters.append({
                    "cluster_id": f"MULE_RING_{idx:02d}",
                    "members": members,
                    "size": len(members),
                    "internal_edges": cluster_subgraph.number_of_edges(),
                    "density": nx.density(cluster_subgraph),
                    "avg_risk_score": np.mean([
                        self.risk_scores.get(m, {}).get("mule_probability", 0)
                        for m in members
                    ]),
                })

        # Sort by risk
        self.clusters.sort(key=lambda c: c["avg_risk_score"], reverse=True)
        print(f"   🕸️  {len(self.clusters)} potential mule ring clusters detected")

    def _compute_cluster_metrics(self):
        """Compute velocity and behavioral metrics for each cluster."""
        for cluster in self.clusters:
            members = set(cluster["members"])
            timestamps = []
            amounts = []
            channels = set()

            for u, v, data in self.G.edges(data=True):
                if (data.get("edge_type") == "TRANSFERRED_TO" and
                        u in members and v in members):
                    ts = data.get("timestamp", "")
                    if ts:
                        try:
                            timestamps.append(datetime.fromisoformat(ts))
                        except (ValueError, TypeError):
                            pass
                    amounts.append(data.get("amount", 0))
                    ch = data.get("channel_type")
                    if ch:
                        channels.add(ch)

            # Velocity metrics
            if len(timestamps) >= 2:
                timestamps.sort()
                deltas = [
                    (timestamps[i + 1] - timestamps[i]).total_seconds()
                    for i in range(len(timestamps) - 1)
                ]
                cluster["avg_velocity_seconds"] = float(np.mean(deltas))
                cluster["min_velocity_seconds"] = float(np.min(deltas))
                cluster["max_hop_chain_minutes"] = float(
                    (timestamps[-1] - timestamps[0]).total_seconds() / 60
                )
            else:
                cluster["avg_velocity_seconds"] = 0
                cluster["min_velocity_seconds"] = 0
                cluster["max_hop_chain_minutes"] = 0

            # Amount metrics
            cluster["total_volume"] = float(sum(amounts))
            cluster["avg_transaction"] = float(np.mean(amounts)) if amounts else 0
            cluster["channels_used"] = list(channels)
            cluster["channel_diversity"] = len(channels) / 4.0

            # Hub detection
            hub = max(
                cluster["members"],
                key=lambda m: self.risk_scores.get(m, {}).get("mule_probability", 0)
            )
            cluster["hub_account"] = hub

    def _generate_summary(self) -> Dict:
        """Generate the final risk intelligence report."""
        return {
            "timestamp": datetime.now().isoformat(),
            "total_accounts_analyzed": len(self.risk_scores),
            "flagged_accounts": len(self.flagged_accounts),
            "clusters_detected": len(self.clusters),
            "high_risk_accounts": [
                {
                    **self.risk_scores[acc_id],
                    "cluster_id": self._find_cluster(acc_id),
                }
                for acc_id in self.flagged_accounts[:20]  # Top 20
            ],
            "clusters": self.clusters,
            "risk_distribution": {
                "escalate": sum(
                    1 for r in self.risk_scores.values()
                    if r.get("recommended_action") == "Escalate"
                ),
                "freeze": sum(
                    1 for r in self.risk_scores.values()
                    if r.get("recommended_action") == "Freeze"
                ),
                "monitor": sum(
                    1 for r in self.risk_scores.values()
                    if r.get("recommended_action") == "Monitor"
                ),
                "clear": sum(
                    1 for r in self.risk_scores.values()
                    if r.get("recommended_action") == "Clear"
                ),
            },
        }

    def _find_cluster(self, account_id: str) -> Optional[str]:
        """Find which cluster an account belongs to."""
        for cluster in self.clusters:
            if account_id in cluster["members"]:
                return cluster["cluster_id"]
        return None

    def get_flagged_accounts(self) -> List[str]:
        return self.flagged_accounts

    def get_clusters(self) -> List[Dict]:
        return self.clusters
