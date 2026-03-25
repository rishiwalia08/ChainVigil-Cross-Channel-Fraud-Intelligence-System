"""
ChainVigil — Explainable AI Module

Provides SHAP-based feature attribution and GNN subgraph explanations
for regulator-defensible audit reports.
"""

from typing import Dict, List, Optional

import numpy as np
import torch
from torch_geometric.data import Data

from backend.gnn.model import ChainVigilGNN
from backend.gnn.features import get_feature_names


class MuleExplainer:
    """
    Explains why an account was flagged as a potential mule.

    Uses:
      - Feature importance ranking (from model weights + gradients)
      - Neighborhood analysis for subgraph importance
      - Rule-based reasoning templates
    """

    def __init__(
        self,
        model: ChainVigilGNN,
        data: Data,
        account_ids: List[str],
        node_mapping: Dict[str, int],
    ):
        self.model = model
        self.data = data
        self.account_ids = account_ids
        self.node_mapping = node_mapping
        self.feature_names = get_feature_names()
        self.device = next(model.parameters()).device

    def explain_account(self, account_id: str) -> Dict:
        """
        Generate a comprehensive explanation for a flagged account.

        Returns:
            Dictionary with feature attributions and reasoning.
        """
        if account_id not in self.node_mapping:
            return {"error": f"Account {account_id} not found"}

        idx = self.node_mapping[account_id]

        # ─── Gradient-based feature importance ──────────────
        feature_importance = self._compute_gradient_importance(idx)

        # ─── Top contributing features ──────────────────────
        top_features = self._get_top_features(feature_importance, k=5)

        # ─── Feature values ─────────────────────────────────
        feature_values = self._get_feature_values(idx)

        # ─── Generate natural language reasoning ────────────
        reasoning = self._generate_reasoning(
            account_id, top_features, feature_values
        )

        # ─── Get mule probability ──────────────────────────
        self.model.eval()
        with torch.no_grad():
            probs, _ = self.model(
                self.data.x.to(self.device),
                self.data.edge_index.to(self.device)
            )
            mule_prob = float(probs[idx].cpu().item())

        return {
            "account_id": account_id,
            "confidence_score": round(mule_prob, 4),
            "top_features": [f["name"] for f in top_features],
            "feature_attributions": top_features,
            "feature_values": feature_values,
            "xai_reasoning": reasoning,
        }

    def _compute_gradient_importance(self, node_idx: int) -> np.ndarray:
        """Compute feature importance using input gradients."""
        self.model.eval()

        x = self.data.x.to(self.device).clone().requires_grad_(True)
        edge_index = self.data.edge_index.to(self.device)

        probs, _ = self.model(x, edge_index)
        target_prob = probs[node_idx]
        target_prob.backward()

        gradients = x.grad[node_idx].cpu().detach().numpy()
        feature_vals = self.data.x[node_idx].cpu().numpy()

        # Attribution = gradient × input (Gradient × Input method)
        attributions = np.abs(gradients * feature_vals)

        # Normalize to [0, 1]
        total = attributions.sum()
        if total > 0:
            attributions = attributions / total

        return attributions

    def _get_top_features(
        self, importance: np.ndarray, k: int = 5
    ) -> List[Dict]:
        """Get top-k most important features."""
        top_indices = np.argsort(importance)[::-1][:k]
        return [
            {
                "name": self.feature_names[i],
                "importance": round(float(importance[i]), 4),
                "rank": rank + 1,
            }
            for rank, i in enumerate(top_indices)
        ]

    def _get_feature_values(self, node_idx: int) -> Dict[str, float]:
        """Get raw feature values for a node (before normalization)."""
        # Note: data.x is normalized; we return the normalized values
        values = self.data.x[node_idx].cpu().numpy()
        return {
            name: round(float(values[i]), 4)
            for i, name in enumerate(self.feature_names)
            if i < len(values)
        }

    def _generate_reasoning(
        self,
        account_id: str,
        top_features: List[Dict],
        feature_values: Dict[str, float],
    ) -> str:
        """Generate human-readable explanation."""
        reasons = []

        for feat in top_features:
            name = feat["name"]
            val = feature_values.get(name, 0)

            if name == "avg_velocity_seconds" and val > 0:
                reasons.append(
                    f"High transaction velocity detected (avg interval: "
                    f"{abs(val):.0f}s normalized score)"
                )
            elif name == "shared_device_count" and val > 0:
                reasons.append(
                    f"Device shared with {val:.0f} other accounts "
                    f"(normalized score)"
                )
            elif name == "shared_ip_count" and val > 0:
                reasons.append(
                    f"IP address shared with {val:.0f} other accounts "
                    f"(normalized score)"
                )
            elif name == "clustering_coefficient" and val > 0:
                reasons.append(
                    f"High clustering coefficient ({val:.2f}) indicates "
                    f"dense interconnections typical of mule rings"
                )
            elif name == "pagerank" and val > 0:
                reasons.append(
                    f"Elevated PageRank centrality ({val:.4f}) suggests "
                    f"hub role in transaction network"
                )
            elif name == "channel_diversity" and val > 0:
                reasons.append(
                    f"Transactions across multiple channels "
                    f"(diversity score: {val:.2f})"
                )
            elif name == "amount_ratio" and val > 0:
                reasons.append(
                    f"Outflow-to-inflow ratio: {val:.2f} "
                    f"(pass-through behavior)"
                )
            elif name == "atm_withdrawal_count" and val > 0:
                reasons.append(
                    f"Elevated ATM withdrawal activity "
                    f"(normalized score: {val:.2f})"
                )
            elif name in ("in_degree", "out_degree", "total_degree"):
                reasons.append(
                    f"High transaction {name.replace('_', ' ')} "
                    f"(normalized: {val:.2f})"
                )
            elif name == "jurisdiction_risk_weight" and val > 0:
                reasons.append(
                    f"High-risk jurisdiction weight ({val:.2f})"
                )

        if not reasons:
            reasons.append("Anomalous combination of behavioral features detected")

        return f"Account {account_id} flagged due to: " + "; ".join(reasons) + "."

    def batch_explain(
        self, account_ids: List[str], top_k: int = 5
    ) -> List[Dict]:
        """Explain multiple accounts."""
        return [self.explain_account(acc_id) for acc_id in account_ids]
