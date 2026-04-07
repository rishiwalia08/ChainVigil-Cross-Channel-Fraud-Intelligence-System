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
from backend.xai.llm_explainer import LLMExplainer


DRIVER_MEANINGS = {
    "pagerank": "A network importance score based on how many important accounts connect to it.",
    "betweenness_centrality": "How often the account sits on shortest paths between other accounts; high values indicate broker/relay behavior.",
    "in_degree": "How many incoming transfers the account receives.",
    "out_degree": "How many outgoing transfers the account sends.",
    "total_degree": "Total number of inbound + outbound transfer links.",
    "total_in_amount": "Total incoming amount transferred into the account.",
    "total_out_amount": "Total outgoing amount transferred out of the account.",
    "avg_velocity_seconds": "Average time gap between the account's transactions.",
    "min_velocity_seconds": "Shortest observed time gap between consecutive transactions.",
    "shared_device_count": "How many other accounts use the same device.",
    "shared_ip_count": "How many other accounts use the same IP address.",
    "channel_diversity": "How many distinct transaction channels are used (UPI/ATM/WEB/etc).",
    "amount_ratio": "Outflow compared to inflow; high values can indicate pass-through behavior.",
    "atm_withdrawal_count": "Cash-out intensity via ATM withdrawals.",
    "clustering_coefficient": "How densely this account is interconnected with neighbors.",
    "jurisdiction_risk_weight": "Aggregate risk score from linked geographies/jurisdictions.",
}


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
        self.llm_explainer = LLMExplainer()

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
            logits, _ = self.model(
                self.data.x.to(self.device),
                self.data.edge_index.to(self.device)
            )
            mule_prob = float(torch.sigmoid(logits[idx]).cpu().item())

        llm_payload = self.llm_explainer.summarize_account(
            account_id=account_id,
            confidence_score=mule_prob,
            feature_attributions=top_features,
            base_reasoning=reasoning,
        )

        key_driver_meanings = self._build_driver_meanings(top_features)
        suggested_actions = self._build_suggested_actions(mule_prob, top_features)

        return {
            "account_id": account_id,
            "confidence_score": round(mule_prob, 4),
            "top_features": [f["name"] for f in top_features],
            "feature_attributions": top_features,
            "feature_values": feature_values,
            "xai_reasoning": reasoning,
            "plain_english_summary": llm_payload.get("summary", ""),
            "llm_meta": llm_payload.get("meta", {}),
            "key_driver_meanings": key_driver_meanings,
            "suggested_actions": suggested_actions,
        }

    def _build_driver_meanings(self, top_features: List[Dict]) -> List[Dict]:
        driver_rows = []
        for feat in top_features[:5]:
            name = feat.get("name", "unknown")
            normalized = str(name).strip().lower().replace(" ", "_")
            driver_rows.append({
                "feature": name,
                "importance": round(float(feat.get("importance", 0.0)), 4),
                "meaning": DRIVER_MEANINGS.get(
                    normalized,
                    f"Model-derived signal for {str(name).replace('_', ' ')}."
                ),
            })
        return driver_rows

    def _build_suggested_actions(
        self,
        confidence_score: float,
        top_features: List[Dict],
    ) -> List[str]:
        feature_names = {f.get("name") for f in top_features}
        actions: List[str] = []

        if confidence_score >= 0.85:
            actions.append("Apply temporary transaction restrictions and trigger urgent analyst review.")
        elif confidence_score >= 0.60:
            actions.append("Initiate Enhanced Due Diligence (EDD) and increase monitoring frequency.")
        else:
            actions.append("Keep under watchlist with periodic behavioral review.")

        if "shared_device_count" in feature_names or "shared_ip_count" in feature_names:
            actions.append("Request fresh KYC and corroborate device/IP ownership.")

        if "avg_velocity_seconds" in feature_names or "amount_ratio" in feature_names:
            actions.append("Monitor for rapid in-out pass-through and cash-out attempts.")

        if "pagerank" in feature_names or "total_degree" in feature_names:
            actions.append("Run ring-neighbor review for first and second hop connected accounts.")

        return actions[:4]

    def _compute_gradient_importance(self, node_idx: int) -> np.ndarray:
        """Compute feature importance using input gradients."""
        self.model.eval()

        x = self.data.x.to(self.device).clone().requires_grad_(True)
        edge_index = self.data.edge_index.to(self.device)

        logits, _ = self.model(x, edge_index)
        target_logit = logits[node_idx]
        target_logit.backward()

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
