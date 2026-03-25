"""
ChainVigil — GNN Inference & Risk Scoring

Loads a trained model and produces mule probability scores
for all accounts in the graph.
"""

import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from torch_geometric.data import Data

from backend.gnn.model import ChainVigilGNN
from backend.config import MODEL_DIR, RISK_THRESHOLD


def load_model(data: Data) -> ChainVigilGNN:
    """Load the best trained model checkpoint."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = ChainVigilGNN(
        in_channels=data.x.shape[1],
    ).to(device)

    path = os.path.join(MODEL_DIR, "best_model.pt")
    if os.path.exists(path):
        checkpoint = torch.load(path, map_location=device, weights_only=True)
        model.load_state_dict(checkpoint["model_state_dict"])
        print(f"✅ Model loaded (AUC: {checkpoint.get('best_val_auc', 'N/A')})")
    else:
        print("⚠️  No checkpoint found, using untrained model")

    model.eval()
    return model


def predict_scores(
    model: ChainVigilGNN,
    data: Data,
    account_ids: List[str],
    threshold: float = RISK_THRESHOLD,
) -> List[Dict]:
    """
    Generate risk scores for all accounts.

    Returns sorted list of account risk assessments.
    """
    device = next(model.parameters()).device
    data = data.to(device)

    with torch.no_grad():
        probs, embeddings = model(data.x, data.edge_index)

    probs_np = probs.cpu().numpy()

    results = []
    for idx, acc_id in enumerate(account_ids):
        score = float(probs_np[idx])
        action = _determine_action(score, threshold)

        results.append({
            "account_id": acc_id,
            "mule_probability": round(score, 4),
            "recommended_action": action,
            "is_flagged": score >= threshold,
        })

    # Sort by risk score descending
    results.sort(key=lambda x: x["mule_probability"], reverse=True)
    return results


def _determine_action(score: float, threshold: float) -> str:
    """Determine recommended action based on risk score."""
    if score >= threshold:
        return "Escalate"
    elif score >= threshold * 0.7:
        return "Freeze"
    elif score >= threshold * 0.5:
        return "Monitor"
    else:
        return "Clear"


def predict_account_score_realtime(
    model: ChainVigilGNN,
    data: Data,
    node_mapping: Dict[str, int],
    account_id: str,
    fallback_score: float = 0.5,
) -> float:
    """
    Get a single account GNN score for real-time APIs.

    Notes:
      - If account is not present in the current `node_mapping`, returns fallback.
      - For full online inference with new nodes, rebuild/extend PyG data incrementally.
    """
    if account_id not in node_mapping:
        return float(fallback_score)

    idx = node_mapping[account_id]
    device = next(model.parameters()).device
    data = data.to(device)

    model.eval()
    with torch.no_grad():
        probs, _ = model(data.x, data.edge_index)
    return float(probs[idx].cpu().item())
