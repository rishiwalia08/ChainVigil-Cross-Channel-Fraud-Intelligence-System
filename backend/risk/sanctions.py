"""
ChainVigil — Behaviour-Based Sanctions Screening

Implements TWO layers of sanctions screening:
1. Watchlist match — direct account ID check against simulated OFAC/UN/ED list
2. Behaviour match — flags accounts whose behavioural profile matches
   known sanctions fingerprints even if NOT on the list

This directly addresses the IBA theme requirement:
"enhances sanctions screening through behaviour-based signals
 rather than simple list matching"
"""

from typing import Dict, List, Optional
import numpy as np
from datetime import datetime

# ── Simulated Sanctions Watchlist (mimics OFAC / UN Security Council / ED) ──
# In production: fetched from RBI Caution List, UN SCSIL, OFAC SDN List
SANCTIONS_WATCHLIST = {
    "ACC-001", "ACC-007", "ACC-023", "ACC-041", "ACC-088",
    "ACC-102", "ACC-134", "ACC-156", "ACC-178", "ACC-199",
    "ACC-203", "ACC-217", "ACC-234", "ACC-256", "ACC-271",
    "ACC-289", "ACC-301", "ACC-312", "ACC-334", "ACC-349",
}

# ── Known Sanctions Behavioural Fingerprints ──
# Each fingerprint is a normalised feature vector:
# [in_degree, out_degree, channel_diversity, shared_device_count,
#  shared_ip_count, atm_withdrawal_count, betweenness_centrality_norm,
#  jurisdiction_risk_weight, avg_out_amount_norm, velocity_score]
SANCTIONS_FINGERPRINTS = [
    # High-velocity international mule
    {
        "name": "High-Velocity Cross-Border Mule",
        "vector": [0.1, 0.9, 0.8, 0.6, 0.7, 0.2, 0.8, 0.9, 0.7, 0.9],
        "description": "Low inbound, high outbound with multiple channels and high jurisdictional risk",
    },
    # Shell account / pass-through
    {
        "name": "Shell Account Pass-Through",
        "vector": [0.8, 0.8, 0.3, 0.9, 0.8, 0.0, 0.5, 0.5, 0.5, 0.8],
        "description": "High degree both ways, extreme device/IP sharing, no ATM activity",
    },
    # ATM cash-out mule
    {
        "name": "ATM Cash-Out Operative",
        "vector": [0.3, 0.1, 0.2, 0.4, 0.3, 0.95, 0.2, 0.6, 0.8, 0.7],
        "description": "Receives funds then immediately withdraws via ATM across jurisdictions",
    },
    # Structuring specialist
    {
        "name": "Structuring Specialist",
        "vector": [0.2, 0.7, 0.5, 0.3, 0.4, 0.1, 0.4, 0.4, 0.95, 0.6],
        "description": "Consistent high-value outflows just below reporting threshold",
    },
]

COSINE_MATCH_THRESHOLD = 0.82  # Min cosine similarity to flag a behaviour match


def _cosine_similarity(a: List[float], b: List[float]) -> float:
    """Compute cosine similarity between two vectors."""
    a_arr = np.array(a, dtype=float)
    b_arr = np.array(b, dtype=float)
    norm_a = np.linalg.norm(a_arr)
    norm_b = np.linalg.norm(b_arr)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a_arr, b_arr) / (norm_a * norm_b))


def _account_to_vector(risk_data: Dict) -> List[float]:
    """
    Convert an account's risk data to a normalised feature vector
    that can be compared against sanctions fingerprints.
    """
    def _norm(val, max_val):
        return min(1.0, float(val or 0) / max_val) if max_val else 0.0

    return [
        _norm(risk_data.get("in_degree", 0), 50),
        _norm(risk_data.get("out_degree", 0), 50),
        float(risk_data.get("channel_diversity", 0)),
        _norm(risk_data.get("shared_device_count", 0), 10),
        _norm(risk_data.get("shared_ip_count", 0), 10),
        _norm(risk_data.get("atm_withdrawal_count", 0), 20),
        float(risk_data.get("betweenness_centrality", 0)),
        float(risk_data.get("jurisdiction_risk_weight", 0)),
        _norm(risk_data.get("avg_out_amount", 0), 1_000_000),
        float(risk_data.get("mule_probability", 0)),
    ]


class SanctionsScreener:
    """
    Two-layer sanctions and watchlist screening engine.
    """

    def __init__(self, watchlist: Optional[set] = None):
        self.watchlist = watchlist or SANCTIONS_WATCHLIST
        self.fingerprints = SANCTIONS_FINGERPRINTS

    def screen_account(self, account_id: str, risk_data: Dict) -> Dict:
        """
        Full screening of a single account.
        Returns match type, confidence, and explanation.
        """
        # Layer 1: Direct watchlist match
        list_match = account_id in self.watchlist

        # Layer 2: Behaviour-based match
        vec = _account_to_vector(risk_data)
        behaviour_matches = []

        for fp in self.fingerprints:
            similarity = _cosine_similarity(vec, fp["vector"])
            if similarity >= COSINE_MATCH_THRESHOLD:
                behaviour_matches.append({
                    "fingerprint_name": fp["name"],
                    "similarity_score": round(similarity * 100, 1),
                    "description": fp["description"],
                })

        # Determine result
        if list_match and behaviour_matches:
            match_type = "CONFIRMED_SANCTIONS_MATCH"
            confidence = 99
            explanation = (
                f"Account '{account_id}' is on the sanctions watchlist AND "
                f"behavioural profile matches: {behaviour_matches[0]['fingerprint_name']}."
            )
        elif list_match:
            match_type = "WATCHLIST_MATCH"
            confidence = 95
            explanation = f"Account '{account_id}' appears on the RBI/OFAC/UN watchlist."
        elif behaviour_matches:
            best = behaviour_matches[0]
            match_type = "BEHAVIOURAL_MATCH"
            confidence = int(best["similarity_score"])
            explanation = (
                f"Account not on watchlist but behavioural profile ({best['similarity_score']}% "
                f"similarity) matches '{best['fingerprint_name']}': {best['description']}"
            )
        else:
            match_type = "CLEAR"
            confidence = 0
            explanation = "No sanctions match detected."

        return {
            "account_id": account_id,
            "match_type": match_type,
            "confidence_score": confidence,
            "watchlist_hit": list_match,
            "behaviour_matches": behaviour_matches,
            "explanation": explanation,
            "recommended_action": (
                "Immediate freeze & report to FIU-IND" if match_type in ("CONFIRMED_SANCTIONS_MATCH", "WATCHLIST_MATCH")
                else "Enhanced Due Diligence" if match_type == "BEHAVIOURAL_MATCH"
                else "No action required"
            ),
            "screened_at": datetime.now().isoformat(),
        }

    def screen_all(self, risk_scores: List[Dict]) -> Dict:
        """Screen all accounts and return summary."""
        results = []
        watchlist_hits = []
        behaviour_hits = []

        for account in risk_scores:
            account_id = account.get("account_id", "")
            result = self.screen_account(account_id, account)
            results.append(result)

            if result["match_type"] == "WATCHLIST_MATCH":
                watchlist_hits.append(result)
            elif result["match_type"] in ("BEHAVIOURAL_MATCH", "CONFIRMED_SANCTIONS_MATCH"):
                behaviour_hits.append(result)

        # Sort confirmed/watchlist first
        results.sort(key=lambda x: x["confidence_score"], reverse=True)

        return {
            "total_screened": len(results),
            "watchlist_hits": len(watchlist_hits),
            "behavioural_matches": len(behaviour_hits),
            "total_alerts": len(watchlist_hits) + len(behaviour_hits),
            "screening_method": ["Watchlist (RBI/OFAC/UN)", "Behaviour-Based Fingerprint Matching"],
            "top_alerts": results[:15],
            "all_results": results,
            "screened_at": datetime.now().isoformat(),
        }
