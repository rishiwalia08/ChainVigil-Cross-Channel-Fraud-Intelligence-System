from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List


def parse_ts(ts: str | None) -> datetime:
    if not ts:
        return datetime.now(timezone.utc)
    try:
        clean = ts.replace("Z", "+00:00")
        dt = datetime.fromisoformat(clean)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return datetime.now(timezone.utc)


class RuleEngine:
    """Deterministic rules for banking-grade risk scoring."""

    def evaluate(self, txn, G, risk_lookup: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        score = 0.0
        reasons: List[str] = []
        hard_block = False

        if txn.source_id == txn.target_id:
            hard_block = True
            score = 1.0
            reasons.append("self_transfer_detected")

        if txn.amount >= 100000:
            score += 0.35
            reasons.append("very_high_amount")
        elif txn.amount >= 50000:
            score += 0.2
            reasons.append("high_amount")

        if txn.channel_type.upper() == "ATM" and txn.amount >= 20000:
            score += 0.1
            reasons.append("high_value_atm")

        src_risk = risk_lookup.get(txn.source_id, {}).get("mule_probability", 0.0)
        dst_risk = risk_lookup.get(txn.target_id, {}).get("mule_probability", 0.0)
        if max(src_risk, dst_risk) >= 0.85:
            score += 0.3
            reasons.append("known_high_risk_party")
        elif max(src_risk, dst_risk) >= 0.6:
            score += 0.15
            reasons.append("known_medium_risk_party")

        # Velocity: source transferred repeatedly in short window
        now_ts = parse_ts(txn.timestamp)
        recent_cutoff = now_ts - timedelta(minutes=10)
        recent_outgoing = 0
        if txn.source_id in G:
            for _, _, edge in G.out_edges(txn.source_id, data=True):
                if edge.get("edge_type") != "TRANSFERRED_TO":
                    continue
                edge_ts = parse_ts(edge.get("timestamp"))
                if edge_ts >= recent_cutoff:
                    recent_outgoing += 1

        if recent_outgoing >= 5:
            score += 0.25
            reasons.append("rapid_outgoing_velocity")
        elif recent_outgoing >= 3:
            score += 0.12
            reasons.append("moderate_outgoing_velocity")

        # Circular flow approximation (2-hop return path target->source)
        has_return_path = False
        if txn.target_id in G and txn.source_id in G:
            for nbr in G.successors(txn.target_id):
                if nbr == txn.source_id:
                    has_return_path = True
                    break
        if has_return_path:
            score += 0.2
            reasons.append("circular_flow_signal")

        return {
            "score": min(1.0, round(score, 4)),
            "hard_block": hard_block,
            "reasons": reasons,
        }
