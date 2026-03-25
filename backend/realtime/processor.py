import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, List

from backend.realtime.action_engine import ActionEngine
from backend.realtime.blockchain import FraudLedger
from backend.realtime.rule_engine import RuleEngine


class DeviceIpIntel:
    def score(self, txn, G) -> Dict[str, Any]:
        score = 0.0
        reasons: List[str] = []

        if txn.device_id and txn.device_id in G:
            users = {
                src for src, _, d in G.in_edges(txn.device_id, data=True)
                if d.get("edge_type") == "USED_DEVICE"
            }
            if len(users) >= 6:
                score += 0.25
                reasons.append("device_shared_many_accounts")
            elif len(users) >= 3:
                score += 0.12
                reasons.append("device_shared_multiple_accounts")

        if txn.ip_address and txn.ip_address in G:
            users = {
                src for src, _, d in G.in_edges(txn.ip_address, data=True)
                if d.get("edge_type") == "LOGGED_FROM"
            }
            if len(users) >= 8:
                score += 0.25
                reasons.append("ip_shared_many_accounts")
            elif len(users) >= 4:
                score += 0.12
                reasons.append("ip_shared_multiple_accounts")

        return {
            "score": min(1.0, round(score, 4)),
            "reasons": reasons,
        }


class HybridScorer:
    def __init__(self, w_gnn: float = 0.55, w_rule: float = 0.3, w_intel: float = 0.15):
        self.w_gnn = w_gnn
        self.w_rule = w_rule
        self.w_intel = w_intel

    def combine(self, gnn_score: float, rule_score: float, intel_score: float) -> float:
        score = (
            self.w_gnn * gnn_score
            + self.w_rule * rule_score
            + self.w_intel * intel_score
        )
        return min(1.0, round(score, 4))


class RealTimeProcessor:
    """Processes a transaction with graph update -> scoring -> action -> ledger."""

    def __init__(self, graph_builder, graph, ledger_file: str, ledger_salt: str = "chainvigil-ledger-salt"):
        self.graph_builder = graph_builder
        self.G = graph
        self.ledger = FraudLedger(ledger_file)
        self.ledger_salt = ledger_salt
        self.rule_engine = RuleEngine()
        self.intel_engine = DeviceIpIntel()
        self.hybrid_scorer = HybridScorer()
        self.action_engine = ActionEngine()

    def _anon(self, account_id: str) -> str:
        return hashlib.sha256(f"{self.ledger_salt}:{account_id}".encode()).hexdigest()[:16]

    def _estimate_gnn_score(self, txn, risk_lookup: Dict[str, Dict[str, Any]]) -> float:
        src = risk_lookup.get(txn.source_id, {}).get("mule_probability")
        dst = risk_lookup.get(txn.target_id, {}).get("mule_probability")
        known = [s for s in [src, dst] if isinstance(s, (int, float))]
        if known:
            return round(float(max(known)), 4)
        return 0.5 if txn.amount < 25000 else 0.62

    def process(self, txn, risk_lookup: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        self.graph_builder.add_transaction_live(
            source_id=txn.source_id,
            target_id=txn.target_id,
            transaction_id=txn.transaction_id,
            amount=txn.amount,
            channel_type=txn.channel_type,
            timestamp=txn.timestamp or datetime.now(timezone.utc).isoformat(),
            geo_location=txn.geo_location,
            device_id=txn.device_id,
            ip_address=txn.ip_address,
            is_suspicious=False,
        )

        gnn_score = self._estimate_gnn_score(txn, risk_lookup)
        rule_out = self.rule_engine.evaluate(txn, self.G, risk_lookup)
        intel_out = self.intel_engine.score(txn, self.G)
        hybrid_score = self.hybrid_scorer.combine(gnn_score, rule_out["score"], intel_out["score"])
        decision = self.action_engine.decide(hybrid_score, hard_block=rule_out["hard_block"])

        reasons = list(dict.fromkeys(rule_out["reasons"] + intel_out["reasons"]))
        payload = {
            "transaction_id": txn.transaction_id,
            "source_anon": self._anon(txn.source_id),
            "target_anon": self._anon(txn.target_id),
            "amount": txn.amount,
            "channel_type": txn.channel_type,
            "gnn_score": gnn_score,
            "rule_score": rule_out["score"],
            "intel_score": intel_out["score"],
            "hybrid_score": hybrid_score,
            "decision": decision,
            "reasons": reasons,
        }
        ledger_tx_id = self.ledger.append(payload)

        return {
            "transaction_id": txn.transaction_id,
            "gnn_score": gnn_score,
            "rule_score": rule_out["score"],
            "intel_score": intel_out["score"],
            "hybrid_score": hybrid_score,
            "decision": decision,
            "reasons": reasons,
            "ledger_tx_id": ledger_tx_id,
            "processed_at": datetime.now(timezone.utc).isoformat(),
        }
