"""
ChainVigil — Temporal Anomaly Detector  (STEP 2)

Why this exists:
  The existing RiskIntelligenceEngine computed cluster-level velocity but
  gave NO per-account temporal signal.  A mule account often shows a
  sharp burst of transactions over minutes — something GNNs miss because
  they work on graph structure, not time series.

What we add:
  - Per-account transaction timeline built from graph edges
  - Burst detection  (> N tx in X minutes)
  - Rapid fund-movement chains  (money leaving < T seconds after arrival)
  - Output: temporal_risk_score in [0, 1], attached to each account summary

Design philosophy:
  Lightweight — pure Python + networkx (already a dependency).
  No LSTM / RNN unless the model needs sequence embeddings later.
"""

from collections import defaultdict
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import networkx as nx


# ── Thresholds (tunable via env/config) ──────────────────────────────────────

BURST_COUNT_THRESHOLD = 5          # > 5 tx in the window = burst
BURST_WINDOW_MINUTES = 10          # within a 10-minute rolling window
RAPID_RELAY_SECONDS = 120          # money forwarded within 2 minutes of receipt


def _parse_ts(ts_str: Optional[str]) -> Optional[datetime]:
    """Parse ISO timestamp string, return UTC datetime or None."""
    if not ts_str:
        return None
    try:
        clean = ts_str.replace("Z", "+00:00")
        dt = datetime.fromisoformat(clean)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return None


class TemporalAnomalyDetector:
    """
    Scans the NetworkX graph for time-domain fraud signals per account.

    Usage:
        detector = TemporalAnomalyDetector(G)
        scores   = detector.score_all()   # {account_id: {...}}
    """

    def __init__(self, G: nx.MultiDiGraph):
        self.G = G

    # ── Public API ────────────────────────────────────────────────────────

    def score_all(self) -> Dict[str, Dict]:
        """
        Compute temporal risk for every Account node in the graph.
        Returns dict keyed by account_id.
        """
        results = {}
        account_nodes = [
            n for n, d in self.G.nodes(data=True)
            if d.get("entity_type") == "Account"
        ]
        for acc_id in account_nodes:
            results[acc_id] = self._score_account(acc_id)
        return results

    def score_account(self, account_id: str) -> Dict:
        """Score a single account."""
        return self._score_account(account_id)

    # ── Internal helpers ──────────────────────────────────────────────────

    def _score_account(self, acc_id: str) -> Dict:
        """Compute the temporal_risk_score and collect signals."""

        outgoing_ts = self._outgoing_timestamps(acc_id)
        incoming_ts = self._incoming_timestamps(acc_id)

        burst_flag, burst_detail = self._detect_burst(outgoing_ts)
        relay_flag, relay_detail = self._detect_rapid_relay(
            acc_id, incoming_ts, outgoing_ts
        )

        # Combine signals into a single score
        score = 0.0
        signals = []

        if burst_flag:
            score += 0.55
            signals.append(burst_detail)

        if relay_flag:
            score += 0.45
            signals.append(relay_detail)

        score = min(1.0, round(score, 4))

        return {
            "temporal_risk_score": score,
            "burst_detected": burst_flag,
            "rapid_relay_detected": relay_flag,
            "temporal_signals": signals,
            "outgoing_tx_count": len(outgoing_ts),
            "incoming_tx_count": len(incoming_ts),
        }

    def _outgoing_timestamps(self, acc_id: str) -> List[datetime]:
        """Collect sorted timestamps of outgoing TRANSFERRED_TO edges."""
        ts_list = []
        if acc_id not in self.G:
            return ts_list
        for _, _, data in self.G.out_edges(acc_id, data=True):
            if data.get("edge_type") == "TRANSFERRED_TO":
                dt = _parse_ts(data.get("timestamp"))
                if dt:
                    ts_list.append(dt)
        return sorted(ts_list)

    def _incoming_timestamps(self, acc_id: str) -> List[Tuple[datetime, str]]:
        """
        Collect (timestamp, sender_id) tuples for incoming transfers.
        Used for relay detection.
        """
        ts_list = []
        if acc_id not in self.G:
            return ts_list
        for src, _, data in self.G.in_edges(acc_id, data=True):
            if data.get("edge_type") == "TRANSFERRED_TO":
                dt = _parse_ts(data.get("timestamp"))
                if dt:
                    ts_list.append((dt, src))
        return sorted(ts_list, key=lambda x: x[0])

    def _detect_burst(
        self, timestamps: List[datetime]
    ) -> Tuple[bool, str]:
        """
        Sliding-window burst detection.

        A burst = more than BURST_COUNT_THRESHOLD transactions whose
        timestamps all fall within BURST_WINDOW_MINUTES of each other.
        """
        if len(timestamps) < BURST_COUNT_THRESHOLD:
            return False, ""

        window_seconds = BURST_WINDOW_MINUTES * 60
        for i in range(len(timestamps)):
            window_end = timestamps[i].timestamp() + window_seconds
            count_in_window = sum(
                1 for t in timestamps[i:]
                if t.timestamp() <= window_end
            )
            if count_in_window >= BURST_COUNT_THRESHOLD:
                return True, (
                    f"Burst: {count_in_window} outgoing tx within "
                    f"{BURST_WINDOW_MINUTES} minutes"
                )
        return False, ""

    def _detect_rapid_relay(
        self,
        acc_id: str,
        incoming: List[Tuple[datetime, str]],
        outgoing: List[datetime],
    ) -> Tuple[bool, str]:
        """
        Detect if funds are forwarded within RAPID_RELAY_SECONDS of receipt.
        Classic mule pattern: receive → immediately forward.
        """
        if not incoming or not outgoing:
            return False, ""

        relay_count = 0
        for in_dt, _ in incoming:
            for out_dt in outgoing:
                delta = (out_dt - in_dt).total_seconds()
                if 0 <= delta <= RAPID_RELAY_SECONDS:
                    relay_count += 1

        if relay_count >= 2:
            return True, (
                f"Rapid relay: {relay_count} outgoing tx within "
                f"{RAPID_RELAY_SECONDS}s of receiving funds"
            )
        return False, ""
