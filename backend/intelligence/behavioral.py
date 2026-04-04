"""
ChainVigil — Behavioral Risk Profiler  (STEP 3)

Why this exists:
  The GNN captures structural/topological risk but NOT the time-of-day or
  dormancy-reactivation patterns that compliance officers look for first.
  This module simulates behavioral biometrics without requiring ML:
    - account dormancy then sudden re-activation
    - unusual transaction hours (late night / early morning)
    - device / IP switching frequency

Design: rule-based feature extraction over graph node attributes + edges.
        Returns behavioral_risk_score in [0, 1] per account.
"""

from datetime import datetime, timezone, time as dtime
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx


# ── Thresholds ────────────────────────────────────────────────────────────────

DORMANCY_DAYS = 30             # account inactive > 30 days = potentially dormant
ODD_HOUR_START = 0             # 00:00 – 05:59 = odd-hours flag
ODD_HOUR_END = 6
DEVICE_SWITCH_THRESHOLD = 3    # > 3 unique devices/IPs per account = suspicious


def _parse_hour(ts_str: Optional[str]) -> Optional[int]:
    """Return hour-of-day (0-23) from ISO timestamp string."""
    if not ts_str:
        return None
    try:
        clean = ts_str.replace("Z", "+00:00")
        dt = datetime.fromisoformat(clean)
        return dt.hour
    except Exception:
        return None


def _parse_dt(ts_str: Optional[str]) -> Optional[datetime]:
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


class BehavioralProfiler:
    """
    Extracts behavioral risk signals per account from graph node attrs + edges.

    Usage:
        profiler = BehavioralProfiler(G)
        scores   = profiler.profile_all()   # {account_id: {...}}
    """

    def __init__(self, G: nx.MultiDiGraph):
        self.G = G

    # ── Public API ────────────────────────────────────────────────────────

    def profile_all(self) -> Dict[str, Dict]:
        results = {}
        account_nodes = [
            n for n, d in self.G.nodes(data=True)
            if d.get("entity_type") == "Account"
        ]
        for acc_id in account_nodes:
            results[acc_id] = self._profile_account(acc_id)
        return results

    def profile_account(self, account_id: str) -> Dict:
        return self._profile_account(account_id)

    # ── Internal logic ────────────────────────────────────────────────────

    def _profile_account(self, acc_id: str) -> Dict:
        node_data = self.G.nodes.get(acc_id, {})

        # 1. Dormancy → sudden activation
        dormancy_flag, dormancy_detail = self._check_dormancy(
            acc_id, node_data
        )

        # 2. Unusual transaction hours
        odd_hour_flag, odd_hour_detail = self._check_odd_hours(acc_id)

        # 3. Device / IP switching
        switch_flag, switch_detail = self._check_device_ip_switching(
            acc_id, node_data
        )

        # Weighted combination
        score = 0.0
        signals = []

        if dormancy_flag:
            score += 0.45
            signals.append(dormancy_detail)
        if odd_hour_flag:
            score += 0.30
            signals.append(odd_hour_detail)
        if switch_flag:
            score += 0.25
            signals.append(switch_detail)

        score = min(1.0, round(score, 4))

        return {
            "behavioral_risk_score": score,
            "dormancy_reactivation": dormancy_flag,
            "odd_hour_activity": odd_hour_flag,
            "device_ip_switching": switch_flag,
            "behavioral_signals": signals,
        }

    def _check_dormancy(
        self, acc_id: str, node_data: Dict
    ) -> Tuple[bool, str]:
        """
        Dormancy = account has no outgoing tx for a long period, then suddenly
        becomes active again.  We approximate this from the graph edge timestamps.
        """
        tx_timestamps = []
        if acc_id not in self.G:
            return False, ""

        for _, _, data in self.G.out_edges(acc_id, data=True):
            if data.get("edge_type") == "TRANSFERRED_TO":
                dt = _parse_dt(data.get("timestamp"))
                if dt:
                    tx_timestamps.append(dt)

        if len(tx_timestamps) < 2:
            return False, ""

        tx_timestamps.sort()
        # Find the maximum gap between consecutive transactions
        max_gap_days = 0.0
        for i in range(len(tx_timestamps) - 1):
            gap = (tx_timestamps[i + 1] - tx_timestamps[i]).total_seconds() / 86400
            if gap > max_gap_days:
                max_gap_days = gap

        if max_gap_days >= DORMANCY_DAYS:
            return True, (
                f"Dormancy reactivation: gap of {max_gap_days:.0f} days "
                f"detected between consecutive transactions"
            )
        return False, ""

    def _check_odd_hours(self, acc_id: str) -> Tuple[bool, str]:
        """
        Flag if > 30 % of transactions occur between midnight and 6 AM.
        """
        if acc_id not in self.G:
            return False, ""

        hours = []
        for _, _, data in self.G.out_edges(acc_id, data=True):
            if data.get("edge_type") == "TRANSFERRED_TO":
                h = _parse_hour(data.get("timestamp"))
                if h is not None:
                    hours.append(h)

        if not hours:
            return False, ""

        odd = sum(1 for h in hours if ODD_HOUR_START <= h < ODD_HOUR_END)
        ratio = odd / len(hours)

        if ratio >= 0.30:
            return True, (
                f"Odd-hour activity: {ratio * 100:.0f}% of transactions "
                f"between {ODD_HOUR_START:02d}:00–{ODD_HOUR_END:02d}:00"
            )
        return False, ""

    def _check_device_ip_switching(
        self, acc_id: str, node_data: Dict
    ) -> Tuple[bool, str]:
        """
        Use shared_device_count / shared_ip_count node attributes if available,
        or count unique USED_DEVICE / ACCESSED_FROM_IP  edges in the graph.
        """
        # Prefer pre-computed feature from graph node
        device_count = node_data.get("shared_device_count", 0)
        ip_count = node_data.get("shared_ip_count", 0)

        # Fallback: count distinct device / IP neighbours
        if device_count == 0 and ip_count == 0 and acc_id in self.G:
            for nbr, _, data in self.G.out_edges(acc_id, data=True):
                etype = data.get("edge_type", "")
                if etype == "USED_DEVICE":
                    device_count += 1
                elif etype == "ACCESSED_FROM_IP":
                    ip_count += 1

        total = device_count + ip_count
        if total >= DEVICE_SWITCH_THRESHOLD:
            return True, (
                f"Device/IP switching: {device_count} shared devices, "
                f"{ip_count} shared IPs (high switching frequency)"
            )
        return False, ""
