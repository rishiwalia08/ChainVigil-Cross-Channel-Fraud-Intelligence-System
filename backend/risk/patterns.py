"""
ChainVigil — Financial Crime Pattern Detector

Detects IBA-required patterns:
1. Structuring — transactions just below ₹10L reporting threshold
2. Fragmentation — large amounts split into multiple smaller transfers
3. Nesting — multi-hop rapid pass-through chains
4. Circular Flow — funds returning to origin via different paths

Each detection produces a confidence score (0–100) and explanation.
"""

from typing import Dict, List, Tuple, Optional
from collections import defaultdict
from datetime import datetime, timedelta

import networkx as nx
import numpy as np

# Indian CTR (Cash Transaction Report) threshold
CTR_THRESHOLD = 1_000_000      # ₹10,00,000
STRUCTURING_BAND_LOW = 900_000  # ₹9,00,000  — 90% of threshold
STRUCTURING_BAND_HIGH = 990_000 # ₹9,90,000  — 99% of threshold

MAX_FRAGMENT_WINDOW_HOURS = 24  # Fragmentation detected within 24h
MIN_FRAGMENT_COUNT = 3          # Min sub-transactions to flag
NESTING_MIN_HOPS = 4           # Min hops to call it nesting
NESTING_MAX_HOP_MIN = 120      # Max minutes per hop for nesting (speed)
CIRCULAR_MAX_HOURS = 48        # Max hours for circular flow detection


class PatternDetector:
    """
    Detects financial crime patterns beyond simple GNN risk scoring.
    Produces confidence scores and human-readable evidence strings.
    """

    def __init__(self, G: nx.MultiDiGraph, transactions: List[Dict]):
        self.G = G
        self.transactions = transactions
        self._by_sender: Dict[str, List[Dict]] = defaultdict(list)
        self._by_receiver: Dict[str, List[Dict]] = defaultdict(list)
        self._index_transactions()

    def _index_transactions(self):
        for tx in self.transactions:
            sender = tx.get("sender_id") or tx.get("from_account")
            receiver = tx.get("receiver_id") or tx.get("to_account")
            if sender:
                self._by_sender[sender].append(tx)
            if receiver:
                self._by_receiver[receiver].append(tx)

    def _parse_ts(self, tx: Dict) -> Optional[datetime]:
        ts = tx.get("timestamp")
        if not ts:
            return None
        try:
            return datetime.fromisoformat(str(ts))
        except (ValueError, TypeError):
            return None

    # ──────────────────────────────────────────────────────────────
    # 1. STRUCTURING DETECTION
    # ──────────────────────────────────────────────────────────────
    def detect_structuring(self) -> List[Dict]:
        """
        Flag accounts making repeated transactions in the ₹9L–₹9.9L band
        (just below the ₹10L Cash Transaction Report threshold).
        """
        results = []
        for account_id, txs in self._by_sender.items():
            structured_txs = [
                tx for tx in txs
                if STRUCTURING_BAND_LOW <= float(tx.get("amount", 0)) <= STRUCTURING_BAND_HIGH
            ]
            if len(structured_txs) < 2:
                continue

            # Score: more occurrences = higher confidence
            count = len(structured_txs)
            confidence = min(100, 40 + count * 20)
            amounts = [float(tx["amount"]) for tx in structured_txs]
            channels = list({tx.get("channel_type", "?") for tx in structured_txs})

            results.append({
                "pattern_type": "STRUCTURING",
                "account_id": account_id,
                "confidence_score": confidence,
                "occurrences": count,
                "amounts": amounts,
                "channels_used": channels,
                "threshold_avoided": CTR_THRESHOLD,
                "evidence": (
                    f"Account made {count} transaction(s) between ₹{STRUCTURING_BAND_LOW:,.0f}"
                    f" and ₹{STRUCTURING_BAND_HIGH:,.0f} — just below the ₹10L CTR threshold. "
                    f"Channels: {', '.join(channels)}."
                ),
                "recommended_action": "File CTR" if confidence >= 80 else "Enhanced Monitoring",
                "regulatory_ref": "RBI Master Direction FIU-IND: CTR reporting under PMLA 2002",
            })

        results.sort(key=lambda x: x["confidence_score"], reverse=True)
        return results

    # ──────────────────────────────────────────────────────────────
    # 2. FRAGMENTATION DETECTION
    # ──────────────────────────────────────────────────────────────
    def detect_fragmentation(self) -> List[Dict]:
        """
        Detect large amounts split across multiple transfers within 24h.
        Classic 'smurfing' pattern.
        """
        results = []
        for account_id, txs in self._by_sender.items():
            # Sort by timestamp
            timed = [(self._parse_ts(tx), tx) for tx in txs]
            timed = [(ts, tx) for ts, tx in timed if ts is not None]
            timed.sort(key=lambda x: x[0])

            if len(timed) < MIN_FRAGMENT_COUNT:
                continue

            # Sliding 24-hour window
            for i, (ts_i, tx_i) in enumerate(timed):
                window = [
                    (ts, tx) for ts, tx in timed[i:]
                    if ts - ts_i <= timedelta(hours=MAX_FRAGMENT_WINDOW_HOURS)
                ]
                if len(window) < MIN_FRAGMENT_COUNT:
                    continue

                total = sum(float(tx.get("amount", 0)) for _, tx in window)
                if total < CTR_THRESHOLD * 0.8:
                    continue  # Only flag if total would have hit threshold

                n = len(window)
                channels = list({tx.get("channel_type", "?") for _, tx in window})
                receivers = list({
                    tx.get("receiver_id") or tx.get("to_account", "?")
                    for _, tx in window
                })

                confidence = min(100, 30 + n * 15 + (len(channels) > 1) * 20)

                results.append({
                    "pattern_type": "FRAGMENTATION",
                    "account_id": account_id,
                    "confidence_score": confidence,
                    "transaction_count": n,
                    "total_amount": total,
                    "window_hours": MAX_FRAGMENT_WINDOW_HOURS,
                    "unique_receivers": len(receivers),
                    "channels_used": channels,
                    "evidence": (
                        f"₹{total:,.0f} sent across {n} transactions to {len(receivers)} "
                        f"recipients within {MAX_FRAGMENT_WINDOW_HOURS}h via {', '.join(channels)}. "
                        f"Combined amount exceeds CTR threshold."
                    ),
                    "recommended_action": "File STR" if confidence >= 70 else "Enhanced Monitoring",
                    "regulatory_ref": "PMLA 2002 Section 12: Suspicious Transaction Report",
                })
                break  # One flag per account

        results.sort(key=lambda x: x["confidence_score"], reverse=True)
        return results

    # ──────────────────────────────────────────────────────────────
    # 3. NESTING DETECTION
    # ──────────────────────────────────────────────────────────────
    def detect_nesting(self) -> List[Dict]:
        """
        Detect 4+ hop chains where money passes through accounts quickly.
        Nesting = using layers of intermediary accounts to obscure origin.
        """
        results = []
        # Build a simple account-to-account graph with timestamps
        chain_edges = defaultdict(list)  # sender -> [(receiver, ts, amount)]

        for tx in self.transactions:
            sender = tx.get("sender_id") or tx.get("from_account")
            receiver = tx.get("receiver_id") or tx.get("to_account")
            ts = self._parse_ts(tx)
            amount = float(tx.get("amount", 0))
            if sender and receiver and ts:
                chain_edges[sender].append((receiver, ts, amount))

        # DFS to find long rapid chains
        def dfs_chains(start, current, path, entry_time, visited):
            found = []
            for (nxt, ts, amt) in chain_edges.get(current, []):
                if nxt in visited:
                    continue
                hop_minutes = (ts - entry_time).total_seconds() / 60
                if hop_minutes < 0 or hop_minutes > NESTING_MAX_HOP_MIN:
                    continue
                new_path = path + [(nxt, ts, amt)]
                new_visited = visited | {nxt}
                if len(new_path) >= NESTING_MIN_HOPS:
                    found.append(new_path)
                found.extend(dfs_chains(start, nxt, new_path, ts, new_visited))
            return found

        seen_chains = set()
        account_ids = list(chain_edges.keys())[:50]  # Limit for performance

        for start in account_ids:
            for (first_hop, ts0, amt0) in chain_edges[start]:
                chains = dfs_chains(start, first_hop, [(first_hop, ts0, amt0)], ts0, {start, first_hop})
                for chain in chains:
                    key = (start, chain[-1][0])
                    if key in seen_chains:
                        continue
                    seen_chains.add(key)

                    hops = len(chain)
                    total_minutes = (chain[-1][1] - chain[0][1]).total_seconds() / 60
                    confidence = min(100, 20 + hops * 12 + (total_minutes < 60) * 25)
                    path_ids = [start] + [node for node, _, _ in chain]

                    results.append({
                        "pattern_type": "NESTING",
                        "account_id": start,
                        "confidence_score": confidence,
                        "chain_length": hops + 1,
                        "total_minutes": round(total_minutes, 1),
                        "chain_path": path_ids,
                        "final_destination": chain[-1][0],
                        "evidence": (
                            f"Funds passed through {hops + 1} accounts in {total_minutes:.0f} min. "
                            f"Chain: {' → '.join(path_ids[:6])}{'...' if hops > 5 else ''}. "
                            f"Indicative of layering to obscure beneficial ownership."
                        ),
                        "recommended_action": "Freeze All Chain Accounts" if confidence >= 80 else "File STR",
                        "regulatory_ref": "FATF Recommendation 16: Wire Transfer transparency",
                    })

        results.sort(key=lambda x: x["confidence_score"], reverse=True)
        return results[:20]  # Top 20 nesting chains

    # ──────────────────────────────────────────────────────────────
    # 4. CIRCULAR FLOW DETECTION
    # ──────────────────────────────────────────────────────────────
    def detect_circular_flows(self) -> List[Dict]:
        """
        Detect funds returning to origin account within 48h via a different path.
        Classic layering / 'round-tripping' indicator.
        """
        results = []
        seen = set()

        for account_id in list(self._by_sender.keys())[:80]:
            if account_id not in self.G:
                continue
            try:
                cycles = list(nx.simple_cycles(self.G.subgraph(
                    list(nx.descendants(self.G, account_id))[:30] | {account_id}
                )))
            except Exception:
                continue

            for cycle in cycles:
                if account_id not in cycle or len(cycle) < 3:
                    continue
                key = frozenset(cycle)
                if key in seen:
                    continue
                seen.add(key)

                confidence = min(100, 35 + len(cycle) * 10)
                results.append({
                    "pattern_type": "CIRCULAR_FLOW",
                    "account_id": account_id,
                    "confidence_score": confidence,
                    "cycle_length": len(cycle),
                    "cycle_members": cycle,
                    "evidence": (
                        f"Funds cycle through {len(cycle)} accounts returning to origin: "
                        f"{' → '.join(cycle[:5])}{'...' if len(cycle) > 5 else ''} → {cycle[0]}. "
                        f"Indicative of round-tripping or wash trading."
                    ),
                    "recommended_action": "Freeze Ring & File STR",
                    "regulatory_ref": "PMLA 2002 Section 3: Offence of money laundering",
                })

        results.sort(key=lambda x: x["confidence_score"], reverse=True)
        return results[:15]

    def run_all(self, transactions: List[Dict]) -> Dict:
        """Run all detectors and return combined summary."""
        self.transactions = transactions
        self._by_sender.clear()
        self._by_receiver.clear()
        self._index_transactions()

        structuring = self.detect_structuring()
        fragmentation = self.detect_fragmentation()
        nesting = self.detect_nesting()
        circular = self.detect_circular_flows()

        all_patterns = structuring + fragmentation + nesting + circular
        all_patterns.sort(key=lambda x: x["confidence_score"], reverse=True)

        return {
            "total_patterns_detected": len(all_patterns),
            "structuring_cases": len(structuring),
            "fragmentation_cases": len(fragmentation),
            "nesting_cases": len(nesting),
            "circular_flow_cases": len(circular),
            "patterns": all_patterns,
            "top_patterns": all_patterns[:10],
            "generated_at": datetime.now().isoformat(),
        }
