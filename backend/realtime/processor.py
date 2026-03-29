"""
ChainVigil — Real-Time Transaction Stream Processor

Provides:
1. SSE (Server-Sent Events) live feed endpoint — streams synthetic
   transactions to the frontend every 2 seconds, scored by rule engine
2. Transaction event scoring against rule thresholds
3. In-memory ring buffer of last 100 events for the "Live Feed" UI tab
"""

import asyncio
import json
import random
from datetime import datetime
from collections import deque
from typing import Dict, AsyncGenerator

# Rule thresholds for real-time pre-scoring
HIGH_AMOUNT_THRESHOLD = 500_000      # ₹5L+ = immediate flag
CTR_THRESHOLD = 1_000_000            # ₹10L CTR band
VELOCITY_FLAG_SECONDS = 300          # < 5 min between transactions = velocity flag
HIGH_RISK_CHANNELS = {"ATM", "UPI"}  # Channels with higher mule risk weight

CHANNELS = ["UPI", "ATM", "WEB", "MOBILE_APP", "NEFT", "RTGS", "IMPS"]
CHANNEL_RISK = {
    "ATM": 0.7, "UPI": 0.6, "MOBILE_APP": 0.5,
    "WEB": 0.4, "NEFT": 0.3, "RTGS": 0.3, "IMPS": 0.4,
}

# In-memory ring buffer of recent live events (last 100)
live_event_buffer: deque = deque(maxlen=100)

# Track account velocity (last tx timestamp per account)
_last_tx_time: Dict[str, datetime] = {}


def _generate_live_transaction() -> Dict:
    """Generate a realistic synthetic transaction for the live feed."""
    acc_count = 150
    sender = f"ACC-{random.randint(1, acc_count):03d}"
    receiver = f"ACC-{random.randint(1, acc_count):03d}"
    while receiver == sender:
        receiver = f"ACC-{random.randint(1, acc_count):03d}"

    channel = random.choices(
        CHANNELS,
        weights=[25, 20, 15, 15, 10, 10, 5],
        k=1
    )[0]

    # 5% chance of suspiciously high amount
    if random.random() < 0.05:
        amount = random.uniform(900_000, 999_000)   # Structuring band
    elif random.random() < 0.08:
        amount = random.uniform(500_000, 1_200_000)  # High value
    else:
        amount = random.uniform(1_000, 150_000)      # Normal

    return {
        "tx_id": f"TX-{datetime.now().strftime('%H%M%S')}-{random.randint(1000,9999)}",
        "sender_id": sender,
        "receiver_id": receiver,
        "amount": round(amount, 2),
        "channel": channel,
        "timestamp": datetime.now().isoformat(),
        "reference": f"REF{random.randint(100000, 999999)}",
    }


def _score_live_transaction(tx: Dict) -> Dict:
    """
    Real-time rule engine scoring of a live transaction.
    Returns risk flags, score, and severity level.
    """
    flags = []
    risk_score = 0.0
    amount = tx.get("amount", 0)
    channel = tx.get("channel", "")
    sender = tx.get("sender_id", "")
    now = datetime.now()

    # Rule 1: High amount
    if amount >= CTR_THRESHOLD:
        flags.append(f"CTR_THRESHOLD_BREACH: ₹{amount:,.0f} exceeds ₹10L reporting limit")
        risk_score += 0.5
    elif 900_000 <= amount < CTR_THRESHOLD:
        flags.append(f"STRUCTURING_ALERT: ₹{amount:,.0f} is in ₹9L–₹10L structuring band")
        risk_score += 0.45

    elif amount >= HIGH_AMOUNT_THRESHOLD:
        flags.append(f"HIGH_VALUE: ₹{amount:,.0f} exceeds ₹5L alert threshold")
        risk_score += 0.25

    # Rule 2: Channel risk weight
    cr = CHANNEL_RISK.get(channel, 0.2)
    risk_score += cr * 0.2

    # Rule 3: Velocity check
    if sender in _last_tx_time:
        elapsed = (now - _last_tx_time[sender]).total_seconds()
        if elapsed < VELOCITY_FLAG_SECONDS:
            flags.append(f"HIGH_VELOCITY: {elapsed:.0f}s since last tx from {sender}")
            risk_score += 0.3

    _last_tx_time[sender] = now

    # Clamp
    risk_score = min(1.0, risk_score)

    if risk_score >= 0.75:
        severity = "CRITICAL"
        color = "red"
    elif risk_score >= 0.45:
        severity = "HIGH"
        color = "orange"
    elif risk_score >= 0.2:
        severity = "MEDIUM"
        color = "yellow"
    else:
        severity = "LOW"
        color = "green"

    return {
        **tx,
        "risk_score": round(risk_score * 100, 1),
        "risk_flags": flags,
        "severity": severity,
        "color": color,
        "confidence": f"{min(99, int(risk_score * 100))}%",
        "channel_risk_weight": cr,
    }


async def live_transaction_generator() -> AsyncGenerator[str, None]:
    """
    Async generator that yields scored transactions as SSE events.
    Used by the /api/stream/live endpoint.
    """
    while True:
        tx = _generate_live_transaction()
        scored = _score_live_transaction(tx)
        live_event_buffer.appendleft(scored)

        # SSE format: data: <json>\n\n
        sse_data = f"data: {json.dumps(scored)}\n\n"
        yield sse_data

        await asyncio.sleep(2)  # One event every 2 seconds


def get_recent_events(limit: int = 50) -> list:
    """Return recent live events from ring buffer."""
    return list(live_event_buffer)[:limit]
