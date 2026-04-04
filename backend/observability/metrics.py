"""
ChainVigil — Observability / Metrics Layer  (STEP 1)

Why this exists:
  The system had zero introspection into its own health.
  This module adds a cheap, dependency-free metrics store that:
    - Tracks fraud rate, model latency, event counts
    - Exposes a /metrics endpoint (Prometheus text OR JSON)
    - Provides structured logging helpers for transactions + predictions

Design: plain Python (no Prometheus client lib required).
        Can be upgraded to prometheus_client later with zero logic changes.
"""

import time
import threading
from collections import defaultdict, deque
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional


# ─── Singleton metrics registry ─────────────────────────────────────────────

class MetricsRegistry:
    """
    Thread-safe, in-process metrics store.

    Counters  — monotonically increasing integers (e.g. total_predictions)
    Gauges    — current floating point values    (e.g. fraud_rate)
    Histograms— ring-buffer of recent samples   (e.g. inference latency ms)
    """

    def __init__(self, histogram_window: int = 500):
        self._lock = threading.Lock()
        self._counters: Dict[str, int] = defaultdict(int)
        self._gauges: Dict[str, float] = {}
        self._histograms: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=histogram_window)
        )
        self._start_time: float = time.time()

        # Pre-declare the metrics we care about
        self._gauges["fraud_rate"] = 0.0          # fraction of flagged accounts
        self._gauges["model_auc"] = 0.0            # last training AUC
        self._gauges["accounts_analyzed"] = 0.0
        self._gauges["flagged_accounts"] = 0.0
        self._gauges["clusters_detected"] = 0.0

    # ── Counters ──────────────────────────────────────────────────────────

    def inc(self, name: str, amount: int = 1) -> None:
        with self._lock:
            self._counters[name] += amount

    def counter(self, name: str) -> int:
        return self._counters.get(name, 0)

    # ── Gauges ───────────────────────────────────────────────────────────

    def set_gauge(self, name: str, value: float) -> None:
        with self._lock:
            self._gauges[name] = value

    def gauge(self, name: str) -> float:
        return self._gauges.get(name, 0.0)

    # ── Histograms ───────────────────────────────────────────────────────

    def observe(self, name: str, value: float) -> None:
        """Record one sample (e.g. latency in ms)."""
        with self._lock:
            self._histograms[name].append(value)

    def histogram_stats(self, name: str) -> Dict[str, float]:
        data = list(self._histograms.get(name, []))
        if not data:
            return {"count": 0, "mean": 0, "p50": 0, "p95": 0, "p99": 0}
        data_sorted = sorted(data)
        n = len(data_sorted)
        def pct(p): return data_sorted[min(int(n * p / 100), n - 1)]
        return {
            "count": n,
            "mean": round(sum(data_sorted) / n, 3),
            "p50": round(pct(50), 3),
            "p95": round(pct(95), 3),
            "p99": round(pct(99), 3),
        }

    # ── Snapshot ─────────────────────────────────────────────────────────

    def snapshot(self) -> Dict[str, Any]:
        """Return a full metrics snapshot (used for /metrics endpoint)."""
        uptime_seconds = round(time.time() - self._start_time, 1)
        return {
            "uptime_seconds": uptime_seconds,
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "counters": dict(self._counters),
            "gauges": dict(self._gauges),
            "latency_ms": {
                "inference": self.histogram_stats("inference_latency_ms"),
                "pipeline": self.histogram_stats("pipeline_latency_ms"),
                "live_scoring": self.histogram_stats("live_scoring_latency_ms"),
            },
        }

    def prometheus_text(self) -> str:
        """
        Emit metrics in Prometheus exposition format.
        Paste this into Grafana's Prometheus datasource or `curl /metrics`.
        """
        lines = []
        snap = self.snapshot()

        lines.append(f"# ChainVigil metrics — {snap['generated_at']}")
        lines.append(f"chainvigil_uptime_seconds {snap['uptime_seconds']}")

        for name, val in snap["counters"].items():
            lines.append(f"chainvigil_counter_{name} {val}")

        for name, val in snap["gauges"].items():
            lines.append(f"chainvigil_gauge_{name} {round(val, 6)}")

        for scope, stats in snap["latency_ms"].items():
            for stat, val in stats.items():
                lines.append(
                    f"chainvigil_latency_{scope}_{stat} {val}"
                )

        return "\n".join(lines) + "\n"


# ── Global singleton ──────────────────────────────────────────────────────────

METRICS = MetricsRegistry()


# ── Context manager for latency tracking ─────────────────────────────────────

class _Timer:
    """Usage:  with timer("inference_latency_ms"): ..."""

    def __init__(self, metric_name: str):
        self._name = metric_name
        self._start: float = 0.0

    def __enter__(self):
        self._start = time.perf_counter()
        return self

    def __exit__(self, *_):
        elapsed_ms = (time.perf_counter() - self._start) * 1000
        METRICS.observe(self._name, elapsed_ms)


def timer(metric_name: str) -> _Timer:
    return _Timer(metric_name)


# ── Structured log helpers ───────────────────────────────────────────────────

def log_prediction(
    account_id: str,
    risk_score: float,
    action: str,
    source: str = "gnn",
) -> None:
    """
    Emit one structured log line per model prediction.
    Can be wired to any logger (stdout, file, ELK, etc.)
    """
    entry = {
        "event": "prediction",
        "ts": datetime.now(timezone.utc).isoformat(),
        "account_id": account_id,
        "risk_score": round(risk_score, 4),
        "action": action,
        "source": source,
    }
    _safe_print(entry)

    # Update counters
    METRICS.inc("total_predictions")
    if action in ("Freeze", "freeze", "FREEZE"):
        METRICS.inc("predictions_freeze")
    elif action in ("Escalate", "escalate", "BLOCK"):
        METRICS.inc("predictions_escalate")
    elif action in ("Monitor", "monitor"):
        METRICS.inc("predictions_monitor")
    else:
        METRICS.inc("predictions_clear")


def log_transaction(
    tx_id: str,
    sender: str,
    receiver: str,
    amount: float,
    channel: str,
    risk_score: float,
    flags: Optional[List[str]] = None,
) -> None:
    """Emit one structured log line per live transaction."""
    entry = {
        "event": "transaction",
        "ts": datetime.now(timezone.utc).isoformat(),
        "tx_id": tx_id,
        "sender": sender,
        "receiver": receiver,
        "amount": amount,
        "channel": channel,
        "risk_score": round(risk_score, 4),
        "flags": flags or [],
    }
    _safe_print(entry)
    METRICS.inc("total_transactions_scored")
    if risk_score >= 0.75:
        METRICS.inc("high_risk_transactions")


def _safe_print(entry: Dict) -> None:
    """JSON-log to stdout. Replace with `logger.info(entry)` in production."""
    import json
    try:
        print(f"[METRICS] {json.dumps(entry)}")
    except Exception:
        pass
