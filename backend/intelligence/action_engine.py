"""
ChainVigil — Automated Action Engine  (STEP 5)

Why this exists:
  The existing ActionEngine (realtime/action_engine.py) is a 12-line stub
  with no structured output or explanation.  We need to close the loop
  by returning a complete decision object that the UI and audit trail can use.

Design:
  - Takes the final_risk_score from RootCauseEngine
  - Applies three-tier decision policy (Freeze / Manual Review / Safe)
  - Returns a typed, structured action response that includes score + explanation
  - Stateless — call it per account, no side effects
"""

from typing import Any, Dict, List, Optional


# ── Thresholds ────────────────────────────────────────────────────────────────

FREEZE_THRESHOLD = 0.85          # Immediate freeze
REVIEW_THRESHOLD = 0.70          # Manual review queue


# ── Action constants ──────────────────────────────────────────────────────────

class Actions:
    FREEZE  = "FREEZE"           # Block all transactions immediately
    REVIEW  = "MANUAL_REVIEW"    # Route to analyst queue
    MONITOR = "MONITOR"          # Passive watch — escalate if more signals
    SAFE    = "SAFE"             # No action required


# ── Playbooks ─────────────────────────────────────────────────────────────────

_PLAYBOOKS: Dict[str, Dict[str, Any]] = {
    Actions.FREEZE: {
        "urgency":     "CRITICAL",
        "sla_hours":   1,
        "steps": [
            "Immediately suspend all outgoing transactions",
            "Flag account in core banking system (CBS)",
            "Notify customer via registered mobile/email",
            "Escalate to Financial Intelligence Unit (FIU-IND) within 24 hours",
            "Initiate SAR (Suspicious Activity Report) generation",
        ],
    },
    Actions.REVIEW: {
        "urgency":     "HIGH",
        "sla_hours":   24,
        "steps": [
            "Place transaction hold pending analyst review",
            "Assign to AML compliance analyst queue",
            "Request KYC re-verification from customer",
            "Cross-reference with sanctions watchlist",
        ],
    },
    Actions.MONITOR: {
        "urgency":     "MEDIUM",
        "sla_hours":   72,
        "steps": [
            "Increase transaction monitoring frequency",
            "Set velocity alert threshold to 50% of current",
            "Flag for next scheduled AML review cycle",
        ],
    },
    Actions.SAFE: {
        "urgency":     "LOW",
        "sla_hours":   0,
        "steps": [],
    },
}


class AutomatedActionEngine:
    """
    Decides an action from the final risk score and returns a complete
    structured response for the API, UI, and audit trail.

    Usage:
        engine   = AutomatedActionEngine()
        response = engine.decide(
            account_id     = "ACC-042",
            final_score    = 0.91,
            explanation    = [...],
            evidence       = {...},
            hard_block     = False,
        )
    """

    def decide(
        self,
        account_id: str,
        final_score: float,
        explanation: Optional[List[str]] = None,
        evidence: Optional[Dict] = None,
        hard_block: bool = False,         # from rule engine (self-transfer etc.)
        risk_tier: Optional[str] = None,
    ) -> Dict[str, Any]:
        """
        Returns a structured action decision.

        Response schema:
        {
            "account_id":    str,
            "risk_score":    float,
            "risk_tier":     str,
            "action":        str,           # FREEZE | MANUAL_REVIEW | MONITOR | SAFE
            "urgency":       str,
            "sla_hours":     int,
            "explanation":   [str, ...],
            "playbook":      [str, ...],
            "evidence":      {raw scores},
        }
        """
        # Hard block overrides score
        if hard_block:
            action = Actions.FREEZE
        elif final_score >= FREEZE_THRESHOLD:
            action = Actions.FREEZE
        elif final_score >= REVIEW_THRESHOLD:
            action = Actions.REVIEW
        elif final_score >= 0.50:
            action = Actions.MONITOR
        else:
            action = Actions.SAFE

        playbook = _PLAYBOOKS[action]

        return {
            "account_id":   account_id,
            "risk_score":   round(final_score, 4),
            "risk_tier":    risk_tier or _tier(final_score),
            "action":       action,
            "urgency":      playbook["urgency"],
            "sla_hours":    playbook["sla_hours"],
            "explanation":  explanation or [],
            "playbook":     playbook["steps"],
            "evidence":     evidence or {},
        }


def _tier(score: float) -> str:
    if score >= 0.85: return "CRITICAL"
    if score >= 0.70: return "HIGH"
    if score >= 0.50: return "MEDIUM"
    return "LOW"
