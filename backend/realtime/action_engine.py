class ActionEngine:
    """Maps hybrid fraud score to operational banking actions."""

    def decide(self, hybrid_score: float, hard_block: bool = False) -> str:
        if hard_block or hybrid_score >= 0.9:
            return "FREEZE"
        if hybrid_score >= 0.75:
            return "BLOCK"
        if hybrid_score >= 0.55:
            return "MONITOR"
        return "ALLOW"
