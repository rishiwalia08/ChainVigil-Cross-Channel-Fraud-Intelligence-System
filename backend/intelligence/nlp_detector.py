"""
ChainVigil — Lightweight NLP Fraud Detector  (STEP 6)

Why this exists:
  Transaction notes / references often contain social-engineering cues
  ("urgent payment", "lottery win", "advance fee") or structuring hints
  ("broken into", "split amount", "avoid tax") that purely numeric models miss.

Design:
  - Keyword + pattern based (no LLM API calls required)
  - Optional: embedding similarity via TF-IDF cosine distance if scikit-learn
    is available (it IS in the existing requirements.txt)
  - Returns: {is_suspicious, nlp_risk_score, matched_patterns, categories}

  Keeping it dependency-free first — TF-IDF upgrade is clearly marked.
"""

import re
from typing import Dict, List, Optional, Tuple


# ── Fraud indicator lexicon ───────────────────────────────────────────────────
#
#   Each entry: (regex_pattern, category, weight)
#   Weight: contribution to nlp_risk_score [0, 1]

_PATTERNS: List[Tuple[str, str, float]] = [
    # Social engineering / advance fee scams
    (r"\burgent\b",              "urgency_cue",         0.20),
    (r"\badvance\s*fee\b",       "advance_fee",         0.60),
    (r"\blottery\b",             "lottery_scam",        0.65),
    (r"\bwinnings?\b",           "lottery_scam",        0.50),
    (r"\binheritance\b",         "advance_fee",         0.55),
    (r"\bbusiness\s*opportunit", "advance_fee",         0.40),
    (r"\bsend\s*money\b",        "money_transfer_cue",  0.30),
    (r"\btransfer\s*now\b",      "urgency_cue",         0.25),

    # Structuring / smurfing hints
    (r"\bsplit\b",               "structuring",         0.35),
    (r"\bbroken\s+into\b",       "structuring",         0.40),
    (r"\bavoid\s*tax\b",         "structuring",         0.70),
    (r"\bunder\s*reporting\b",   "structuring",         0.65),
    (r"\bcash\s*only\b",         "cash_preference",     0.30),
    (r"\bno\s*receipt\b",        "cash_preference",     0.35),

    # Money laundering cues
    (r"\blayer\b",               "layering",            0.45),
    (r"\bwash\b",                "laundering",          0.50),
    (r"\bclean\b",               "laundering",          0.20),
    (r"\bshell\s*compan",        "shell_entity",        0.70),
    (r"\bnominee\b",             "nominee_mule",        0.60),
    (r"\bmule\b",                "nominee_mule",        0.85),

    # Coercion / job-ad mule
    (r"\bwork\s*from\s*home\b",  "job_scam",            0.30),
    (r"\beasy\s*money\b",        "job_scam",            0.45),
    (r"\bcommission\b",          "job_scam",            0.20),
    (r"\brecruit\b",             "recruitment",         0.15),

    # Terror / high risk
    (r"\bdonat(?:e|ion)\b",      "charity_scam",        0.25),
    (r"\bterror\b",              "terror_flag",         0.95),
    (r"\bfund(?:ing)?\s*jiha",   "terror_flag",         1.00),

    # Generic red flags
    (r"\btest\s*transfer\b",     "test_transaction",    0.30),
    (r"\brefund\b",              "refund_scam",         0.20),
]

_COMPILED = [
    (re.compile(pat, re.IGNORECASE), cat, weight)
    for pat, cat, weight in _PATTERNS
]

# Score cap: multiple matches don't exceed 1.0
_MAX_SCORE = 1.0


class NLPFraudDetector:
    """
    Lightweight keyword + regex NLP fraud detector.

    Usage:
        detector = NLPFraudDetector()
        result   = detector.analyze("urgent transfer, advance fee required")
    """

    def analyze(self, text: Optional[str]) -> Dict:
        """
        Analyze a transaction description / note for fraud signals.

        Args:
            text: raw transaction note, reference, or description

        Returns:
            {
              "is_suspicious":    bool,
              "nlp_risk_score":   float,      # [0, 1]
              "matched_patterns": [str],      # categories matched
              "matched_terms":    [str],      # actual text snippets matched
              "categories":       {cat: weight}
            }
        """
        if not text or not isinstance(text, str) or not text.strip():
            return self._empty_result()

        matched_patterns = []
        matched_terms = []
        total_score = 0.0
        category_scores: Dict[str, float] = {}

        for compiled_re, category, weight in _COMPILED:
            m = compiled_re.search(text)
            if m:
                matched_patterns.append(category)
                matched_terms.append(m.group(0))
                category_scores[category] = max(
                    category_scores.get(category, 0), weight
                )
                total_score += weight

        # Cap and normalise
        total_score = min(_MAX_SCORE, round(total_score, 4))
        is_suspicious = total_score >= 0.30

        return {
            "is_suspicious":   is_suspicious,
            "nlp_risk_score":  total_score,
            "matched_patterns": list(set(matched_patterns)),
            "matched_terms":   list(set(matched_terms)),
            "categories":      category_scores,
        }

    def analyze_batch(self, texts: List[Optional[str]]) -> List[Dict]:
        """Analyze a list of transaction notes."""
        return [self.analyze(t) for t in texts]

    @staticmethod
    def _empty_result() -> Dict:
        return {
            "is_suspicious":   False,
            "nlp_risk_score":  0.0,
            "matched_patterns": [],
            "matched_terms":   [],
            "categories":      {},
        }
