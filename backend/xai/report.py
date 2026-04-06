"""
ChainVigil — FIU-IND Regulator-Ready SAR Report Generator

Produces Suspicious Activity Reports (SAR) in a format aligned with:
- FIU-IND (Financial Intelligence Unit — India) reporting guidelines
- PMLA (Prevention of Money Laundering Act) 2002 requirements
- RBI Master Direction on KYC

SAR includes confidence scores, pattern evidence, and recommended regulatory actions.
"""

import json
import hashlib
from typing import Dict, List, Optional
from datetime import datetime

# Reporting entity defaults (override via env in production)
REPORTING_ENTITY = "ChainVigil Fraud Intelligence Platform"
REPORTING_BANK = "Participating Member Bank"


def _confidence_label(score: float) -> str:
    if score >= 90: return "VERY HIGH"
    if score >= 70: return "HIGH"
    if score >= 50: return "MEDIUM"
    return "LOW"


def _generate_sar_reference() -> str:
    """Generate a unique SAR reference number."""
    ts = datetime.now().strftime("%Y%m%d%H%M%S")
    return f"SAR-CHAINVIGIL-{ts}"


def generate_sar_report(
    risk_summary: Dict,
    patterns: Optional[Dict] = None,
    sanctions: Optional[Dict] = None,
    graph_stats: Optional[Dict] = None,
    xai_by_account: Optional[Dict[str, Dict]] = None,
) -> Dict:
    """
    Generate a comprehensive FIU-IND aligned SAR report.

    Args:
        risk_summary: Output from RiskIntelligenceEngine.analyze()
        patterns: Output from PatternDetector.run_all()
        sanctions: Output from SanctionsScreener.screen_all()
        graph_stats: Graph statistics dictionary

    Returns:
        FIU-IND formatted SAR report as a dictionary
    """
    now = datetime.now()
    sar_ref = _generate_sar_reference()

    # ── Section 1: Report Header ──────────────────────────────────
    header = {
        "sar_reference_number": sar_ref,
        "report_type": "Suspicious Activity Report (SAR)",
        "reporting_entity": REPORTING_ENTITY,
        "reporting_bank": REPORTING_BANK,
        "generated_at": now.isoformat(),
        "report_date": now.strftime("%Y-%m-%d"),
        "reporting_period": {
            "from": now.strftime("%Y-%m-01"),
            "to": now.strftime("%Y-%m-%d"),
        },
        "regulatory_framework": [
            "Prevention of Money Laundering Act (PMLA) 2002",
            "PMLA (Maintenance of Records) Rules 2005",
            "FIU-IND Reporting Format v3.0",
            "RBI Master Direction on KYC (2016, updated 2023)",
            "FATF Recommendations 16, 20, 40",
        ],
        "classification": "CONFIDENTIAL — REGULATORY SUBMISSION",
    }

    # ── Section 2: Executive Summary ─────────────────────────────
    flagged = risk_summary.get("flagged_accounts", 0)
    clusters = risk_summary.get("clusters_detected", 0)
    total_analyzed = risk_summary.get("total_accounts_analyzed", 0)
    pattern_count = patterns.get("total_patterns_detected", 0) if patterns else 0
    sanctions_alerts = sanctions.get("total_alerts", 0) if sanctions else 0

    overall_confidence = min(99, int(
        (flagged / max(total_analyzed, 1)) * 100 * 0.4 +
        (min(clusters, 10) / 10) * 100 * 0.3 +
        (min(pattern_count, 20) / 20) * 100 * 0.3
    ))

    executive_summary = {
        "total_accounts_analyzed": total_analyzed,
        "mule_accounts_flagged": flagged,
        "mule_ring_clusters_detected": clusters,
        "financial_crime_patterns_detected": pattern_count,
        "sanctions_alerts": sanctions_alerts,
        "overall_risk_confidence_score": overall_confidence,
        "overall_risk_confidence_label": _confidence_label(overall_confidence),
        "recommended_regulatory_actions": _get_recommended_actions(
            flagged, clusters, pattern_count, sanctions_alerts
        ),
        "priority_level": "CRITICAL" if overall_confidence >= 80 else "HIGH" if overall_confidence >= 60 else "MEDIUM",
    }

    # ── Section 3: Suspicious Subjects ───────────────────────────
    high_risk_accounts = risk_summary.get("high_risk_accounts", [])
    subjects = []
    for acc in high_risk_accounts[:20]:
        acc_id = acc.get("account_id", "")
        mule_prob = acc.get("mule_probability", 0)
        confidence_score = round(mule_prob * 100, 1)

        # Check if in any pattern
        matched_patterns = []
        if patterns:
            for p in patterns.get("patterns", []):
                if p.get("account_id") == acc_id:
                    matched_patterns.append(p.get("pattern_type"))

        # Check sanctions
        sanctions_flag = False
        if sanctions:
            for s in sanctions.get("top_alerts", []):
                if s.get("account_id") == acc_id and s.get("match_type") != "CLEAR":
                    sanctions_flag = True
                    break

        # Anonymize for inter-bank sharing
        anon_id = hashlib.sha256(acc_id.encode()).hexdigest()[:16].upper()

        xai_payload = (xai_by_account or {}).get(acc_id, {})
        xai_subject = None
        if xai_payload:
            xai_subject = {
                "confidence_score": xai_payload.get("confidence_score"),
                "xai_reasoning": xai_payload.get("xai_reasoning"),
                "feature_attributions": xai_payload.get("feature_attributions", [])[:5],
            }

        subjects.append({
            "subject_reference": f"SUBJ-{anon_id}",
            "account_id_hash": anon_id,
            "risk_score": confidence_score,
            "confidence_label": _confidence_label(confidence_score),
            "suspicious_activity_types": matched_patterns or ["MULE_NETWORK"],
            "sanctions_alert": sanctions_flag,
            "cluster_id": acc.get("cluster_id"),
            "recommended_action": acc.get("recommended_action", "Monitor"),
            "regulatory_filing": _get_filing_type(confidence_score, sanctions_flag, matched_patterns),
            "supporting_evidence": {
                "gnn_mule_probability": round(mule_prob, 4),
                "recommended_action": acc.get("recommended_action"),
                "matched_financial_crime_patterns": matched_patterns,
                "sanctions_watchlist_hit": sanctions_flag,
            },
            "xai_auditor": xai_subject,
        })

    # ── Section 4: Pattern Analysis ──────────────────────────────
    pattern_section = None
    if patterns:
        by_type = {}
        for p in patterns.get("patterns", []):
            pt = p.get("pattern_type", "UNKNOWN")
            if pt not in by_type:
                by_type[pt] = []
            by_type[pt].append({
                "account_id": p.get("account_id"),
                "confidence_score": p.get("confidence_score"),
                "evidence": p.get("evidence"),
                "recommended_action": p.get("recommended_action"),
                "regulatory_ref": p.get("regulatory_ref"),
            })

        pattern_section = {
            "structuring_cases": by_type.get("STRUCTURING", []),
            "fragmentation_cases": by_type.get("FRAGMENTATION", []),
            "nesting_cases": by_type.get("NESTING", []),
            "circular_flow_cases": by_type.get("CIRCULAR_FLOW", []),
            "total_cases": patterns.get("total_patterns_detected", 0),
        }

    # ── Section 5: Cluster / Mule Ring Analysis ───────────────────
    cluster_section = []
    for cluster in risk_summary.get("clusters", [])[:10]:
        cluster_section.append({
            "cluster_id": cluster.get("cluster_id"),
            "size": cluster.get("size"),
            "avg_risk_score": round(cluster.get("avg_risk_score", 0) * 100, 1),
            "confidence_label": _confidence_label(cluster.get("avg_risk_score", 0) * 100),
            "channels_used": cluster.get("channels_used", []),
            "total_volume": cluster.get("total_volume", 0),
            "avg_velocity_seconds": cluster.get("avg_velocity_seconds", 0),
            "min_velocity_seconds": cluster.get("min_velocity_seconds", 0),
            "hub_account": cluster.get("hub_account"),
            "recommended_action": "Freeze All Members & File STR" if cluster.get("avg_risk_score", 0) >= 0.85 else "Enhanced Monitoring",
        })

    # ── Section 6: Sanctions Screening Summary ────────────────────
    sanctions_section = None
    if sanctions:
        sanctions_section = {
            "total_screened": sanctions.get("total_screened", 0),
            "watchlist_hits": sanctions.get("watchlist_hits", 0),
            "behavioural_matches": sanctions.get("behavioural_matches", 0),
            "screening_methods": sanctions.get("screening_method", []),
            "top_alerts": sanctions.get("top_alerts", [])[:5],
        }

    # ── Section 7: Recommended Regulatory Actions ─────────────────
    risk_dist = risk_summary.get("risk_distribution", {})
    actions_section = {
        "freeze_accounts": risk_dist.get("freeze", 0),
        "escalate_to_fiu_ind": risk_dist.get("escalate", 0),
        "enhanced_monitoring": risk_dist.get("monitor", 0),
        "clear": risk_dist.get("clear", 0),
        "str_filings_recommended": max(
            risk_dist.get("escalate", 0),
            len([p for p in (patterns or {}).get("patterns", []) if p.get("confidence_score", 0) >= 70])
        ),
        "ctr_filings_recommended": len([
            p for p in (patterns or {}).get("patterns", [])
            if p.get("pattern_type") == "STRUCTURING"
        ]),
    }

    # ── Assemble Full Report ──────────────────────────────────────
    return {
        "report_header": header,
        "executive_summary": executive_summary,
        "suspicious_subjects": subjects,
        "mule_ring_clusters": cluster_section,
        "financial_crime_patterns": pattern_section,
        "sanctions_screening": sanctions_section,
        "recommended_actions": actions_section,
        "report_metadata": {
            "generated_by": "ChainVigil v1.0 — GNN-Based Fraud Intelligence",
            "model_architecture": "GraphSAGE + Graph Attention Network (GAT)",
            "xai_method": "Gradient × Input Feature Attribution",
            "privacy_method": "SHA-256 Account Anonymization",
            "graph_stats": graph_stats or {},
        },
    }


def _get_recommended_actions(flagged: int, clusters: int, patterns: int, sanctions: int) -> List[str]:
    actions = []
    if flagged > 0:
        actions.append(f"File STR for {flagged} high-risk accounts with FIU-IND")
    if clusters > 0:
        actions.append(f"Initiate ring investigation for {clusters} mule ring cluster(s)")
    if patterns > 0:
        actions.append(f"Review {patterns} financial crime pattern alerts (structuring/fragmentation/nesting)")
    if sanctions > 0:
        actions.append(f"Escalate {sanctions} sanctions match(es) to Compliance & Legal immediately")
    if not actions:
        actions.append("Continue routine monitoring")
    return actions


def _get_filing_type(confidence: float, sanctions: bool, patterns: List[str]) -> str:
    if sanctions:
        return "IMMEDIATE STR + SANCTIONS REPORT"
    if confidence >= 90 or "STRUCTURING" in patterns:
        return "STR Filing (Suspicious Transaction Report)"
    if "FRAGMENTATION" in patterns or confidence >= 70:
        return "CTR Filing (Cash Transaction Report)"
    if confidence >= 50:
        return "Enhanced Due Diligence (EDD)"
    return "Routine Monitoring"
