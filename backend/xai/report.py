"""
ChainVigil — Audit Report Generator

Generates structured, regulator-ready audit reports
for flagged accounts and mule ring clusters.
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional

from backend.config import DATA_DIR


class AuditReportGenerator:
    """Generates JSON audit reports for regulatory compliance."""

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = output_dir or os.path.join(DATA_DIR, "..", "reports")
        os.makedirs(self.output_dir, exist_ok=True)

    def generate_account_report(
        self,
        account_id: str,
        risk_score: Dict,
        explanation: Dict,
        cluster_info: Optional[Dict] = None,
    ) -> Dict:
        """Generate a single account audit report."""
        report = {
            "report_id": f"RPT-{datetime.now().strftime('%Y%m%d%H%M%S')}-{account_id}",
            "generated_at": datetime.now().isoformat(),
            "report_type": "ACCOUNT_RISK_ASSESSMENT",
            "account_id": account_id,
            "confidence_score": risk_score.get("mule_probability", 0),
            "recommended_action": risk_score.get("recommended_action", "Monitor"),
            "top_features": explanation.get("top_features", []),
            "feature_attributions": explanation.get("feature_attributions", []),
            "xai_reasoning": explanation.get("xai_reasoning", ""),
            "cluster_id": cluster_info.get("cluster_id") if cluster_info else None,
            "cluster_details": cluster_info,
            "regulator_status": "PENDING_REVIEW",
            "model_version": "ChainVigil-GNN-v1.0",
        }
        return report

    def generate_cluster_report(
        self,
        cluster: Dict,
        member_explanations: List[Dict],
    ) -> Dict:
        """Generate a mule ring cluster audit report."""
        report = {
            "report_id": f"RPT-CLUSTER-{datetime.now().strftime('%Y%m%d%H%M%S')}-{cluster['cluster_id']}",
            "generated_at": datetime.now().isoformat(),
            "report_type": "MULE_RING_ASSESSMENT",
            "cluster_id": cluster["cluster_id"],
            "cluster_size": cluster["size"],
            "members": cluster["members"],
            "hub_account": cluster.get("hub_account"),
            "avg_risk_score": cluster.get("avg_risk_score", 0),
            "density": cluster.get("density", 0),
            "total_volume": cluster.get("total_volume", 0),
            "avg_velocity_seconds": cluster.get("avg_velocity_seconds", 0),
            "channels_used": cluster.get("channels_used", []),
            "member_explanations": member_explanations,
            "regulator_status": "PENDING_REVIEW",
            "model_version": "ChainVigil-GNN-v1.0",
        }
        return report

    def generate_full_report(
        self,
        risk_summary: Dict,
        explanations: Dict[str, Dict],
    ) -> Dict:
        """Generate a comprehensive system-wide audit report."""
        report = {
            "report_id": f"RPT-FULL-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "generated_at": datetime.now().isoformat(),
            "report_type": "FULL_SYSTEM_AUDIT",
            "summary": {
                "total_accounts_analyzed": risk_summary.get("total_accounts_analyzed", 0),
                "flagged_accounts": risk_summary.get("flagged_accounts", 0),
                "clusters_detected": risk_summary.get("clusters_detected", 0),
                "risk_distribution": risk_summary.get("risk_distribution", {}),
            },
            "high_risk_accounts": [
                {
                    **acc,
                    "explanation": explanations.get(acc["account_id"], {}),
                }
                for acc in risk_summary.get("high_risk_accounts", [])
            ],
            "clusters": risk_summary.get("clusters", []),
            "model_version": "ChainVigil-GNN-v1.0",
            "regulator_status": "GENERATED",
        }

        # Save to disk
        filepath = os.path.join(self.output_dir, "full_audit_report.json")
        with open(filepath, "w") as f:
            json.dump(report, f, indent=2, default=str)
        print(f"📄 Full audit report saved to {filepath}")

        return report

    def save_report(self, report: Dict, filename: Optional[str] = None) -> str:
        """Save a report to disk."""
        if filename is None:
            filename = f"{report.get('report_id', 'report')}.json"
        filepath = os.path.join(self.output_dir, filename)
        with open(filepath, "w") as f:
            json.dump(report, f, indent=2, default=str)
        return filepath
