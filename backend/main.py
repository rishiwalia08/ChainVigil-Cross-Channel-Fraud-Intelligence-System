"""
ChainVigil — FastAPI Backend Server

Main entry point with API routes for:
  - Data generation
  - Graph ingestion
  - GNN training & inference
  - Risk analysis
  - XAI explanations
  - Audit reports
"""

import os
import json
import asyncio
from pathlib import Path
from typing import Optional, List

# ── Observability (Step 1) ────────────────────────────────────────────────────
from backend.observability.metrics import METRICS, timer, log_prediction

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, FileResponse
from fastapi.staticfiles import StaticFiles

from backend.config import DATA_DIR, MODEL_DIR
from backend.models.schemas import (
    GenerateDataRequest, IngestRequest, TrainRequest,
    GraphStats, PipelineStatus
)
from backend.data.generator import generate_all_data
from backend.graph.builder import GraphBuilder
from backend.graph.neo4j_client import Neo4jClient


from contextlib import asynccontextmanager

# ─── Lifecycle Events ─────────────────────────────────────────

@asynccontextmanager
async def lifespan_context(app: FastAPI):
    """Startup must be INSTANT so Render can detect the port immediately.
    Neo4j and pipeline are lazy-initialized on first API call.
    """
    print("✅ ChainVigil API started. Visit /docs or use the UI to run the pipeline.")
    yield
    """Clean up resources."""
    if state["neo4j_client"]:
        state["neo4j_client"].close()

# ─── App Setup ──────────────────────────────────────────────────

app = FastAPI(
    title="ChainVigil API",
    description="Cross-Channel Mule Detection using Graph Intelligence & GNN",
    version="1.0.0",
    lifespan=lifespan_context,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─── Global State ──────────────────────────────────────────────

state = {
    "neo4j_client": None,
    "graph_builder": None,
    "nx_graph": None,
    "pyg_data": None,
    "node_mapping": None,
    "account_ids": None,
    "trainer": None,
    "risk_scores": None,
    "risk_summary": None,
    "explanations": {},
    "init_status": "not_started",  # not_started | running | complete | failed
    "init_error": None,
}

# ─── Static Files (React Frontend) ─────────────────────────────

FRONTEND_DIST = Path(__file__).parent.parent / "frontend-app" / "dist"

if FRONTEND_DIST.exists():
    app.mount("/assets", StaticFiles(directory=str(FRONTEND_DIST / "assets")), name="assets")
    print(f"✅ React frontend mounted from {FRONTEND_DIST}")
else:
    print(f"⚠️  Frontend dist not found at {FRONTEND_DIST} — serving API only")



# ─── Health & Info ─────────────────────────────────────────────

@app.api_route("/health", methods=["GET", "HEAD"])
async def health():
    return {"status": "healthy"}

@app.get("/api/status")
async def api_status():
    """API status — same info that used to be at root."""
    return {
        "name": "ChainVigil API",
        "version": "1.0.0",
        "description": "Cross-Channel Mule Detection System",
        "neo4j_connected": state["neo4j_client"] is not None and state["neo4j_client"].is_connected,
        "graph_loaded": state["nx_graph"] is not None,
        "model_trained": state["trainer"] is not None,
        "init_status": state["init_status"],
        "init_error": state["init_error"],
    }


# ─── Step 1: Observability — /metrics endpoint ─────────────────────────────

@app.get("/metrics")
async def get_metrics(format: str = "json"):
    """
    Expose system metrics.
    ?format=prometheus  → Prometheus text exposition format (for Grafana scrape)
    ?format=json        → Pretty JSON (default, for dashboard widgets)

    Suggested Grafana dashboard panels:
      - chainvigil_gauge_fraud_rate          → Stat panel (fraud %)
      - chainvigil_gauge_flagged_accounts    → Stat panel
      - chainvigil_latency_inference_p95     → Time series (ms)
      - chainvigil_counter_total_predictions → Counter / rate
    """
    if format == "prometheus":
        from fastapi.responses import PlainTextResponse
        return PlainTextResponse(METRICS.prometheus_text(), media_type="text/plain")
    return METRICS.snapshot()


@app.api_route("/", methods=["GET", "HEAD"])
async def root():
    """Serve React frontend index.html, or fallback API info."""
    index_file = FRONTEND_DIST / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    return {
        "name": "ChainVigil API",
        "version": "1.0.0",
        "note": "Frontend not built. Visit /docs for API documentation.",
        "init_status": state["init_status"],
    }


# ─── Phase 1: Data Generation ─────────────────────────────────

@app.post("/api/generate", response_model=PipelineStatus)
async def generate_data(request: GenerateDataRequest = GenerateDataRequest()):
    """Generate synthetic multi-channel transaction data."""
    try:
        data = generate_all_data(
            num_accounts=request.num_accounts,
            num_transactions=request.num_transactions,
            num_mule_rings=request.num_mule_rings,
        )

        mule_count = data["accounts"]["is_mule"].sum()

        return PipelineStatus(
            stage="Data Generation",
            status="complete",
            message=f"Generated {len(data['accounts'])} accounts, "
                    f"{len(data['transactions'])} transactions, "
                    f"{mule_count} mule accounts",
            details={
                "accounts": len(data["accounts"]),
                "transactions": len(data["transactions"]),
                "devices": len(data["devices"]),
                "ips": len(data["ips"]),
                "atm_withdrawals": len(data["atm_withdrawals"]),
                "mule_rings": len(data["rings"]),
                "mule_accounts": int(mule_count),
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Phase 1: Graph Construction ──────────────────────────────

@app.post("/api/ingest", response_model=PipelineStatus)
async def ingest_to_graph(request: IngestRequest = IngestRequest()):
    """Ingest generated data into the Unified Entity Graph."""
    try:
        builder = GraphBuilder(state["neo4j_client"])
        data_dir = request.data_path or DATA_DIR

        G = builder.build(data_dir=data_dir)
        state["graph_builder"] = builder
        state["nx_graph"] = G

        stats = builder.get_stats()
        return PipelineStatus(
            stage="Graph Construction",
            status="complete",
            message=f"Graph built: {stats['nx_nodes']} nodes, {stats['nx_edges']} edges",
            details=stats,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/graph/stats")
async def graph_stats():
    """Get current graph statistics."""
    if state["nx_graph"] is None:
        raise HTTPException(status_code=404, detail="Graph not built yet. Run /api/ingest first.")

    stats = state["graph_builder"].get_stats()

    # Add Neo4j stats if available
    if state["neo4j_client"] and state["neo4j_client"].is_connected:
        stats["neo4j"] = state["neo4j_client"].get_graph_stats()

    return stats


@app.get("/api/graph/visual")
async def graph_visual(max_nodes: int = 300):
    """Get graph data for force-directed visualization."""
    G = state["nx_graph"]
    if G is None:
        raise HTTPException(status_code=404, detail="Graph not built yet.")

    import random
    random.seed(42)

    risk_lookup = {}
    if state["risk_scores"]:
        risk_lookup = {r["account_id"]: r for r in state["risk_scores"]}

    # Collect cluster memberships
    cluster_lookup = {}
    if state["risk_summary"] and state["risk_summary"].get("clusters"):
        for cluster in state["risk_summary"]["clusters"]:
            for member in cluster.get("members", []):
                cluster_lookup[member] = cluster["cluster_id"]

    # Prioritize: all mule accounts + their neighbors + sample of normals
    account_nodes = [n for n, d in G.nodes(data=True) if d.get("entity_type") == "Account"]
    mule_nodes = [n for n in account_nodes if G.nodes[n].get("is_mule")]
    
    # Get neighbors of mules (devices, IPs connected to them)
    priority_nodes = set(mule_nodes)
    for m in mule_nodes:
        for neighbor in list(G.successors(m)) + list(G.predecessors(m)):
            priority_nodes.add(neighbor)

    # Fill remaining with random normal accounts
    remaining = [n for n in account_nodes if n not in priority_nodes]
    random.shuffle(remaining)
    all_nodes = list(priority_nodes.union(set(remaining[:max(0, max_nodes - len(priority_nodes))])))
    all_nodes = list(all_nodes)[:max_nodes]
    node_set = set(all_nodes)

    # Build nodes list
    nodes = []
    for n in all_nodes:
        data = G.nodes[n]
        entity_type = data.get("entity_type", "Unknown")
        risk = risk_lookup.get(n, {})

        node = {
            "id": n,
            "entity_type": entity_type,
            "is_mule": data.get("is_mule", False),
            "risk_score": risk.get("mule_probability", 0),
            "cluster_id": cluster_lookup.get(n),
            "action": risk.get("recommended_action"),
        }
        nodes.append(node)

    # Build links list (only between nodes in our set)
    links = []
    seen_edges = set()
    for u, v, data in G.edges(data=True):
        if u in node_set and v in node_set:
            edge_key = f"{u}->{v}-{data.get('edge_type', '')}"
            if edge_key not in seen_edges:
                seen_edges.add(edge_key)
                links.append({
                    "source": u,
                    "target": v,
                    "edge_type": data.get("edge_type", ""),
                    "amount": data.get("amount", 0),
                    "channel": data.get("channel_type", ""),
                    "is_suspicious": data.get("is_suspicious", False),
                })

    return {
        "nodes": nodes,
        "links": links,
        "total_nodes_in_graph": G.number_of_nodes(),
        "total_edges_in_graph": G.number_of_edges(),
        "showing_nodes": len(nodes),
        "showing_links": len(links),
    }


# ─── Phase 2: GNN Training ────────────────────────────────────

@app.post("/api/train", response_model=PipelineStatus)
async def train_model(request: TrainRequest = TrainRequest()):
    """Train the GNN mule detection model."""
    if state["nx_graph"] is None:
        raise HTTPException(
            status_code=400,
            detail="Graph not built. Run /api/generate then /api/ingest first."
        )

    try:
        from backend.gnn.dataset import nx_to_pyg
        from backend.gnn.train import Trainer

        # Convert to PyG
        pyg_data, node_mapping, account_ids = nx_to_pyg(state["nx_graph"])
        state["pyg_data"] = pyg_data
        state["node_mapping"] = node_mapping
        state["account_ids"] = account_ids

        # Train
        trainer = Trainer(pyg_data, lr=request.learning_rate)
        results = trainer.train(epochs=request.epochs)
        state["trainer"] = trainer

        return PipelineStatus(
            stage="GNN Training",
            status="complete",
            message=f"Training complete. Best Val AUC: {results['best_val_auc']:.4f}",
            details=results,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Phase 2: Risk Scoring ────────────────────────────────────

@app.post("/api/analyze", response_model=PipelineStatus)
async def run_risk_analysis():
    """Run full risk analysis: GNN scores → Risk Engine → Clusters."""
    if state["trainer"] is None:
        raise HTTPException(
            status_code=400,
            detail="Model not trained. Run /api/train first."
        )

    try:
        from backend.gnn.predict import predict_scores, load_model
        from backend.risk.engine import RiskIntelligenceEngine

        # Get predictions
        with timer("inference_latency_ms"):
            risk_scores = predict_scores(
                state["trainer"].model,
                state["pyg_data"],
                state["account_ids"],
            )
        state["risk_scores"] = risk_scores

        # Run risk engine
        engine = RiskIntelligenceEngine(
            state["nx_graph"], risk_scores
        )
        summary = engine.analyze()
        state["risk_summary"] = summary

        # ── Step 1: Update observability gauges ───────────────────────────
        total = summary["total_accounts_analyzed"]
        flagged = summary["flagged_accounts"]
        METRICS.set_gauge("accounts_analyzed", total)
        METRICS.set_gauge("flagged_accounts", flagged)
        METRICS.set_gauge("clusters_detected", summary["clusters_detected"])
        METRICS.set_gauge("fraud_rate", round(flagged / total, 4) if total else 0)
        METRICS.set_gauge("model_auc", state["trainer"].best_val_auc
                          if hasattr(state["trainer"], "best_val_auc") else 0)
        METRICS.inc("analyze_runs")

        # Log top predictions
        for r in risk_scores[:20]:
            log_prediction(
                account_id=r["account_id"],
                risk_score=r.get("mule_probability", 0),
                action=r.get("recommended_action", "Unknown"),
            )

        return PipelineStatus(
            stage="Risk Analysis",
            status="complete",
            message=f"Analyzed {summary['total_accounts_analyzed']} accounts. "
                    f"Flagged: {summary['flagged_accounts']}, "
                    f"Clusters: {summary['clusters_detected']}",
            details={
                "flagged": summary["flagged_accounts"],
                "clusters": summary["clusters_detected"],
                "risk_distribution": summary["risk_distribution"],
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Phase 4: XAI & Reports ───────────────────────────────────

@app.get("/api/explain/{account_id}")
async def explain_account(account_id: str):
    """Get XAI explanation for a specific account."""
    if state["trainer"] is None:
        raise HTTPException(status_code=400, detail="Model not trained.")

    try:
        from backend.xai.explainer import MuleExplainer

        explainer = MuleExplainer(
            state["trainer"].model,
            state["pyg_data"],
            state["account_ids"],
            state["node_mapping"],
        )
        explanation = explainer.explain_account(account_id)
        state["explanations"][account_id] = explanation
        return explanation
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/report")
async def get_full_report():
    """Generate full audit report."""
    if state["risk_summary"] is None:
        raise HTTPException(status_code=400, detail="Run /api/analyze first.")

    try:
        from backend.xai.report import AuditReportGenerator

        generator = AuditReportGenerator()
        report = generator.generate_full_report(
            state["risk_summary"],
            state["explanations"],
        )
        return report
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Data Endpoints ───────────────────────────────────────────

@app.get("/api/accounts")
async def list_accounts(limit: int = 50, offset: int = 0):
    """List accounts with risk scores."""
    if state["risk_scores"] is None:
        # Return from CSV if available
        import pandas as pd
        accounts_path = os.path.join(DATA_DIR, "accounts.csv")
        if os.path.exists(accounts_path):
            df = pd.read_csv(accounts_path)
            records = df.iloc[offset:offset + limit].to_dict("records")
            return {"accounts": records, "total": len(df)}
        raise HTTPException(status_code=404, detail="No data available.")

    scores = state["risk_scores"][offset:offset + limit]
    return {"accounts": scores, "total": len(state["risk_scores"])}


@app.get("/api/accounts/{account_id}")
async def get_account(account_id: str):
    """Get account details with neighbors and risk score."""
    G = state["nx_graph"]
    if G is None:
        raise HTTPException(status_code=404, detail="Graph not built.")

    if account_id not in G:
        raise HTTPException(status_code=404, detail=f"Account {account_id} not found.")

    node_data = dict(G.nodes[account_id])

    # Get neighbors
    successors = list(G.successors(account_id))[:20]
    predecessors = list(G.predecessors(account_id))[:20]

    # Get risk score
    risk = None
    if state["risk_scores"]:
        for r in state["risk_scores"]:
            if r["account_id"] == account_id:
                risk = r
                break

    return {
        "account_id": account_id,
        **node_data,
        "risk_score": risk,
        "successors": successors,
        "predecessors": predecessors,
    }


@app.get("/api/clusters")
async def list_clusters():
    """List detected mule ring clusters."""
    if state["risk_summary"] is None:
        raise HTTPException(status_code=404, detail="Run /api/analyze first.")
    return {"clusters": state["risk_summary"].get("clusters", [])}


@app.get("/api/clusters/{cluster_id}")
async def get_cluster(cluster_id: str):
    """Get details of a specific cluster."""
    if state["risk_summary"] is None:
        raise HTTPException(status_code=404, detail="Run /api/analyze first.")

    for cluster in state["risk_summary"].get("clusters", []):
        if cluster["cluster_id"] == cluster_id:
            return cluster

    raise HTTPException(status_code=404, detail=f"Cluster {cluster_id} not found.")


# ─── Privacy-Preserving Export ────────────────────────────────

@app.get("/api/export/anonymized")
async def export_anonymized():
    """
    Export anonymized graph data for privacy-preserving inter-bank sharing.
    
    All PII is stripped. Account IDs are replaced with SHA-256 hashes.
    Graph topology, risk scores, and cluster memberships are preserved
    so receiving institutions can cross-reference patterns without
    accessing customer identities.
    """
    import hashlib
    from datetime import datetime

    G = state["nx_graph"]
    if G is None:
        raise HTTPException(status_code=404, detail="Graph not built. Run pipeline first.")

    SALT = "chainvigil_interbank_2025"

    def anonymize_id(original_id: str) -> str:
        return hashlib.sha256(f"{SALT}:{original_id}".encode()).hexdigest()[:16]

    risk_lookup = {}
    if state["risk_scores"]:
        risk_lookup = {r["account_id"]: r for r in state["risk_scores"]}

    cluster_lookup = {}
    if state["risk_summary"] and state["risk_summary"].get("clusters"):
        for cluster in state["risk_summary"]["clusters"]:
            for member in cluster.get("members", []):
                cluster_lookup[member] = cluster["cluster_id"]

    # Build anonymized nodes (accounts only — no device/IP PII)
    anon_nodes = []
    id_map = {}
    for n, data in G.nodes(data=True):
        if data.get("entity_type") != "Account":
            continue
        anon_id = anonymize_id(n)
        id_map[n] = anon_id
        risk = risk_lookup.get(n, {})

        anon_nodes.append({
            "anon_id": anon_id,
            "risk_score": round(risk.get("mule_probability", 0), 4),
            "recommended_action": risk.get("recommended_action", "Unknown"),
            "is_flagged": risk.get("is_flagged", False),
            "cluster_id": cluster_lookup.get(n),
            # Behavioral features only — no PII
            "features": {
                "in_degree": G.in_degree(n),
                "out_degree": G.out_degree(n),
                "channel_diversity": data.get("channel_diversity", 0),
                "shared_device_count": data.get("shared_device_count", 0),
                "shared_ip_count": data.get("shared_ip_count", 0),
                "jurisdiction_risk": data.get("jurisdiction_risk", 0),
            }
        })

    # Build anonymized edges (transfers only)
    anon_edges = []
    for u, v, data in G.edges(data=True):
        if u in id_map and v in id_map and data.get("edge_type") == "TRANSFERRED_TO":
            anon_edges.append({
                "source": id_map[u],
                "target": id_map[v],
                "channel": data.get("channel_type", ""),
                "amount_bucket": _bucket_amount(data.get("amount", 0)),
                "is_suspicious": data.get("is_suspicious", False),
            })

    # Anonymized cluster summaries
    anon_clusters = []
    if state["risk_summary"] and state["risk_summary"].get("clusters"):
        for cluster in state["risk_summary"]["clusters"]:
            anon_clusters.append({
                "cluster_id": cluster["cluster_id"],
                "size": cluster["size"],
                "density": cluster.get("density", 0),
                "avg_risk_score": cluster.get("avg_risk_score", 0),
                "channels_used": cluster.get("channels_used", []),
                "members": [id_map.get(m, "unknown") for m in cluster.get("members", [])],
            })

    return {
        "export_type": "anonymized_interbank_sharing",
        "generated_at": datetime.utcnow().isoformat() + "Z",
        "hash_algorithm": "SHA-256 (salted)",
        "pii_removed": [
            "account_holder_name", "bank_name", "country",
            "device_id", "ip_address", "atm_location"
        ],
        "data_retained": [
            "graph_topology", "risk_scores", "behavioral_features",
            "cluster_memberships", "channel_types", "amount_buckets"
        ],
        "summary": {
            "total_accounts": len(anon_nodes),
            "total_transfers": len(anon_edges),
            "flagged_accounts": sum(1 for n in anon_nodes if n["is_flagged"]),
            "clusters_detected": len(anon_clusters),
        },
        "nodes": anon_nodes,
        "edges": anon_edges,
        "clusters": anon_clusters,
    }


def _bucket_amount(amount: float) -> str:
    """Convert exact amounts to buckets to prevent re-identification."""
    if amount < 5000: return "<5K"
    if amount < 10000: return "5K-10K"
    if amount < 25000: return "10K-25K"
    if amount < 50000: return "25K-50K"
    if amount < 100000: return "50K-1L"
    return ">1L"


# ─── Pipeline Shortcut ────────────────────────────────────────

@app.post("/api/pipeline/run", response_model=PipelineStatus)
async def run_full_pipeline():
    """Run the entire pipeline: Generate → Ingest → Train → Analyze."""
    try:
        # Step 1: Generate
        data = generate_all_data()

        # Step 2: Build graph
        builder = GraphBuilder(state["neo4j_client"])
        G = builder.build(data)
        state["graph_builder"] = builder
        state["nx_graph"] = G

        # Step 3: Convert & Train
        from backend.gnn.dataset import nx_to_pyg
        from backend.gnn.train import Trainer

        pyg_data, node_mapping, account_ids = nx_to_pyg(G)
        state["pyg_data"] = pyg_data
        state["node_mapping"] = node_mapping
        state["account_ids"] = account_ids

        trainer = Trainer(pyg_data)
        results = trainer.train()
        state["trainer"] = trainer

        # Step 4: Risk Analysis
        from backend.gnn.predict import predict_scores
        from backend.risk.engine import RiskIntelligenceEngine

        risk_scores = predict_scores(trainer.model, pyg_data, account_ids)
        state["risk_scores"] = risk_scores

        engine = RiskIntelligenceEngine(G, risk_scores)
        summary = engine.analyze()
        state["risk_summary"] = summary

        return PipelineStatus(
            stage="Full Pipeline",
            status="complete",
            message=f"Pipeline complete. AUC: {results['best_val_auc']:.4f}, "
                    f"Flagged: {summary['flagged_accounts']}, "
                    f"Clusters: {summary['clusters_detected']}",
            details={
                "training": results,
                "risk": {
                    "flagged": summary["flagged_accounts"],
                    "clusters": summary["clusters_detected"],
                    "distribution": summary["risk_distribution"],
                },
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ─── Financial Crime Pattern Detection ─────────────────────────

@app.get("/api/patterns")
async def get_patterns():
    """Detect structuring, fragmentation, nesting, and circular flow patterns."""
    from backend.risk.patterns import PatternDetector
    import json, os

    if state["nx_graph"] is None:
        raise HTTPException(status_code=400, detail="Run pipeline first (/api/pipeline/run)")

    # Load transactions for pattern analysis
    tx_file = os.path.join(
        os.path.dirname(__file__), "data", "sample_data", "transactions.json"
    )
    transactions = []
    if os.path.exists(tx_file):
        with open(tx_file) as f:
            transactions = json.load(f)

    detector = PatternDetector(state["nx_graph"], transactions)
    result = detector.run_all(transactions)
    state["patterns"] = result
    return result


# ─── Sanctions Screening ────────────────────────────────────────

@app.get("/api/sanctions/summary")
async def get_sanctions_summary():
    """Run behaviour-based sanctions screening on all flagged accounts."""
    from backend.risk.sanctions import SanctionsScreener

    if state["risk_scores"] is None:
        raise HTTPException(status_code=400, detail="Run pipeline first (/api/pipeline/run)")

    screener = SanctionsScreener()
    result = screener.screen_all(state["risk_scores"])
    state["sanctions"] = result
    return result


@app.get("/api/sanctions/check/{account_id}")
async def check_sanctions(account_id: str):
    """Screen a single account against watchlist and behavioural fingerprints."""
    from backend.risk.sanctions import SanctionsScreener

    risk_data = {}
    if state["risk_scores"]:
        for r in state["risk_scores"]:
            if r.get("account_id") == account_id:
                risk_data = r
                break

    screener = SanctionsScreener()
    return screener.screen_account(account_id, risk_data)


# ─── FIU-IND SAR Report ─────────────────────────────────────────

@app.get("/api/report/sar")
async def get_sar_report():
    """Generate FIU-IND compliant Suspicious Activity Report (SAR)."""
    from backend.xai.report import generate_sar_report

    if state["risk_summary"] is None:
        raise HTTPException(status_code=400, detail="Run pipeline first (/api/pipeline/run)")

    # Run patterns + sanctions if not already done
    if state.get("patterns") is None:
        try:
            await get_patterns()
        except Exception:
            pass

    if state.get("sanctions") is None:
        try:
            await get_sanctions_summary()
        except Exception:
            pass

    graph_stats = None
    if state["nx_graph"] is not None:
        G = state["nx_graph"]
        graph_stats = {
            "nodes": G.number_of_nodes(),
            "edges": G.number_of_edges(),
        }

    # Build XAI details for top suspicious subjects (for SAR rendering)
    xai_by_account = {}
    top_accounts = [
        acc.get("account_id")
        for acc in state["risk_summary"].get("high_risk_accounts", [])[:20]
        if acc.get("account_id")
    ]

    if top_accounts and state.get("trainer") is not None and state.get("pyg_data") is not None:
        from backend.xai.explainer import MuleExplainer

        explainer = MuleExplainer(
            state["trainer"].model,
            state["pyg_data"],
            state["account_ids"],
            state["node_mapping"],
        )

        for account_id in top_accounts:
            if account_id in state["explanations"]:
                xai_by_account[account_id] = state["explanations"][account_id]
                continue

            try:
                explanation = explainer.explain_account(account_id)
                state["explanations"][account_id] = explanation
                xai_by_account[account_id] = explanation
            except Exception:
                # Keep SAR generation resilient even if one explanation fails
                continue

    sar = generate_sar_report(
        risk_summary=state["risk_summary"],
        patterns=state.get("patterns"),
        sanctions=state.get("sanctions"),
        graph_stats=graph_stats,
        xai_by_account=xai_by_account,
    )
    return sar


# ─── Intelligence Layer: Unified AI Analysis  (Steps 2–6) ─────────────────

@app.get("/api/intelligence/analyze/{account_id}")
async def intelligence_analyze(account_id: str, text: Optional[str] = None):
    """
    Full intelligence pipeline for a single account.

    Combines:
      • Temporal anomaly score    (Step 2)
      • Behavioral risk profile   (Step 3)
      • Root-cause explanation    (Step 4)
      • Automated action decision (Step 5)
      • NLP text analysis         (Step 6, pass ?text=<note>)

    Returns a single unified decision object.
    """
    G = state["nx_graph"]
    if G is None:
        raise HTTPException(status_code=400, detail="Run pipeline first (/api/pipeline/run)")
    if account_id not in G:
        raise HTTPException(status_code=404, detail=f"Account {account_id} not found.")

    try:
        from backend.intelligence.temporal   import TemporalAnomalyDetector
        from backend.intelligence.behavioral  import BehavioralProfiler
        from backend.intelligence.root_cause  import RootCauseEngine
        from backend.intelligence.action_engine import AutomatedActionEngine
        from backend.intelligence.nlp_detector  import NLPFraudDetector

        # Step 2: Temporal
        temporal_result = TemporalAnomalyDetector(G).score_account(account_id)

        # Step 3: Behavioral
        behavioral_result = BehavioralProfiler(G).profile_account(account_id)

        # Step 6: NLP (optional — pass transaction note via ?text=)
        nlp_result = NLPFraudDetector().analyze(text) if text else None

        # Gather GNN score and existing XAI features
        gnn_score = 0.0
        xai_features = []
        rule_reasons = []
        cluster_id = None
        high_risk_neighbors = 0

        if state["risk_scores"]:
            for r in state["risk_scores"]:
                if r.get("account_id") == account_id:
                    gnn_score = r.get("mule_probability", 0.0)
                    break

        if state["explanations"] and account_id in state["explanations"]:
            xai = state["explanations"][account_id]
            xai_features = xai.get("feature_attributions", [])

        if state["risk_summary"]:
            for cluster in state["risk_summary"].get("clusters", []):
                if account_id in cluster.get("members", []):
                    cluster_id = cluster["cluster_id"]
                    high_risk_neighbors = cluster.get("size", 0) - 1
                    break

        # Step 4: Root-cause explanation
        root_cause = RootCauseEngine().explain(
            account_id=account_id,
            gnn_score=gnn_score,
            rule_reasons=rule_reasons,
            xai_features=xai_features,
            temporal_result=temporal_result,
            behavioral_result=behavioral_result,
            nlp_result=nlp_result,
            cluster_id=cluster_id,
            high_risk_neighbors=high_risk_neighbors,
        )

        # Step 5: Automated action decision
        decision = AutomatedActionEngine().decide(
            account_id=account_id,
            final_score=root_cause["final_risk_score"],
            explanation=root_cause["explanation"],
            evidence=root_cause["evidence"],
            risk_tier=root_cause["risk_tier"],
        )

        # Observability
        METRICS.inc("intelligence_queries")
        log_prediction(
            account_id=account_id,
            risk_score=root_cause["final_risk_score"],
            action=decision["action"],
            source="intelligence",
        )

        return {
            "account_id":        account_id,
            "gnn_score":         round(gnn_score, 4),
            "temporal":          temporal_result,
            "behavioral":        behavioral_result,
            "nlp":               nlp_result,
            "root_cause":        root_cause,
            "decision":          decision,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/intelligence/nlp")
async def nlp_analyze(payload: dict):
    """
    Analyze a transaction note or description for NLP fraud signals.  (Step 6)

    Request body:
        {"text": "<transaction description>",
         "tx_id": "<optional>"}           # optional, for logging

    Returns NLP fraud detection result.
    """
    from backend.intelligence.nlp_detector import NLPFraudDetector

    text  = payload.get("text", "")
    tx_id = payload.get("tx_id", "unknown")

    result = NLPFraudDetector().analyze(text)
    METRICS.inc("nlp_analyses")
    if result["is_suspicious"]:
        METRICS.inc("nlp_suspicious_flagged")

    return {
        "tx_id":  tx_id,
        "input":  text[:200],          # truncate for response safety
        **result,
    }


@app.get("/api/intelligence/temporal/{account_id}")
async def temporal_analyze(account_id: str):
    """Get temporal anomaly score for a single account.  (Step 2)"""
    G = state["nx_graph"]
    if G is None:
        raise HTTPException(status_code=400, detail="Run graph build first.")
    from backend.intelligence.temporal import TemporalAnomalyDetector
    result = TemporalAnomalyDetector(G).score_account(account_id)
    METRICS.inc("temporal_queries")
    return {"account_id": account_id, **result}


@app.get("/api/intelligence/behavioral/{account_id}")
async def behavioral_analyze(account_id: str):
    """Get behavioral risk profile for a single account.  (Step 3)"""
    G = state["nx_graph"]
    if G is None:
        raise HTTPException(status_code=400, detail="Run graph build first.")
    from backend.intelligence.behavioral import BehavioralProfiler
    result = BehavioralProfiler(G).profile_account(account_id)
    METRICS.inc("behavioral_queries")
    return {"account_id": account_id, **result}


# ─── SSE: Real-Time Live Transaction Feed ───────────────────────

from fastapi.responses import StreamingResponse

@app.get("/api/stream/live")
async def live_transaction_stream():
    """
    Server-Sent Events (SSE) real-time transaction feed.
    Streams one scored transaction every 2 seconds.
    Frontend subscribes with EventSource('/api/stream/live').
    """
    from backend.realtime.processor import live_transaction_generator
    return StreamingResponse(
        live_transaction_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


@app.get("/api/stream/recent")
async def get_recent_live_events():
    """Return the last 50 live events from the ring buffer (for initial load)."""
    from backend.realtime.processor import get_recent_events
    return {"events": get_recent_events(50)}


# ─── Catch-all: React SPA routing ────────────────────────────
# Must be LAST route so it doesn't override API routes above

from fastapi import Request

@app.get("/{full_path:path}")
async def spa_fallback(request: Request, full_path: str):
    """Serve React app for all non-API routes (enables client-side routing)."""
    index_file = FRONTEND_DIST / "index.html"
    if index_file.exists():
        return FileResponse(str(index_file))
    raise HTTPException(status_code=404, detail=f"Path /{full_path} not found")

