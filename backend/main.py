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
import time
import hashlib
import random
import asyncio
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Optional, Dict, Any, List

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from backend.config import DATA_DIR, MODEL_DIR
from backend.models.schemas import (
    GenerateDataRequest, IngestRequest, TrainRequest,
    GraphStats, PipelineStatus
)
from backend.data.generator import generate_all_data
from backend.graph.builder import GraphBuilder
from backend.graph.neo4j_client import Neo4jClient

# ─── App Setup ──────────────────────────────────────────────────

app = FastAPI(
    title="ChainVigil API",
    description="Cross-Channel Mule Detection using Graph Intelligence & GNN",
    version="1.0.0",
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
    "nft_cases": [],
    "processed_txn_ids": set(),
    "transaction_decisions": {},
    "init_in_progress": False,
    "init_last_status": "not_started",
    "init_last_error": None,
}


async def _auto_init_pipeline_once():
    """Initialize full pipeline once on startup (best effort)."""
    if state["init_in_progress"]:
        return

    state["init_in_progress"] = True
    state["init_last_status"] = "running"
    state["init_last_error"] = None

    try:
        await run_full_pipeline()
        state["init_last_status"] = "complete"
    except Exception as e:
        state["init_last_status"] = "failed"
        state["init_last_error"] = str(e)
    finally:
        state["init_in_progress"] = False


# ─── Real-Time Models & Helpers ─────────────────────────────

LEDGER_FILE = os.path.join(DATA_DIR, "..", "reports", "fraud_ledger.jsonl")
LEDGER_SALT = os.getenv("LEDGER_SALT", "chainvigil-ledger-salt")


class TransactionCheckRequest(BaseModel):
    transaction_id: str = Field(..., min_length=4)
    source_id: str
    target_id: str
    amount: float = Field(..., gt=0)
    channel_type: str = Field(default="UPI")
    timestamp: Optional[str] = None
    geo_location: Optional[str] = None
    device_id: Optional[str] = None
    ip_address: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TransactionDecisionResponse(BaseModel):
    transaction_id: str
    gnn_score: float
    rule_score: float
    intel_score: float
    hybrid_score: float
    decision: str
    reasons: List[str]
    ledger_tx_id: str
    processed_at: str


class StreamSimulationRequest(BaseModel):
    num_transactions: int = Field(default=25, ge=1, le=2000)
    interval_ms: int = Field(default=0, ge=0, le=5000)


def _parse_ts(ts: Optional[str]) -> datetime:
    if not ts:
        return datetime.now(timezone.utc)
    try:
        clean = ts.replace("Z", "+00:00")
        dt = datetime.fromisoformat(clean)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except Exception:
        return datetime.now(timezone.utc)


def _anonymize_account(account_id: str) -> str:
    return hashlib.sha256(f"{LEDGER_SALT}:{account_id}".encode()).hexdigest()[:16]


class FraudLedger:
    def __init__(self, file_path: str):
        self.path = Path(file_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("")

    def _read_all(self) -> List[Dict[str, Any]]:
        raw = self.path.read_text().strip()
        if not raw:
            return []
        blocks = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                blocks.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return blocks

    def _last_hash(self) -> str:
        blocks = self._read_all()
        return blocks[-1]["block_hash"] if blocks else "GENESIS"

    def append(self, payload: Dict[str, Any]) -> str:
        block = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "prev_hash": self._last_hash(),
            "payload": payload,
        }
        raw = json.dumps(block, sort_keys=True)
        block_hash = hashlib.sha256(raw.encode()).hexdigest()
        block["block_hash"] = block_hash
        with self.path.open("a") as f:
            f.write(json.dumps(block) + "\n")
        return block_hash

    def verify(self) -> Dict[str, Any]:
        blocks = self._read_all()
        if not blocks:
            return {"valid": True, "blocks": 0, "message": "ledger empty"}

        expected_prev = "GENESIS"
        for i, block in enumerate(blocks):
            stored_hash = block.get("block_hash", "")
            if block.get("prev_hash") != expected_prev:
                return {
                    "valid": False,
                    "blocks": len(blocks),
                    "failed_at_index": i,
                    "reason": "prev_hash mismatch",
                }

            to_hash = {
                "ts": block.get("ts"),
                "prev_hash": block.get("prev_hash"),
                "payload": block.get("payload"),
            }
            computed = hashlib.sha256(json.dumps(to_hash, sort_keys=True).encode()).hexdigest()
            if computed != stored_hash:
                return {
                    "valid": False,
                    "blocks": len(blocks),
                    "failed_at_index": i,
                    "reason": "block_hash mismatch",
                }

            expected_prev = stored_hash

        return {"valid": True, "blocks": len(blocks), "last_hash": expected_prev}

    def recent(self, limit: int = 20) -> List[Dict[str, Any]]:
        blocks = self._read_all()
        if limit <= 0:
            return []
        return list(reversed(blocks[-limit:]))


class RuleEngine:
    def evaluate(
        self,
        txn: TransactionCheckRequest,
        G,
        risk_lookup: Dict[str, Dict[str, Any]],
    ) -> Dict[str, Any]:
        score = 0.0
        reasons: List[str] = []
        hard_block = False

        if txn.source_id == txn.target_id:
            hard_block = True
            score = 1.0
            reasons.append("self_transfer_detected")

        if txn.amount >= 100000:
            score += 0.35
            reasons.append("very_high_amount")
        elif txn.amount >= 50000:
            score += 0.2
            reasons.append("high_amount")

        if txn.channel_type.upper() == "ATM" and txn.amount >= 20000:
            score += 0.1
            reasons.append("high_value_atm")

        src_risk = risk_lookup.get(txn.source_id, {}).get("mule_probability", 0.0)
        dst_risk = risk_lookup.get(txn.target_id, {}).get("mule_probability", 0.0)
        if max(src_risk, dst_risk) >= 0.85:
            score += 0.3
            reasons.append("known_high_risk_party")
        elif max(src_risk, dst_risk) >= 0.6:
            score += 0.15
            reasons.append("known_medium_risk_party")

        # Velocity rule: source sent multiple transfers in last 10 minutes
        now_ts = _parse_ts(txn.timestamp)
        recent_cutoff = now_ts - timedelta(minutes=10)
        recent_outgoing = 0
        if txn.source_id in G:
            for _, _, edge in G.out_edges(txn.source_id, data=True):
                if edge.get("edge_type") != "TRANSFERRED_TO":
                    continue
                edge_ts = _parse_ts(edge.get("timestamp"))
                if edge_ts >= recent_cutoff:
                    recent_outgoing += 1

        if recent_outgoing >= 5:
            score += 0.25
            reasons.append("rapid_outgoing_velocity")
        elif recent_outgoing >= 3:
            score += 0.12
            reasons.append("moderate_outgoing_velocity")

        return {
            "score": min(1.0, round(score, 4)),
            "hard_block": hard_block,
            "reasons": reasons,
        }


class DeviceIpIntel:
    def score(self, txn: TransactionCheckRequest, G) -> Dict[str, Any]:
        score = 0.0
        reasons: List[str] = []

        if txn.device_id and txn.device_id in G:
            users = {
                src for src, _, d in G.in_edges(txn.device_id, data=True)
                if d.get("edge_type") == "USED_DEVICE"
            }
            if len(users) >= 6:
                score += 0.25
                reasons.append("device_shared_many_accounts")
            elif len(users) >= 3:
                score += 0.12
                reasons.append("device_shared_multiple_accounts")

        if txn.ip_address and txn.ip_address in G:
            users = {
                src for src, _, d in G.in_edges(txn.ip_address, data=True)
                if d.get("edge_type") == "LOGGED_FROM"
            }
            if len(users) >= 8:
                score += 0.25
                reasons.append("ip_shared_many_accounts")
            elif len(users) >= 4:
                score += 0.12
                reasons.append("ip_shared_multiple_accounts")

        return {
            "score": min(1.0, round(score, 4)),
            "reasons": reasons,
        }


class HybridScorer:
    def __init__(self, w_gnn: float = 0.55, w_rule: float = 0.3, w_intel: float = 0.15):
        self.w_gnn = w_gnn
        self.w_rule = w_rule
        self.w_intel = w_intel

    def combine(self, gnn_score: float, rule_score: float, intel_score: float) -> float:
        score = (
            self.w_gnn * gnn_score
            + self.w_rule * rule_score
            + self.w_intel * intel_score
        )
        return min(1.0, round(score, 4))


class ActionEngine:
    def decide(self, hybrid_score: float, hard_block: bool = False) -> str:
        if hard_block or hybrid_score >= 0.9:
            return "FREEZE"
        if hybrid_score >= 0.75:
            return "BLOCK"
        if hybrid_score >= 0.5:
            return "MONITOR"
        return "ALLOW"


def _upsert_live_entities_and_edges(G, txn: TransactionCheckRequest):
    if txn.source_id not in G:
        G.add_node(txn.source_id, entity_type="Account", is_mule=False)
    if txn.target_id not in G:
        G.add_node(txn.target_id, entity_type="Account", is_mule=False)

    G.add_edge(
        txn.source_id,
        txn.target_id,
        edge_type="TRANSFERRED_TO",
        transaction_id=txn.transaction_id,
        amount=float(txn.amount),
        timestamp=(txn.timestamp or datetime.now(timezone.utc).isoformat()),
        channel_type=txn.channel_type,
        geo_location=txn.geo_location or "",
        is_suspicious=False,
        ingestion_mode="realtime",
    )

    if txn.device_id:
        if txn.device_id not in G:
            G.add_node(txn.device_id, entity_type="Device")
        G.add_edge(txn.source_id, txn.device_id, edge_type="USED_DEVICE")

    if txn.ip_address:
        if txn.ip_address not in G:
            G.add_node(txn.ip_address, entity_type="IPAddress")
        G.add_edge(txn.source_id, txn.ip_address, edge_type="LOGGED_FROM")


def _estimate_gnn_score(txn: TransactionCheckRequest) -> float:
    # Low-latency fallback: use latest account risk cache (if available).
    # In a full online setup, this should run incremental feature update + model inference.
    risk_lookup = {}
    if state.get("risk_scores"):
        risk_lookup = {r["account_id"]: r for r in state["risk_scores"]}

    src = risk_lookup.get(txn.source_id, {}).get("mule_probability")
    dst = risk_lookup.get(txn.target_id, {}).get("mule_probability")

    known_scores = [s for s in [src, dst] if isinstance(s, (int, float))]
    if known_scores:
        return round(float(max(known_scores)), 4)

    # Cold-start heuristic
    return 0.5 if txn.amount < 25000 else 0.62


def _process_transaction(txn: TransactionCheckRequest) -> Dict[str, Any]:
    G = state.get("nx_graph")
    if G is None:
        raise HTTPException(status_code=400, detail="Graph not built. Run /api/ingest or /api/pipeline/run first.")

    # Idempotency handling
    if txn.transaction_id in state["transaction_decisions"]:
        return state["transaction_decisions"][txn.transaction_id]

    _upsert_live_entities_and_edges(G, txn)

    risk_lookup = {}
    if state.get("risk_scores"):
        risk_lookup = {r["account_id"]: r for r in state["risk_scores"]}

    gnn_score = _estimate_gnn_score(txn)
    rule_engine = RuleEngine()
    intel_engine = DeviceIpIntel()
    hybrid_scorer = HybridScorer()
    action_engine = ActionEngine()
    ledger = FraudLedger(LEDGER_FILE)

    rule_out = rule_engine.evaluate(txn, G, risk_lookup)
    intel_out = intel_engine.score(txn, G)
    hybrid_score = hybrid_scorer.combine(
        gnn_score=gnn_score,
        rule_score=rule_out["score"],
        intel_score=intel_out["score"],
    )

    decision = action_engine.decide(hybrid_score, hard_block=rule_out["hard_block"])
    reasons = list(dict.fromkeys(rule_out["reasons"] + intel_out["reasons"]))

    ledger_payload = {
        "transaction_id": txn.transaction_id,
        "source_anon": _anonymize_account(txn.source_id),
        "target_anon": _anonymize_account(txn.target_id),
        "amount": txn.amount,
        "channel_type": txn.channel_type,
        "gnn_score": gnn_score,
        "rule_score": rule_out["score"],
        "intel_score": intel_out["score"],
        "hybrid_score": hybrid_score,
        "decision": decision,
        "reasons": reasons,
    }
    ledger_tx_id = ledger.append(ledger_payload)

    result = {
        "transaction_id": txn.transaction_id,
        "gnn_score": gnn_score,
        "rule_score": rule_out["score"],
        "intel_score": intel_out["score"],
        "hybrid_score": hybrid_score,
        "decision": decision,
        "reasons": reasons,
        "ledger_tx_id": ledger_tx_id,
        "processed_at": datetime.now(timezone.utc).isoformat(),
    }

    state["processed_txn_ids"].add(txn.transaction_id)
    state["transaction_decisions"][txn.transaction_id] = result
    return result


# ─── Lifecycle Events ─────────────────────────────────────────

@app.on_event("startup")
async def startup():
    """Initialize Neo4j connection on startup."""
    try:
        client = Neo4jClient()
        if client.is_connected:
            state["neo4j_client"] = client
    except Exception as e:
        print(f"⚠️  Neo4j not available: {e}")
        print("   Running in NetworkX-only mode")

    auto_init = os.getenv("AUTO_INIT_PIPELINE", "1").lower() in {"1", "true", "yes", "on"}
    if auto_init:
        asyncio.create_task(_auto_init_pipeline_once())


@app.on_event("shutdown")
async def shutdown():
    """Clean up resources."""
    if state["neo4j_client"]:
        state["neo4j_client"].close()


# ─── Health & Info ─────────────────────────────────────────────

@app.get("/")
async def root():
    return {
        "name": "ChainVigil API",
        "version": "1.0.0",
        "description": "Cross-Channel Mule Detection System",
        "status": "running",
        "note": "Service is up. Run POST /api/pipeline/run once to build graph and train model.",
        "quickstart": "/api/quickstart/init",
        "neo4j_connected": state["neo4j_client"] is not None and state["neo4j_client"].is_connected,
        "graph_loaded": state["nx_graph"] is not None,
        "model_trained": state["trainer"] is not None,
        "init_in_progress": state["init_in_progress"],
        "init_last_status": state["init_last_status"],
        "init_last_error": state["init_last_error"],
    }


@app.get("/health")
async def health():
    return {"status": "healthy"}


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
        from backend.realtime.nft import FraudCaseNFTRegistry

        # Get predictions
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

        # Minimal NFT-like certification for high-risk clusters
        nft_registry = FraudCaseNFTRegistry(
            os.path.join(DATA_DIR, "..", "reports", "nft_cases.json")
        )
        minted = []
        for cluster in summary.get("clusters", []):
            if cluster.get("avg_risk_score", 0) < 0.75:
                continue
            case_payload = {
                "case_id": f"CASE-{cluster['cluster_id']}",
                "cluster_id": cluster["cluster_id"],
                "members": cluster.get("members", []),
                "size": cluster.get("size", 0),
                "risk_score": cluster.get("avg_risk_score", 0),
                "explanation": (
                    f"Potential mule ring with {cluster.get('size', 0)} accounts, "
                    f"density {cluster.get('density', 0):.2f}, "
                    f"velocity {cluster.get('avg_velocity_seconds', 0):.1f}s."
                ),
            }
            minted.append(nft_registry.mint_case_certificate(case_payload))
        state["nft_cases"] = minted

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
                "nft_cases_minted": len(state["nft_cases"]),
            },
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/nft/cases")
async def list_nft_cases():
    """List minted NFT-like fraud case certificates."""
    return {
        "count": len(state.get("nft_cases", [])),
        "cases": state.get("nft_cases", []),
    }


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


@app.get("/api/quickstart/init", response_model=PipelineStatus)
async def quickstart_init():
    """Browser-friendly one-click pipeline init for first deployment run."""
    return await run_full_pipeline()


# ─── Real-Time Fraud APIs ───────────────────────────────────

@app.post("/api/transaction/check", response_model=TransactionDecisionResponse)
async def check_transaction(request: TransactionCheckRequest):
    """
    Real-time transaction decision endpoint.

    Flow:
      1) Update graph incrementally
      2) Compute hybrid score (GNN cache + rules + device/IP intel)
            3) Return action (ALLOW/MONITOR/BLOCK/FREEZE)
      4) Write tamper-evident audit block
    """
    try:
        result = _process_transaction(request)
        return TransactionDecisionResponse(**result)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/transaction/simulate-stream")
async def simulate_stream(request: StreamSimulationRequest = StreamSimulationRequest()):
    """
    Simulate streaming transactions and return decision summary.
    """
    G = state.get("nx_graph")
    if G is None:
        raise HTTPException(status_code=400, detail="Graph not built. Run /api/ingest or /api/pipeline/run first.")

    accounts = [n for n, d in G.nodes(data=True) if d.get("entity_type") == "Account"]
    if len(accounts) < 2:
        raise HTTPException(status_code=400, detail="Not enough account nodes for simulation.")

    decisions = {
        "ALLOW": 0,
        "MONITOR": 0,
        "BLOCK": 0,
        "FREEZE": 0,
    }
    samples = []

    for i in range(request.num_transactions):
        src = random.choice(accounts)
        dst = random.choice(accounts)
        while dst == src:
            dst = random.choice(accounts)

        txn = TransactionCheckRequest(
            transaction_id=f"SIM-{int(time.time() * 1000)}-{i}",
            source_id=src,
            target_id=dst,
            amount=round(random.uniform(500, 150000), 2),
            channel_type=random.choice(["UPI", "ATM", "WEB", "MOBILE_APP"]),
            timestamp=datetime.now(timezone.utc).isoformat(),
            geo_location=random.choice(["Mumbai", "Delhi", "Lagos", "Dubai", "London"]),
            device_id=f"DEV-SIM-{random.randint(100, 999)}",
            ip_address=f"10.20.{random.randint(0, 255)}.{random.randint(1, 254)}",
        )

        result = _process_transaction(txn)
        decisions[result["decision"]] = decisions.get(result["decision"], 0) + 1
        if len(samples) < 10:
            samples.append(result)

        if request.interval_ms > 0:
            await asyncio.sleep(request.interval_ms / 1000)

    return {
        "status": "complete",
        "processed": request.num_transactions,
        "decision_distribution": decisions,
        "sample_results": samples,
    }


@app.get("/api/ledger/verify")
async def verify_ledger():
    """Verify integrity of the tamper-evident audit ledger."""
    try:
        ledger = FraudLedger(LEDGER_FILE)
        return ledger.verify()
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/ledger/recent")
async def recent_ledger_entries(limit: int = 20):
    """Get recent ledger blocks for audit review."""
    try:
        ledger = FraudLedger(LEDGER_FILE)
        return {
            "entries": ledger.recent(limit=limit),
            "limit": limit,
            "file": LEDGER_FILE,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
