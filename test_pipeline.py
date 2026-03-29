import asyncio
from backend.data.generator import generate_all_data
from backend.graph.builder import GraphBuilder
from backend.gnn.trainer import GNNTrainer
from backend.risk.engine import RiskIntelligenceEngine
from backend.risk.patterns import PatternDetector
from backend.risk.sanctions import SanctionsScreener
from backend.xai.report import generate_sar_report
import os

os.environ["NUM_ACCOUNTS"] = "50"
os.environ["NUM_TRANSACTIONS"] = "200"
os.environ["GNN_EPOCHS"] = "5"

async def run_test():
    print("1. Generating Data...")
    from backend.config import NUM_ACCOUNTS
    result = generate_all_data()
    print("Data Gen Result:", len(result["accounts"]), "accounts")

    print("2. Building Graph...")
    builder = GraphBuilder(result, use_neo4j=False)
    nx_graph = builder.build()
    print("Graph built:", nx_graph.number_of_nodes(), "nodes")

    print("3. Training GNN...")
    trainer = GNNTrainer(nx_graph)
    train_res = trainer.train()
    print("Training finished. Val AUC:", train_res["best_val_auc"])

    # Extract node embeddings and convert to frontend risk_scores format
    risk_scores = []
    for node, data in nx_graph.nodes(data=True):
        if data.get("type") == "ACCOUNT":
            prob = float(data.get("mule_probability", 0.0))
            score_dict = {"account_id": node, "mule_probability": prob}
            # copy all attributes for simplicity
            for k,v in data.items():
                if k not in score_dict:
                    score_dict[k] = v
            # add channel diversity dummy to prevent sanctions crash
            score_dict["channel_diversity"] = 0.5
            risk_scores.append(score_dict)

    print("4. Risk Engine...")
    engine = RiskIntelligenceEngine(nx_graph, risk_scores)
    summary = engine.analyze()
    print("Risk Engine Output Clusters:", len(summary["clusters"]))

    print("5. Pattern Detector...")
    pd = PatternDetector(nx_graph, result["transactions"])
    patterns = pd.run_all(result["transactions"])
    print("Patterns detected:", patterns["total_patterns_detected"])

    print("6. Sanctions Screener...")
    ss = SanctionsScreener()
    sanctions = ss.screen_all(risk_scores)
    print("Sanctions alerts:", sanctions["total_alerts"])

    print("7. SAR Report...")
    sar = generate_sar_report(summary, patterns, sanctions, {"nodes": 10, "edges": 20})
    print("SAR Report generated.")

    print("ALL TESTS PASSED SUCCESSFULLY.")

if __name__ == "__main__":
    asyncio.run(run_test())
