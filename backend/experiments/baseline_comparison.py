"""
ChainVigil — Baseline Comparison Experiment

Compares 3 models on the SAME 20 tabular features and train/test split:
  1. Logistic Regression (linear baseline)
  2. XGBoost (tree-based, no graph structure)
  3. ChainVigil GNN (graph edges + neighborhood aggregation)

Purpose: Determine whether the GNN's relational inductive bias adds
         value beyond what tabular features alone can achieve.

Usage:
    python -m backend.experiments.baseline_comparison
"""

import sys
import time
import warnings
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, classification_report,
)

import torch
from torch_geometric.data import Data

warnings.filterwarnings("ignore")


# ═══════════════════════════════════════════════════════════════
#  Step 1: Generate data + build graph + extract features
# ═══════════════════════════════════════════════════════════════

def prepare_data() -> Tuple[Data, np.ndarray, np.ndarray, List[str]]:
    """
    Run the full pipeline up to feature extraction.
    Returns PyG data, feature matrix X, labels y, and account_ids.
    """
    from backend.data.generator import generate_all_data
    from backend.graph.builder import GraphBuilder
    from backend.gnn.dataset import nx_to_pyg
    from backend.gnn.features import compute_node_features, get_feature_names

    print("═" * 65)
    print("  ChainVigil — Baseline Comparison Experiment")
    print("═" * 65)

    # Generate
    print("\n📦 Phase 1: Generating synthetic data...")
    data_dict = generate_all_data()

    # Build graph
    print("\n🔗 Phase 2: Building Unified Entity Graph...")
    builder = GraphBuilder()
    G = builder.build(data_dict)

    # Extract account nodes
    account_ids = sorted([
        n for n, d in G.nodes(data=True)
        if d.get("entity_type") == "Account"
    ])

    # Compute features
    print("\n📊 Phase 3: Computing 20 node features...")
    feature_df = compute_node_features(G, account_ids)
    feature_names = get_feature_names()

    for feat in feature_names:
        if feat not in feature_df.columns:
            feature_df[feat] = 0.0

    X_raw = feature_df.loc[account_ids][feature_names].values.astype(np.float32)

    # Labels
    y_raw = np.array([
        1 if G.nodes[acc].get("is_mule", False) else 0
        for acc in account_ids
    ])

    # Convert to PyG (uses the same stratified split)
    print("\n📐 Phase 4: Converting to PyTorch Geometric...")
    pyg_data, node_mapping, _ = nx_to_pyg(G, feature_df)

    return pyg_data, X_raw, y_raw, account_ids


# ═══════════════════════════════════════════════════════════════
#  Step 2: Extract masks into numpy for sklearn models
# ═══════════════════════════════════════════════════════════════

def get_split_arrays(
    pyg_data: Data, X_raw: np.ndarray, y_raw: np.ndarray
) -> Dict[str, Dict[str, np.ndarray]]:
    """
    Use the SAME stratified masks from PyG for all models.
    Normalizes features with z-score (train stats only — no test leakage).
    """
    train_idx = pyg_data.train_mask.cpu().numpy()
    val_idx = pyg_data.val_mask.cpu().numpy()
    test_idx = pyg_data.test_mask.cpu().numpy()

    # Fit scaler on train set only
    train_mean = X_raw[train_idx].mean(axis=0)
    train_std = X_raw[train_idx].std(axis=0)
    train_std[train_std == 0] = 1.0

    X_norm = (X_raw - train_mean) / train_std

    return {
        "train": {"X": X_norm[train_idx], "y": y_raw[train_idx]},
        "val":   {"X": X_norm[val_idx],   "y": y_raw[val_idx]},
        "test":  {"X": X_norm[test_idx],  "y": y_raw[test_idx]},
    }


# ═══════════════════════════════════════════════════════════════
#  Step 3: Train & evaluate baselines
# ═══════════════════════════════════════════════════════════════

def evaluate_model(
    name: str, y_true: np.ndarray, y_prob: np.ndarray
) -> Dict[str, float]:
    """Compute standard classification metrics."""
    y_pred = (y_prob > 0.5).astype(int)

    try:
        auc = roc_auc_score(y_true, y_prob)
    except ValueError:
        auc = 0.0

    return {
        "model": name,
        "AUC-ROC": round(auc, 4),
        "F1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "Precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "Recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
    }


def run_logistic_regression(splits: Dict) -> Dict[str, float]:
    """Train Logistic Regression on tabular features (no graph)."""
    print("\n" + "─" * 65)
    print("  🔬 Model 1: Logistic Regression (tabular only)")
    print("─" * 65)

    t0 = time.time()
    clf = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        solver="lbfgs",
        random_state=42,
    )
    clf.fit(splits["train"]["X"], splits["train"]["y"])
    elapsed = time.time() - t0

    # Predict probabilities on test set
    y_prob = clf.predict_proba(splits["test"]["X"])[:, 1]
    metrics = evaluate_model("Logistic Regression", splits["test"]["y"], y_prob)
    metrics["Train Time (s)"] = round(elapsed, 2)

    print(f"   Train time: {elapsed:.2f}s")
    print(f"   Test AUC: {metrics['AUC-ROC']:.4f} | F1: {metrics['F1']:.4f}")
    print(classification_report(
        splits["test"]["y"],
        (y_prob > 0.5).astype(int),
        target_names=["Normal", "Mule"],
        zero_division=0,
    ))

    return metrics


def run_xgboost(splits: Dict) -> Dict[str, float]:
    """Train XGBoost on tabular features (no graph)."""
    print("\n" + "─" * 65)
    print("  🌲 Model 2: XGBoost / Gradient Boosting (tabular only)")
    print("─" * 65)

    # Compute scale_pos_weight for class imbalance
    n_neg = (splits["train"]["y"] == 0).sum()
    n_pos = (splits["train"]["y"] == 1).sum()

    t0 = time.time()
    try:
        from xgboost import XGBClassifier
        clf = XGBClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            scale_pos_weight=n_neg / max(n_pos, 1),
            eval_metric="auc",
            random_state=42,
            verbosity=0,
        )
        model_name = "XGBoost"
    except ImportError:
        # Fallback to sklearn's GradientBoosting if xgboost not installed
        print("   ⚠️  xgboost not installed, using sklearn GradientBoosting")
        clf = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
        )
        model_name = "GradientBoosting"

    clf.fit(splits["train"]["X"], splits["train"]["y"])
    elapsed = time.time() - t0

    y_prob = clf.predict_proba(splits["test"]["X"])[:, 1]
    metrics = evaluate_model(model_name, splits["test"]["y"], y_prob)
    metrics["Train Time (s)"] = round(elapsed, 2)

    print(f"   Train time: {elapsed:.2f}s")
    print(f"   Test AUC: {metrics['AUC-ROC']:.4f} | F1: {metrics['F1']:.4f}")
    print(classification_report(
        splits["test"]["y"],
        (y_prob > 0.5).astype(int),
        target_names=["Normal", "Mule"],
        zero_division=0,
    ))

    # Feature importance
    if hasattr(clf, "feature_importances_"):
        from backend.gnn.features import get_feature_names
        feat_names = get_feature_names()
        importances = clf.feature_importances_
        top_k = np.argsort(importances)[::-1][:8]
        print("   📊 Top-8 Feature Importances:")
        for i, idx in enumerate(top_k):
            print(f"      {i+1}. {feat_names[idx]}: {importances[idx]:.4f}")

    return metrics


def run_gnn(pyg_data: Data) -> Dict[str, float]:
    """Train the ChainVigil GNN (uses graph edges + aggregation)."""
    print("\n" + "─" * 65)
    print("  🧠 Model 3: ChainVigil GNN (graph structure)")
    print("─" * 65)

    from backend.gnn.train import Trainer

    t0 = time.time()
    trainer = Trainer(pyg_data)
    results = trainer.train(epochs=200)
    elapsed = time.time() - t0

    # Get test metrics from the training results
    test_metrics = results.get("test_metrics", {})
    metrics = {
        "model": "ChainVigil GNN",
        "AUC-ROC": round(test_metrics.get("auc", 0), 4),
        "F1": round(test_metrics.get("f1", 0), 4),
        "Precision": round(test_metrics.get("precision", 0), 4),
        "Recall": round(test_metrics.get("recall", 0), 4),
        "Train Time (s)": round(elapsed, 2),
    }

    return metrics


# ═══════════════════════════════════════════════════════════════
#  Step 4: Comparison table
# ═══════════════════════════════════════════════════════════════

def print_results_table(results: List[Dict]):
    """Print a formatted comparison table."""
    print("\n" + "═" * 65)
    print("  📊 RESULTS: Model Comparison")
    print("═" * 65)

    df = pd.DataFrame(results)
    df.set_index("model", inplace=True)

    print(f"\n{df.to_string()}")

    print("\n" + "─" * 65)

    # Analysis
    gnn_auc = df.loc["ChainVigil GNN", "AUC-ROC"] if "ChainVigil GNN" in df.index else 0
    best_baseline = df.drop("ChainVigil GNN", errors="ignore")["AUC-ROC"].max()
    delta = gnn_auc - best_baseline

    print("\n  🔍 Analysis:")

    if best_baseline >= 0.99:
        print("  ⚠️  Baselines achieve ≥0.99 AUC — feature space is likely")
        print("     linearly separable. The GNN's graph structure may not be")
        print("     adding significant value on this synthetic data.")
        print("     → On REAL bank data (weaker signals), the GNN's relational")
        print("       inductive bias would likely provide a bigger advantage.")
    elif delta > 0.05:
        print(f"  ✅ GNN outperforms best baseline by +{delta:.4f} AUC.")
        print("     The graph structure is providing meaningful signal")
        print("     beyond tabular features alone.")
    elif delta > 0.01:
        print(f"  📊 GNN has a modest edge (+{delta:.4f} AUC) over baselines.")
        print("     Graph structure helps but features carry most signal.")
    else:
        print(f"  ⚖️  GNN and baselines perform similarly (Δ={delta:.4f}).")
        print("     On synthetic data, the strong features dominate.")

    print("\n  💡 Key Insight:")
    print("     Synthetic mule rings have strong, engineered signals")
    print("     (shared devices, high velocity, connected subgraphs).")
    print("     Real-world mule detection would have subtler patterns")
    print("     where graph neighborhood aggregation matters more.")

    print("\n" + "═" * 65)


# ═══════════════════════════════════════════════════════════════
#  Main
# ═══════════════════════════════════════════════════════════════

def main():
    # Prepare data (shared across all models)
    pyg_data, X_raw, y_raw, account_ids = prepare_data()

    # Get train/val/test splits (same masks for all)
    splits = get_split_arrays(pyg_data, X_raw, y_raw)

    print(f"\n📋 Dataset Summary:")
    print(f"   Total accounts: {len(y_raw)}")
    print(f"   Mules: {y_raw.sum()} ({y_raw.mean()*100:.1f}%)")
    print(f"   Train: {len(splits['train']['y'])} | "
          f"Val: {len(splits['val']['y'])} | "
          f"Test: {len(splits['test']['y'])}")
    print(f"   Features: {X_raw.shape[1]}")

    # Run all 3 models
    results = []

    r1 = run_logistic_regression(splits)
    results.append(r1)

    r2 = run_xgboost(splits)
    results.append(r2)

    r3 = run_gnn(pyg_data)
    results.append(r3)

    # Print comparison
    print_results_table(results)


if __name__ == "__main__":
    main()
