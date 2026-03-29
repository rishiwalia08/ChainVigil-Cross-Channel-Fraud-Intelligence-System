"""
ChainVigil — GNN Training Loop

Semi-supervised training with class imbalance handling.
Evaluates using AUC-ROC, F1, Precision, Recall.
"""

import os
import json
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    roc_auc_score, f1_score, precision_score,
    recall_score, classification_report
)
from torch_geometric.data import Data

from backend.gnn.model import ChainVigilGNN
from backend.config import (
    GNN_LEARNING_RATE, GNN_EPOCHS, GNN_HIDDEN_DIM,
    GNN_NUM_LAYERS, GNN_DROPOUT, MODEL_DIR
)


class Trainer:
    """GNN model trainer with evaluation and checkpointing."""

    def __init__(
        self,
        data: Data,
        hidden_dim: int = GNN_HIDDEN_DIM,
        num_layers: int = GNN_NUM_LAYERS,
        dropout: float = GNN_DROPOUT,
        lr: float = GNN_LEARNING_RATE,
    ):
        self.data = data
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize model
        self.model = ChainVigilGNN(
            in_channels=data.x.shape[1],
            hidden_channels=hidden_dim,
            num_layers=num_layers,
            dropout=dropout,
        ).to(self.device)

        # Handle class imbalance with weighted loss
        num_pos = data.y[data.train_mask].sum().item()
        num_neg = data.train_mask.sum().item() - num_pos
        pos_weight = torch.tensor([num_neg / max(num_pos, 1)]).to(self.device)

        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=lr, weight_decay=5e-4
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode="max", patience=20, factor=0.5
        )

        # Move data to device
        self.data = self.data.to(self.device)

        self.best_val_auc = 0.0
        self.history = {"train_loss": [], "val_auc": [], "val_f1": []}

    def train(self, epochs: int = GNN_EPOCHS) -> Dict:
        """Run the full training loop."""
        print(f"\n🧠 Training ChainVigil GNN on {self.device}")
        print(f"   Model params: {sum(p.numel() for p in self.model.parameters()):,}")
        print(f"   Epochs: {epochs} | LR: {self.optimizer.defaults['lr']}")
        print(f"   Train: {self.data.train_mask.sum()} | "
              f"Val: {self.data.val_mask.sum()} | "
              f"Test: {self.data.test_mask.sum()}")
        print("─" * 60)

        for epoch in range(1, epochs + 1):
            # ─── Train step ────────────────────────────────
            self.model.train()
            self.optimizer.zero_grad()

            probs, _ = self.model(self.data.x, self.data.edge_index)
            loss = self.criterion(
                probs[self.data.train_mask],
                self.data.y[self.data.train_mask].float()
            )
            loss.backward()
            self.optimizer.step()

            train_loss = loss.item()
            self.history["train_loss"].append(train_loss)

            # ─── Validation ────────────────────────────────
            if epoch % 10 == 0 or epoch == 1:
                val_metrics = self._evaluate(self.data.val_mask)
                self.history["val_auc"].append(val_metrics["auc"])
                self.history["val_f1"].append(val_metrics["f1"])

                self.scheduler.step(val_metrics["auc"])

                # Checkpoint best model
                if val_metrics["auc"] > self.best_val_auc:
                    self.best_val_auc = val_metrics["auc"]
                    self._save_checkpoint(epoch)

                if epoch % 50 == 0 or epoch == 1:
                    print(
                        f"   Epoch {epoch:>4d} | "
                        f"Loss: {train_loss:.4f} | "
                        f"Val AUC: {val_metrics['auc']:.4f} | "
                        f"Val F1: {val_metrics['f1']:.4f}"
                    )

        # ─── Final evaluation on test set ──────────────────
        print("─" * 60)
        self._load_best_checkpoint()
        test_metrics = self._evaluate(self.data.test_mask, verbose=True)

        results = {
            "best_val_auc": self.best_val_auc,
            "test_metrics": test_metrics,
            "epochs_trained": epochs,
            "model_params": sum(p.numel() for p in self.model.parameters()),
        }

        # Save results
        os.makedirs(MODEL_DIR, exist_ok=True)
        with open(os.path.join(MODEL_DIR, "training_results.json"), "w") as f:
            json.dump(results, f, indent=2)

        return results

    def _evaluate(self, mask: torch.Tensor, verbose: bool = False) -> Dict:
        """Evaluate model on masked nodes."""
        self.model.eval()
        with torch.no_grad():
            probs, _ = self.model(self.data.x, self.data.edge_index)

        probs_np = probs[mask].cpu().numpy()
        labels_np = self.data.y[mask].cpu().numpy()
        preds = (probs_np > 0.5).astype(int)

        try:
            auc = roc_auc_score(labels_np, probs_np)
        except ValueError:
            auc = 0.0

        f1 = f1_score(labels_np, preds, zero_division=0)
        precision = precision_score(labels_np, preds, zero_division=0)
        recall = recall_score(labels_np, preds, zero_division=0)

        metrics = {
            "auc": float(auc),
            "f1": float(f1),
            "precision": float(precision),
            "recall": float(recall),
        }

        if verbose:
            print(f"\n📊 Test Results:")
            print(f"   AUC-ROC:   {auc:.4f}")
            print(f"   F1 Score:  {f1:.4f}")
            print(f"   Precision: {precision:.4f}")
            print(f"   Recall:    {recall:.4f}")
            print(f"\n{classification_report(labels_np, preds, target_names=['Normal', 'Mule'])}")

        return metrics

    def predict(self) -> np.ndarray:
        """Get mule probability scores for ALL nodes."""
        self.model.eval()
        with torch.no_grad():
            probs, _ = self.model(self.data.x, self.data.edge_index)
        return probs.cpu().numpy()

    def get_embeddings(self) -> np.ndarray:
        """Get node embeddings for visualization / XAI."""
        self.model.eval()
        with torch.no_grad():
            embeddings = self.model.get_embedding(
                self.data.x, self.data.edge_index
            )
        return embeddings.cpu().numpy()

    def _save_checkpoint(self, epoch: int):
        """Save model checkpoint."""
        os.makedirs(MODEL_DIR, exist_ok=True)
        path = os.path.join(MODEL_DIR, "best_model.pt")
        torch.save({
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "best_val_auc": self.best_val_auc,
        }, path)

    def _load_best_checkpoint(self):
        """Load best model checkpoint."""
        path = os.path.join(MODEL_DIR, "best_model.pt")
        if os.path.exists(path):
            checkpoint = torch.load(path, map_location=self.device, weights_only=True)
            self.model.load_state_dict(checkpoint["model_state_dict"])
            print(f"   ✅ Loaded best model (epoch {checkpoint['epoch']}, "
                  f"AUC {checkpoint['best_val_auc']:.4f})")


# ─── CLI Entry Point ──────────────────────────────────────────────

if __name__ == "__main__":
    from backend.data.generator import generate_all_data
    from backend.graph.builder import GraphBuilder
    from backend.gnn.dataset import nx_to_pyg

    print("═" * 60)
    print("  ChainVigil — GNN Training Pipeline")
    print("═" * 60)

    # Step 1: Generate data
    data_dict = generate_all_data()

    # Step 2: Build graph
    builder = GraphBuilder()
    G = builder.build(data_dict)

    # Step 3: Convert to PyG
    print("\n📐 Converting to PyTorch Geometric...")
    pyg_data, node_mapping, account_ids = nx_to_pyg(G)

    # Step 4: Train
    trainer = Trainer(pyg_data)
    results = trainer.train()

    print(f"\n✅ Training complete! Best Val AUC: {results['best_val_auc']:.4f}")
