"""
ChainVigil — GNN Model Architecture

Implements a hybrid GraphSAGE + GAT model with temporal attention
for mule account detection. Outputs per-node mule probability scores.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import SAGEConv, GATConv, BatchNorm

from backend.config import GNN_HIDDEN_DIM, GNN_NUM_LAYERS, GNN_DROPOUT


class ChainVigilGNN(nn.Module):
    """
    Hybrid GraphSAGE + GAT model for mule detection.

    Architecture:
      Input → [SAGEConv → BN → ReLU → Dropout] × L
            → GATConv (multi-head attention)
            → MLP classifier → Sigmoid
    """

    def __init__(
        self,
        in_channels: int,
        hidden_channels: int = GNN_HIDDEN_DIM,
        num_layers: int = GNN_NUM_LAYERS,
        dropout: float = GNN_DROPOUT,
        num_heads: int = 4,
    ):
        super().__init__()

        self.num_layers = num_layers
        self.dropout = dropout

        # ─── GraphSAGE layers ──────────────────────────────
        self.sage_convs = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        self.sage_convs.append(SAGEConv(in_channels, hidden_channels))
        self.batch_norms.append(BatchNorm(hidden_channels))

        for _ in range(num_layers - 2):
            self.sage_convs.append(SAGEConv(hidden_channels, hidden_channels))
            self.batch_norms.append(BatchNorm(hidden_channels))

        # ─── GAT attention layer ───────────────────────────
        self.gat_conv = GATConv(
            hidden_channels,
            hidden_channels // num_heads,
            heads=num_heads,
            dropout=dropout,
            concat=True,
        )
        self.gat_bn = BatchNorm(hidden_channels)

        # ─── MLP classifier ───────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 2, hidden_channels // 4),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_channels // 4, 1),
        )

    def forward(self, x, edge_index):
        """
        Forward pass.

        Args:
            x: Node feature matrix [num_nodes, in_channels]
            edge_index: Edge indices [2, num_edges]

        Returns:
            probs: Mule probability per node [num_nodes, 1]
            embeddings: Node embeddings after GNN layers [num_nodes, hidden]
        """
        # ─── GraphSAGE message passing ─────────────────────
        h = x
        for i in range(self.num_layers - 1):
            h = self.sage_convs[i](h, edge_index)
            h = self.batch_norms[i](h)
            h = F.relu(h)
            h = F.dropout(h, p=self.dropout, training=self.training)

        # ─── GAT attention ─────────────────────────────────
        h = self.gat_conv(h, edge_index)
        h = self.gat_bn(h)
        h = F.relu(h)

        embeddings = h  # Save for visualization / XAI

        # ─── Classification ────────────────────────────────
        logits = self.classifier(h)
        probs = torch.sigmoid(logits)

        return probs.squeeze(-1), embeddings

    def get_embedding(self, x, edge_index):
        """Get node embeddings without classification (for XAI)."""
        h = x
        for i in range(self.num_layers - 1):
            h = self.sage_convs[i](h, edge_index)
            h = self.batch_norms[i](h)
            h = F.relu(h)

        h = self.gat_conv(h, edge_index)
        h = self.gat_bn(h)
        h = F.relu(h)
        return h
