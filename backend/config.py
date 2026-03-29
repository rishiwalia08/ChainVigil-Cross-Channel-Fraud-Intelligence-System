"""
ChainVigil — Configuration & Database Connection Settings
"""
import os

# ─── Neo4j Configuration ────────────────────────────────────────────
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "chainvigil")

# ─── Data Generation Defaults ───────────────────────────────────────
NUM_ACCOUNTS = int(os.getenv("NUM_ACCOUNTS", "250"))
NUM_TRANSACTIONS = int(os.getenv("NUM_TRANSACTIONS", "1250"))
NUM_MULE_RINGS = int(os.getenv("NUM_MULE_RINGS", "5"))
MULE_RING_SIZE_RANGE = (4, 8)

# ─── Channels ───────────────────────────────────────────────────────
CHANNELS = ["UPI", "ATM", "WEB", "MOBILE_APP"]

# ─── GNN Configuration ─────────────────────────────────────────────
GNN_HIDDEN_DIM = 64
GNN_NUM_LAYERS = 2
GNN_LEARNING_RATE = 0.01
GNN_EPOCHS = 100
GNN_DROPOUT = 0.1
RISK_THRESHOLD = 0.85

# ─── Paths ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "sample_data")
MODEL_DIR = os.path.join(BASE_DIR, "gnn", "saved_models")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
