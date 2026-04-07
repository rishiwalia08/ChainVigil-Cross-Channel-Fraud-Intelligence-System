"""
ChainVigil — Configuration & Database Connection Settings
"""
import os

# ─── Neo4j Configuration ────────────────────────────────────────────
NEO4J_URI = os.getenv("NEO4J_URI", "bolt://localhost:7687")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "chainvigil")

# ─── Data Generation Defaults ───────────────────────────────────────
NUM_ACCOUNTS = int(os.getenv("NUM_ACCOUNTS", "900"))        # scaled up for more test diversity
NUM_TRANSACTIONS = int(os.getenv("NUM_TRANSACTIONS", "4500"))  # maintain edge density
NUM_MULE_RINGS = int(os.getenv("NUM_MULE_RINGS", "20"))     # 20 rings → ~100-120 mule nodes
MULE_RING_SIZE_RANGE = (4, 8)   # restored size: more mules per ring

# ─── Channels ───────────────────────────────────────────────────────
CHANNELS = ["UPI", "ATM", "WEB", "MOBILE_APP"]

# ─── GNN Configuration ─────────────────────────────────────────────
GNN_HIDDEN_DIM = 48   # enough capacity to learn mule patterns
GNN_NUM_LAYERS = 3    # 3-hop aggregation for ring detection
GNN_LEARNING_RATE = 0.005
GNN_EPOCHS = 40
GNN_DROPOUT = 0.35    # standard dropout — not hostile
RISK_THRESHOLD = 0.55  # balanced threshold for 100-200 escalations

# ─── Paths ──────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data", "sample_data")
MODEL_DIR = os.path.join(BASE_DIR, "gnn", "saved_models")

os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODEL_DIR, exist_ok=True)
