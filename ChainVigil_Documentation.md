# ChainVigil-Cross-Channel-Fraud-Intelligence-System

**Cross-Channel Fraud Intelligence using Graph Intelligence & GNN**

**Repository:** https://github.com/rishiwalia08/ChainVigil-Cross-Channel-Fraud-Intelligence-System

## Complete Project Documentation

---

## Table of Contents

1. [Problem Statement](#1-problem-statement)
2. [What is ChainVigil?](#2-what-is-chainvigil)
3. [Key Terminology](#3-key-terminology)
4. [System Architecture](#4-system-architecture)
5. [Technology Stack](#5-technology-stack)
6. [Phase 1 — Data Generation](#6-phase-1--data-generation)
7. [Phase 2 — Graph Construction](#7-phase-2--graph-construction)
8. [Phase 3 — GNN Model & Training](#8-phase-3--gnn-model--training)
9. [Phase 4 — Risk Analysis & Cluster Detection](#9-phase-4--risk-analysis--cluster-detection)
10. [Phase 5 — Explainable AI (XAI)](#10-phase-5--explainable-ai-xai)
11. [Phase 6 — Audit Reporting](#11-phase-6--audit-reporting)
12. [Phase 7 — Dashboard & Visualization](#12-phase-7--dashboard--visualization)
13. [API Reference](#13-api-reference)
14. [File Structure & Module Descriptions](#14-file-structure--module-descriptions)
15. [Phase 8 — Privacy-Preserving Inter-Bank Sharing](#15-phase-8--privacy-preserving-inter-bank-sharing)
16. [Experimental Evaluation & Model Justification](#16-experimental-evaluation--model-justification)
17. [How to Run](#17-how-to-run)

---

## 1. Problem Statement

### What is Money Laundering?

Money laundering is the process of making illegally obtained money (from drug trafficking, fraud, terrorism, etc.) appear legitimate. Criminals cannot deposit large sums directly into banks because that would trigger regulatory alerts. Instead, they use a technique involving **money mule accounts**.

### What is a Money Mule?

A money mule is a person (often recruited unknowingly) whose bank account is used to transfer illegal funds. The criminal sends dirty money through a chain of mule accounts, each making small transfers to the next, until the money appears "clean."

**Example of a mule chain:**

```
Criminal Account
    ↓ ₹9,500 (via UPI, 10:00 AM)
Mule Account A
    ↓ ₹9,200 (via Mobile App, 10:08 AM)
Mule Account B
    ↓ ₹8,800 (via Web Banking, 10:13 AM)
Mule Account C
    ↓ ₹8,500 (via ATM withdrawal, 10:16 AM)
Criminal's "Clean" Account
```

Each transfer is:
- **Under ₹10,000** — to avoid automated detection thresholds (this is called "smurfing")
- **Very fast** — the entire chain completes within minutes
- **Across different channels** — UPI, ATM, Web, Mobile App to avoid single-channel monitoring

### Why Traditional Systems Fail

Traditional fraud detection systems are **rule-based** and **transaction-level**:
- They check each transaction individually: "Is this single transfer suspicious?"
- They use simple rules like: "Flag if amount > ₹50,000" or "Flag if > 10 transactions per day"

**The problem:** Each individual transaction in a mule chain looks perfectly normal. ₹9,500 via UPI is a regular transaction. The suspicion only appears when you look at the **pattern across multiple accounts** — the chain, the speed, the shared devices.

### What ChainVigil Does Differently

ChainVigil doesn't look at individual transactions. It builds a **graph** (a network) of ALL accounts, devices, IP addresses, and ATM terminals, and asks:

> "Which groups of connected accounts are behaving suspiciously AS A GROUP?"

It uses a **Graph Neural Network (GNN)** — a type of artificial intelligence specifically designed to learn patterns on networks — to automatically detect mule ring clusters.

---

## 2. What is ChainVigil?

ChainVigil is a **graph-native financial crime detection system** designed to identify cross-channel money mule networks in near real-time. 

It integrates multi-source transaction logs (UPI, ATM, Web, Mobile App) into a **Unified Entity Graph (UEG)** and applies **Graph Neural Networks (GNNs)** to detect high-velocity fund movement and mule-ring clusters.

### What It Flags

| Suspicious Pattern | Description |
|-------------------|-------------|
| **High-velocity fund hops** | Money moving through 3+ accounts within minutes |
| **Transaction fragmentation (Smurfing)** | Many small transfers just under reporting thresholds |
| **Circular fund flows** | Money going A→B→C→A in a loop |
| **Shared device/IP usage** | Multiple "unrelated" accounts using the same phone or internet connection |
| **Cross-channel layering** | Rapid switching between UPI, ATM, Web, and Mobile channels |

---

## 3. Key Terminology

### Graph Theory Terms

| Term | Definition |
|------|-----------|
| **Graph** | A mathematical structure consisting of **nodes** (entities) connected by **edges** (relationships). Think of it like a social network diagram. |
| **Node (Vertex)** | A single entity in the graph. In our case: an Account, Device, IP Address, or ATM Terminal. |
| **Edge (Link)** | A connection between two nodes. In our case: a money transfer, a device login, an IP connection, or an ATM withdrawal. |
| **Directed Graph** | A graph where edges have a direction (A→B is different from B→A). Money transfers are directional. |
| **In-Degree** | The number of edges coming INTO a node. For an account: how many other accounts send money to it. |
| **Out-Degree** | The number of edges going OUT of a node. For an account: how many other accounts it sends money to. |
| **PageRank** | A score measuring how "important" a node is in the network. Originally invented by Google to rank web pages. High PageRank means many important nodes point to you. |
| **Betweenness Centrality** | How many shortest paths between other nodes pass through this node. Nodes with high betweenness are "bridges" — they connect different parts of the network. Mule accounts often sit on these bridges. |
| **Clustering Coefficient** | How tightly connected a node's neighbors are to each other. In mule rings, neighbors tend to be highly interconnected. |
| **Connected Component** | A group of nodes where you can reach any node from any other by following edges. Each mule ring forms its own connected component among flagged accounts. |

### Machine Learning Terms

| Term | Definition |
|------|-----------|
| **GNN (Graph Neural Network)** | A type of neural network designed to learn from graph-structured data. Unlike regular neural networks that process tables or images, GNNs can process nodes AND their connections. |
| **GraphSAGE** | A GNN architecture that learns by **sampling and aggregating** features from a node's neighbors. It says: "Tell me about your friends, and I'll tell you who you are." |
| **GAT (Graph Attention Network)** | A GNN architecture that uses **attention mechanisms** to learn which neighbors are more important. Not all connections matter equally — some neighbors are more relevant than others. |
| **Node Embedding** | A numerical vector (list of numbers) that represents a node's position and role in the graph. Similar nodes get similar embeddings. |
| **Feature** | A measurable property of a node. Examples: transaction count, average transfer amount, number of shared devices. |
| **Feature Engineering** | The process of creating useful features from raw data. We compute 20+ features for each account. |
| **Epoch** | One complete pass through the training data. We train for 200 epochs. |
| **Loss Function** | A mathematical function that measures how wrong the model's predictions are. The model tries to minimize this. |
| **BCEWithLogitsLoss** | Binary Cross-Entropy loss — used when predicting a probability (0 or 1, mule or not mule). |
| **Class Imbalance** | When one class (normal accounts, ~95%) vastly outnumbers the other (mules, ~5%). Without handling this, the model would just predict "everyone is normal" and still be 95% accurate. |
| **AUC-ROC** | Area Under the Receiver Operating Characteristic curve. Measures how well the model distinguishes between mules and normal accounts. 1.0 = perfect, 0.5 = random guessing. |
| **F1 Score** | The harmonic mean of Precision and Recall. Balances between catching all mules (recall) and not falsely accusing innocents (precision). |
| **Precision** | Of all accounts the model flagged as mules, what percentage actually are mules? |
| **Recall** | Of all actual mules, what percentage did the model catch? |
| **Train/Validation/Test Split** | Dividing data into three sets: Train (60%) for learning, Validation (20%) for tuning, Test (20%) for final evaluation. |

### Financial Crime Terms

| Term | Definition |
|------|-----------|
| **Money Mule** | A person whose bank account is used (knowingly or unknowingly) to transfer illicit funds. |
| **Mule Ring** | A coordinated group of mule accounts working together to launder money. |
| **Smurfing** | Breaking a large transaction into many smaller ones (below reporting thresholds) to avoid detection. Named after the tiny blue characters. |
| **Layering** | The second stage of money laundering — moving funds through multiple transactions/channels to obscure the trail. |
| **Hub-Spoke Pattern** | One central account (hub) receives money from many sources and distributes to many destinations (spokes). |
| **Circular Flow** | Money moving in a loop: A→B→C→A. Often used to create the appearance of legitimate business activity. |
| **KYC (Know Your Customer)** | Regulatory requirement for banks to verify the identity of their customers. |
| **STR (Suspicious Transaction Report)** | A report filed by banks to regulators when suspicious activity is detected. |
| **XAI (Explainable AI)** | Techniques that make AI decisions understandable to humans. Required by regulators — they won't accept "the AI said so." |

---

## 4. System Architecture

### High-Level Pipeline

```
STEP 1: DATA GENERATION
  Multi-channel transaction logs (UPI, ATM, Web, Mobile)
  500 accounts, 2500+ transactions, 5 mule rings
                        ↓
STEP 2: GRAPH CONSTRUCTION
  Unified Entity Graph (UEG)
  Nodes: Account, Device, IP, ATM
  Edges: Transfer, UsedDevice, LoggedFrom, WithdrewAt
                        ↓
STEP 3: FEATURE ENGINEERING
  20+ features per account node
  Topological, temporal, behavioral metrics
                        ↓
STEP 4: GNN TRAINING
  GraphSAGE + GAT hybrid model
  Semi-supervised learning with class-balanced loss
                        ↓
STEP 5: RISK SCORING
  Mule probability (0-100%) per account
  Action recommendations: Escalate / Freeze / Monitor / Clear
                        ↓
STEP 6: CLUSTER DETECTION
  Connected component analysis on flagged accounts
  Velocity, volume, and hub identification per cluster
                        ↓
STEP 7: XAI & REPORTING
  Gradient × Input feature attribution
  Human-readable explanations + JSON audit reports
                        ↓
STEP 8: DASHBOARD
  Interactive graph visualization
  Risk tables, cluster cards, XAI auditor, report viewer
```

### Dual Graph Backend

ChainVigil uses two graph backends:

| Backend | Purpose |
|---------|---------|
| **NetworkX** (Python, in-memory) | Used for GNN feature computation, graph analysis, and exporting to PyTorch Geometric. Fast, no external dependencies. |
| **Neo4j** (Graph database, optional) | Used for persistent storage, complex Cypher queries, and production-grade graph operations. Falls back to NetworkX if unavailable. |

---

## 5. Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| Backend API | **FastAPI** (Python) | REST API server with auto-generated Swagger documentation |
| Graph Processing | **NetworkX** | In-memory graph construction and analysis |
| Graph Database | **Neo4j** (optional) | Persistent graph storage and Cypher queries |
| Deep Learning | **PyTorch** | Neural network framework for the GNN model |
| Graph ML | **PyTorch Geometric** | GNN layers (GraphSAGE, GAT), graph data handling |
| Data Processing | **Pandas, NumPy** | Tabular data manipulation and numerical computation |
| Data Generation | **Faker** | Generating realistic synthetic names, locations, etc. |
| ML Metrics | **scikit-learn** | AUC, F1, Precision, Recall computation |
| Frontend | **React + Vite** | Dashboard UI with hot module replacement |
| Graph Visualization | **react-force-graph-2d** | Interactive force-directed graph rendering |
| Data Validation | **Pydantic** | Request/response schema validation |

---

## 6. Phase 1 — Data Generation

**File:** `backend/data/generator.py`

Since real bank transaction data is confidential, ChainVigil generates **synthetic but realistic** data.

### What Gets Generated

| Entity | Count | Description |
|--------|-------|-------------|
| Accounts | 500 | Each with name, bank, country, jurisdiction risk weight |
| Transactions | 2,500+ | Cross-channel transfers with amounts, timestamps, channels |
| Devices | ~100 | Phones/laptops used by accounts |
| IP Addresses | ~80 | Internet connections used by accounts |
| ATM Withdrawals | ~670 | Cash withdrawals at physical ATMs |
| Mule Rings | 5 | Groups of 4-8 accounts with hidden criminal patterns |

### Mule Ring Patterns Injected

**1. Chain Pattern**
```
A → B → C → D → E
Money hops in a straight line, each transfer under 10 minutes
```

**2. Hub-Spoke Pattern**
```
    B
    ↑
C ← A → D
    ↓
    E
One central hub account receives and distributes
```

**3. Circular Flow Pattern**
```
A → B → C → A
Money goes in a loop to obscure the trail
```

**4. Smurfing Pattern**
```
A → B (₹3,200)
A → B (₹4,800)
A → B (₹2,100)
A → B (₹5,900)
Many small amounts to stay under ₹10,000 threshold
```

### Shared Resources

Mule accounts within the same ring also share:
- **Devices** — Multiple "unrelated" accounts logging in from the same phone
- **IP Addresses** — Multiple accounts using the same internet connection

This mimics real-world behavior where mule operators manage multiple accounts from the same location/device.

---

## 7. Phase 2 — Graph Construction

**Files:** `backend/graph/builder.py`, `backend/graph/neo4j_client.py`

### The Unified Entity Graph (UEG)

All generated data is connected into a single network.

**Node Types:**

| Node | Symbol | Example |
|------|--------|---------|
| Account | 🔵 | ACC-001 (Rajesh Kumar, SBI, India) |
| Device | 🟢 | DEV-042 (Samsung Galaxy) |
| IP Address | 🟣 | 192.168.1.45 |
| ATM Terminal | 🟡 | ATM-Mumbai-Central-003 |

**Edge Types:**

| Edge | Direction | Meaning |
|------|-----------|---------|
| TRANSFERRED_TO | A → B | Account A sent money to Account B |
| USED_DEVICE | A → D | Account A logged in using Device D |
| LOGGED_FROM | A → IP | Account A connected from IP address |
| WITHDREW_AT | A → ATM | Account A withdrew cash at this ATM |

**Edge Properties:**
- `amount` — How much money was transferred
- `timestamp` — When the transfer happened
- `channel_type` — Which channel (UPI/ATM/Web/Mobile)
- `is_suspicious` — Whether this edge is part of a known mule pattern

### Graph Scale

| Metric | Typical Value |
|--------|--------------|
| Total Nodes | ~2,100 |
| Total Edges | ~7,900 |
| Account Nodes | 500 |
| Device Nodes | ~100 |
| IP Nodes | ~80 |
| ATM Nodes | ~670 |

---

## 8. Phase 3 — GNN Model & Training

### Feature Engineering

**File:** `backend/gnn/features.py`

Before the AI model can learn, we compute **20+ numerical features** for every account node.

**Topological Features (Graph Structure):**

| Feature | What It Measures |
|---------|-----------------|
| `in_degree` | How many accounts send money TO this account |
| `out_degree` | How many accounts this account sends TO |
| `total_degree` | Total connections (in + out) |
| `pagerank` | Network importance score (Google PageRank algorithm) |
| `betweenness_centrality` | How many shortest paths pass through this account |
| `clustering_coefficient` | How tightly connected neighbors are to each other |
| `avg_neighbor_degree` | Average connectivity of this account's neighbors |

**Temporal Features (Time-based):**

| Feature | What It Measures |
|---------|-----------------|
| `avg_velocity_seconds` | Average time between receiving and sending money |
| `min_velocity_seconds` | Fastest turnaround time |
| `transaction_frequency` | How many transactions per time period |

**Behavioral Features (Activity Patterns):**

| Feature | What It Measures |
|---------|-----------------|
| `total_amount_sent` | Total money sent out |
| `total_amount_received` | Total money received |
| `avg_amount_sent` | Average transfer size outgoing |
| `avg_amount_received` | Average transfer size incoming |
| `amount_ratio` | Ratio of incoming to outgoing amounts |
| `channel_diversity` | Number of different channels used (1-4) |

**Resource Sharing Features:**

| Feature | What It Measures |
|---------|-----------------|
| `shared_device_count` | How many other accounts share the same device |
| `shared_ip_count` | How many other accounts share the same IP |
| `unique_devices_used` | Number of different devices used |
| `jurisdiction_risk` | Risk weight based on the account's country |

### Data Conversion

**File:** `backend/gnn/dataset.py`

Converts the NetworkX graph into a PyTorch Geometric `Data` object:
1. Extracts account nodes only (non-account nodes are used for feature computation)
2. Normalizes all features using **z-score normalization** (subtract mean, divide by standard deviation)
3. Creates edge indices for account-to-account transfers
4. Assigns binary labels: 1 = mule, 0 = normal
5. Creates stratified masks: 60% train, 20% validation, 20% test

### The GNN Model

**File:** `backend/gnn/model.py`

**Architecture: Hybrid GraphSAGE + GAT**

```
Input Features (20 per node)
        ↓
[GraphSAGE Layer 1] — Aggregate neighbor features
        ↓ BatchNorm + ReLU + Dropout
[GraphSAGE Layer 2] — Deeper neighborhood
        ↓ BatchNorm + ReLU + Dropout
[GAT Layer 3] — Attention-weighted aggregation (4 heads)
        ↓ BatchNorm + ReLU + Dropout
[MLP Classifier] — 2-layer fully connected network
        ↓
Output: Mule Probability (0.0 to 1.0)
```

**Why this architecture?**
- **GraphSAGE** is efficient and scalable — it samples a fixed number of neighbors rather than using all of them
- **GAT** adds attention — not all neighbors matter equally; mule connections should get more attention weight
- **Multi-head attention** (4 heads) lets the model look at relationships from multiple angles simultaneously

### Training Process

**File:** `backend/gnn/train.py`

1. **Forward pass**: Feed the graph through the model, get mule probability for each account
2. **Compute loss**: Compare predictions to known labels using weighted BCE loss
3. **Class balancing**: Mule class gets higher loss weight (because there are fewer mules)
4. **Backpropagation**: Adjust model weights to reduce error
5. **Learning rate scheduling**: Gradually reduce learning rate for fine-tuning
6. **Checkpointing**: Save the best model (highest validation AUC)
7. **Repeat** for 200 epochs

**Training Parameters:**

| Parameter | Value |
|-----------|-------|
| Hidden dimensions | 128 |
| Number of GNN layers | 3 |
| Attention heads | 4 |
| Learning rate | 0.001 |
| Epochs | 200 |
| Dropout | 0.3 |
| Train/Val/Test split | 60/20/20 |

---

## 9. Phase 4 — Risk Analysis & Cluster Detection

### Risk Scoring

**File:** `backend/gnn/predict.py`

After training, the model predicts a **mule probability** for every account:

| Score Range | Action | Meaning |
|-------------|--------|---------|
| 85-100% | 🚨 **Escalate** | Immediate investigation required |
| 60-84% | 🧊 **Freeze** | Temporarily freeze account, review pending |
| 40-59% | 👀 **Monitor** | Enhanced monitoring, watchlist |
| 0-39% | ✅ **Clear** | No suspicious activity detected |

### Cluster Detection

**File:** `backend/risk/engine.py`

Goes beyond individual scores to find **mule ring clusters**:

1. **Filter** all accounts with mule probability above threshold
2. **Build subgraph** of only flagged accounts and their connections
3. **Find connected components** — groups of flagged accounts connected by transfers
4. **Filter** clusters with 3+ members (too small = noise)

**For each cluster, compute:**

| Metric | Description |
|--------|-------------|
| `size` | Number of member accounts |
| `density` | How interconnected members are (0 to 1) |
| `total_volume` | Total money flowing through the cluster |
| `avg_velocity_seconds` | Average time between transfers within the cluster |
| `hub_account` | The most connected member (likely the operator) |
| `channels_used` | Which payment channels the cluster uses |
| `avg_risk_score` | Average mule probability across members |

---

## 10. Phase 5 — Explainable AI (XAI)

**File:** `backend/xai/explainer.py`

### Why XAI Matters

Regulators (RBI, FinCEN, etc.) require **explainability**. A bank cannot file a Suspicious Transaction Report (STR) saying "our AI flagged this account." They need to explain:
- **What** specific behaviors triggered the flag
- **How much** each behavior contributed to the decision
- **Why** this account is different from normal accounts

### How It Works: Gradient × Input

ChainVigil uses **Gradient × Input** attribution, a technique that measures how much each input feature contributed to the model's output:

1. **Forward pass**: Run the account through the model, get its mule probability
2. **Backward pass**: Compute gradients — how much would the probability change if each feature changed slightly?
3. **Multiply**: Gradient × actual feature value = importance score
4. **Normalize**: Scale all importance scores to sum to 100%

### Example Output

```
Account: ACC-042
Mule Probability: 91.3%
Recommended Action: ESCALATE

Top Contributing Features:
1. shared_device_count   — 34.2% (shares a phone with 3 flagged accounts)
2. avg_velocity_seconds  — 22.1% (money moves through in under 5 minutes)
3. betweenness_centrality — 18.4% (sits on bridge between suspicious clusters)
4. channel_diversity      — 12.7% (uses all 4 channels — unusual for normal accounts)
5. amount_ratio           — 8.3% (receives and immediately sends similar amounts)

Human-Readable Reasoning:
"Account ACC-042 flagged with 91.3% mule probability.
 Key risk factors: high shared device usage (3 shared accounts),
 rapid fund velocity (avg 4.2 minutes), and high betweenness
 centrality indicating a bridging position in the transfer network."
```

---

## 11. Phase 6 — Audit Reporting

**File:** `backend/xai/report.py`

Generates **structured JSON audit reports** for regulatory compliance:

### Report Structure

```json
{
  "report_id": "AUDIT-2025-02-27-001",
  "generated_at": "2025-02-27T17:30:00Z",
  "summary": {
    "total_accounts_analyzed": 500,
    "flagged_accounts": 28,
    "clusters_detected": 5,
    "risk_distribution": {
      "escalate": 12,
      "freeze": 8,
      "monitor": 8,
      "clear": 472
    }
  },
  "clusters": [
    {
      "cluster_id": "RING-001",
      "members": ["ACC-042", "ACC-087", "ACC-123", "ACC-156"],
      "hub_account": "ACC-042",
      "total_volume": 450000,
      "pattern_type": "chain_with_smurfing"
    }
  ],
  "account_assessments": [
    {
      "account_id": "ACC-042",
      "mule_probability": 0.913,
      "recommended_action": "Escalate",
      "top_features": [...],
      "xai_reasoning": "..."
    }
  ]
}
```

---

## 12. Phase 7 — Dashboard & Visualization

**Files:** `frontend-app/src/App.jsx`, `frontend-app/src/index.css`

### Dashboard Tabs

| Tab | What It Shows |
|-----|--------------|
| **Pipeline** | Control panel to run each step, progress indicators, pipeline logs, training metrics (AUC, F1, Precision, Recall), risk distribution chart |
| **Graph** | Interactive force-directed graph visualization of the Unified Entity Graph. Color-coded nodes by type, directional arrows for transfers, red edges for suspicious links. Drag, zoom, hover for details, click to explain. |
| **Accounts** | Sortable table of all accounts with risk scores, color-coded risk bars, action badges (Escalate/Freeze/Monitor/Clear), and an "Explain" button for each account |
| **Clusters** | Cards for each detected mule ring cluster showing member count, density, total volume, average velocity, hub account, channels used, and clickable member badges |
| **XAI Auditor** | Feature attribution visualization with horizontal bars showing how much each feature contributed to an account's flag. Includes human-readable reasoning text. |
| **Reports** | Full JSON audit report viewer with summary stats and downloadable report data |

### Graph Visualization Features

- **Color-coded nodes**: Blue=Account, Red=Mule, Green=Device, Purple=IP, Amber=ATM
- **Directional arrows**: Show money flow direction on transfer edges
- **Red edges**: Highlight suspicious transfers within mule rings
- **Hover tooltips**: Show node ID, type, risk score, and cluster membership
- **Click interaction**: Click any account node to jump to its XAI explanation
- **Drag & zoom**: Rearrange nodes and zoom into areas of interest
- **Force simulation**: Nodes naturally cluster based on their connections

---

## 13. API Reference

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/` | API info and status |
| GET | `/health` | Health check |
| POST | `/api/generate` | Generate synthetic transaction data |
| POST | `/api/ingest` | Build the Unified Entity Graph |
| POST | `/api/train` | Train the GNN model |
| POST | `/api/analyze` | Run risk analysis and cluster detection |
| POST | `/api/pipeline/run` | Run the entire pipeline end-to-end |
| GET | `/api/graph/stats` | Get graph statistics (node/edge counts) |
| GET | `/api/graph/visual` | Get graph data for visualization (nodes + links) |
| GET | `/api/accounts` | List all accounts with risk scores |
| GET | `/api/accounts/{id}` | Get detailed account info with neighbors |
| GET | `/api/clusters` | List all detected mule ring clusters |
| GET | `/api/clusters/{id}` | Get specific cluster details |
| GET | `/api/explain/{id}` | Get XAI explanation for an account |
| GET | `/api/report` | Generate full audit report |

---

## 14. File Structure & Module Descriptions

```
ChainVigil-Cross-Channel-Fraud-Intelligence-System/
├── README.md                          # Project overview
├── ChainVigil_Documentation.md        # This document
│
├── backend/
│   ├── config.py                      # Central configuration (DB, GNN params, paths)
│   ├── main.py                        # FastAPI server (15+ API endpoints)
│   ├── requirements.txt               # Python dependencies
│   │
│   ├── data/
│   │   ├── generator.py               # Synthetic data factory (500 accounts, 4 mule patterns)
│   │   └── sample_data/               # Generated CSV files (accounts, transactions, devices, IPs)
│   │
│   ├── graph/
│   │   ├── neo4j_client.py            # Neo4j wrapper (optional, falls back to NetworkX)
│   │   ├── builder.py                 # Unified Entity Graph constructor
│   │   └── schema.cypher              # Neo4j constraints and indexes
│   │
│   ├── gnn/
│   │   ├── features.py                # 20+ feature engineering (topological, temporal, behavioral)
│   │   ├── dataset.py                 # NetworkX → PyTorch Geometric converter
│   │   ├── model.py                   # GraphSAGE + GAT hybrid GNN architecture
│   │   ├── train.py                   # Training loop (200 epochs, class-balanced loss)
│   │   ├── predict.py                 # Inference & action recommendations
│   │   └── saved_models/              # Trained model checkpoints (.pt files)
│   │
│   ├── risk/
│   │   └── engine.py                  # Cluster detection, velocity metrics, hub identification
│   │
│   ├── xai/
│   │   ├── explainer.py               # Gradient×Input feature attribution + reasoning
│   │   └── report.py                  # JSON audit report generator
│   │
│   └── models/
│       └── schemas.py                 # Pydantic request/response models
│
└── frontend-app/
    ├── index.html                     # Entry HTML with SEO meta
    ├── package.json                   # NPM dependencies
    └── src/
        ├── main.jsx                   # React entry point
        ├── App.jsx                    # Dashboard (6 tabs, all API integration)
        └── index.css                  # Light theme design system
```

---

## 15. Phase 8 — Privacy-Preserving Inter-Bank Sharing

### Why Is This Needed?

Financial institutions need to collaborate to detect cross-bank mule networks, but sharing raw customer data violates privacy regulations (GDPR, RBI guidelines, etc.). ChainVigil solves this by providing an **anonymization layer** that strips all Personally Identifiable Information (PII) while preserving the graph intelligence needed for cross-referencing suspicious patterns.

### What Data Is Hidden vs. Preserved

| What's Removed (PII) | What's Preserved (Intelligence) |
|---|---|
| Account holder names | Graph topology (who connects to whom) |
| Bank names & countries | Risk scores & recommended actions |
| Device IDs & IP addresses | Behavioral features (degree, velocity, channel diversity) |
| ATM locations | Cluster memberships |
| Exact transaction amounts | Amount buckets (e.g., "5K-10K", "25K-50K") |

### How the Anonymization Works

1. **Account ID Hashing** — All account IDs are replaced with **SHA-256 hashes** using a secret salt. For example, `ACC-001` becomes `a3f8c2e1b9d04752`. This is a one-way transformation — the original ID cannot be recovered from the hash.

2. **Amount Bucketing** — Exact transaction amounts are converted into ranges to prevent re-identification:
   - `< ₹5,000` → `<5K`
   - `₹5,000 – ₹10,000` → `5K-10K`
   - `₹10,000 – ₹25,000` → `10K-25K`
   - `₹25,000 – ₹50,000` → `25K-50K`
   - `₹50,000 – ₹1,00,000` → `50K-1L`
   - `> ₹1,00,000` → `>1L`

3. **Device/IP Stripping** — All device and IP nodes are completely excluded from the export. Only account-to-account transfer edges are included.

4. **Behavioral Features Retained** — Non-identifying graph metrics like in-degree, out-degree, channel diversity, shared device count, and jurisdiction risk are preserved so receiving institutions can assess risk without knowing who the account belongs to.

### What the Exported Data Contains

The anonymized export (`GET /api/export/anonymized`) returns:
- **Anonymized Nodes** — Hashed account IDs with risk scores, recommended actions, flagged status, cluster membership, and behavioral features.
- **Anonymized Edges** — Transfer relationships between hashed accounts with channel type, amount bucket, and suspicion flag.
- **Anonymized Clusters** — Cluster summaries with size, density, avg risk score, channels used, and hashed member list.
- **Metadata** — Hash algorithm info, list of removed PII fields, and list of retained data types.

### Use Case

A bank running ChainVigil detects a suspicious cluster. It exports the anonymized data and shares it with partner banks. The receiving bank can:
- Check if any of the hashed IDs match patterns in their own system (using the same salt for cross-referencing)
- Understand the structure and risk level of the mule ring without ever seeing real customer names or account numbers
- Take preventive action on their side based on behavioral features and risk scores

---

## 16. Experimental Evaluation & Model Justification

### 16.1 Objective

This section evaluates whether the Graph Neural Network (GNN) architecture used in ChainVigil provides measurable performance advantages over classical tabular machine learning models.

We investigate:

1. Does relational modeling (GraphSAGE + GAT) outperform traditional classifiers?
2. Is the graph structure necessary given the engineered features?
3. Are results free from label leakage or contamination?

---

### 16.2 Dataset Overview

Synthetic dataset generated using `backend/data/generator.py`:

- **500 accounts**
- **2,500+ cross-channel transactions**
- **~2,100 total nodes**
- **~7,900 total edges**
- **5 injected mule rings**
- Class imbalance: ~5–7% mule accounts

---

### 16.3 Feature Set (20+ Features)

Features computed in `features.py` include:

#### Topological Features
- `in_degree`
- `out_degree`
- `total_degree`
- `pagerank`
- `betweenness_centrality`
- `clustering_coefficient`

#### Temporal Features
- `avg_velocity_seconds`
- `min_velocity_seconds`
- `max_velocity_seconds`

#### Behavioral Features
- `total_in_amount`
- `total_out_amount`
- `avg_in_amount`
- `avg_out_amount`
- `amount_ratio`
- `channel_diversity`

#### Resource Sharing Features
- `shared_device_count`
- `shared_ip_count`
- `atm_withdrawal_count`
- `atm_total_amount`
- `jurisdiction_risk_weight`

**Important:**
- No feature uses `is_mule`
- No feature uses cluster labels
- Cluster detection occurs only *after* GNN scoring

---

### 16.4 Data Integrity Validation

To ensure experimental rigor:

#### ✅ Node-Level Stratified Split

Implemented in `_create_masks()` inside `dataset.py`.

- Mule and normal accounts separated
- Each class shuffled independently
- Split proportionally (60% train, 20% val, 20% test)
- Ensures balanced representation across splits

#### ✅ No Label Leakage Through Edges

- `edge_index` built only from `TRANSFERRED_TO` edges
- Edge attributes: `amount`, `channel_type`
- Labels stored only in `y`
- No edge contains mule status

#### ✅ No Cluster Label Contamination

- Clustering performed only after model inference
- `clustering_coefficient` refers to NetworkX triangle metric
- No feature depends on risk engine output

---

### 16.5 Models Compared

All models trained on identical splits.

| Model | Description |
|--------|------------|
| Logistic Regression | Linear classifier on engineered features |
| Gradient Boosting | Tree-based ensemble classifier |
| ChainVigil GNN | Hybrid GraphSAGE + GAT architecture |

---

### 16.6 Experimental Results

| Model | AUC-ROC | F1 | Precision | Recall | Training Time |
|--------|---------|----|-----------|--------|---------------|
| Logistic Regression | 1.0 | 1.0 | 1.0 | 1.0 | 0.01s |
| Gradient Boosting | 1.0 | 1.0 | 1.0 | 1.0 | 0.17s |
| ChainVigil GNN | 1.0 | 0.93 | 0.88 | 1.0 | 3.95s |

---

### 16.7 Interpretation

#### Perfect Linear Separability

The dataset is **linearly separable**.

Top three discriminative features:

- `shared_device_count` (~25% contribution)
- `total_degree` (~24%)
- `shared_ip_count` (~23%)

Because synthetic mule patterns were injected with strong structural signals:

- Tabular models perfectly separate classes
- GNN provides no additional predictive advantage
- Message passing introduces minor variance (1 false positive)

This confirms:

> Engineered features alone fully capture synthetic mule behavior.

---

### 16.8 Why the GNN Slightly Underperforms

The GNN achieves F1 = 0.93 due to one false positive.

Reason:

- Message passing aggregates neighbor embeddings.
- When features already perfectly separate classes, aggregation can introduce smoothing noise.
- In a low-noise regime, relational modeling is unnecessary.

This reflects dataset characteristics, not model weakness.

---

### 16.9 Real-World Implications

Synthetic data characteristics:

| Synthetic | Real-World |
|------------|------------|
| Clean device sharing | Shared devices common in households |
| Unique IP correlations | NAT/VPN noise |
| Clear velocity patterns | Irregular timing |
| Distinct clusters | Overlapping financial networks |

In real-world settings:

- Individual features become noisy
- Suspicion emerges from relational context
- Multi-hop structure becomes critical

GNN advantages under noise:

- Relational inductive bias
- Multi-hop contextual reasoning
- Robustness when single-node features are ambiguous

---

### 16.10 Architectural Justification

#### Why GraphSAGE?
- Efficient neighbor sampling
- Scalable to large graphs
- Suitable for inductive node classification

#### Why Add GAT?
- Attention-weighted aggregation
- Not all neighbors equally informative
- Enhances representation learning

#### Why 3 Layers?

Each layer aggregates 1-hop neighbors:

- 1 layer → direct neighbors
- 2 layers → 2-hop context
- 3 layers → cluster-scale context

More layers risk **oversmoothing**:
- Node embeddings converge
- Representations become indistinguishable
- Classification degrades

Three layers balance contextual depth and stability.

---

### 16.11 Computational Trade-Off

| Model | Complexity | Scalability |
|--------|------------|------------|
| Logistic | O(n·d) | Extremely scalable |
| Gradient Boosting | O(n·log n) | Highly scalable |
| GNN | O(E·d) | Requires sampling for large graphs |

On small graphs, tabular models are faster.
On large relational networks, sampling-based GNN becomes advantageous.

---

### 16.12 Limitations

The synthetic generator creates:

- Strong structural signals
- Minimal feature overlap between classes
- High signal-to-noise ratio

This leads to perfect separability.

To better simulate real-world fraud:

- Inject partial device sharing in normal users
- Blur velocity distributions
- Add high-degree benign hubs
- Introduce adversarial noise

---

### 16.13 Conclusion

On clean synthetic data, tabular models achieve perfect classification.

However, ChainVigil demonstrates:

- End-to-end graph-native fraud detection architecture
- Proper ML benchmarking against baselines
- Absence of leakage or contamination
- Architectural reasoning for relational modeling

The experiment validates both implementation correctness and scientific rigor, while acknowledging dataset limitations.

---

## 17. How to Run

### Prerequisites
- Python 3.10+
- Node.js 18+
- pip (Python package manager)
- npm (Node package manager)

### Step 1: Install Backend Dependencies
```bash
cd ChainVigil-Cross-Channel-Fraud-Intelligence-System
pip install -r backend/requirements.txt
pip install torch torch-geometric
```

### Step 2: Start the Backend Server
```bash
python -m uvicorn backend.main:app --reload --port 8000
```
The API will be available at `http://localhost:8000`
Swagger docs at `http://localhost:8000/docs`

### Step 3: Start the Frontend
```bash
cd frontend-app
npm install
npm run dev
```
The dashboard will be available at `http://localhost:5173`

### Step 4: Run the Pipeline
1. Open `http://localhost:5173` in your browser
2. Click **"Run Full Pipeline"**
3. Wait for all 4 steps to complete (takes ~1-2 minutes)
4. Explore the Graph, Accounts, Clusters, XAI, and Reports tabs

---

*Document generated for ChainVigil-Cross-Channel-Fraud-Intelligence-System v1.0.0 — Cross-Channel Fraud Intelligence using Graph Intelligence & GNN*
