"""
ChainVigil — Unified Entity Graph Builder

Constructs the Unified Entity Graph (UEG) from generated data.
Supports both Neo4j (persistent) and NetworkX (in-memory) backends.
"""

import os
import json
from typing import Dict, Optional

import pandas as pd
import networkx as nx

from backend.config import DATA_DIR
from backend.graph.neo4j_client import Neo4jClient


class GraphBuilder:
    """Builds the Unified Entity Graph from multi-channel data."""

    def __init__(self, neo4j_client: Optional[Neo4jClient] = None):
        self.neo4j = neo4j_client
        self.nx_graph = nx.MultiDiGraph()  # Always maintain NetworkX for GNN export
        self._stats = {
            "accounts": 0, "devices": 0, "ips": 0,
            "atm_terminals": 0, "transactions": 0, "edges": 0
        }

    # ─── Load Data ──────────────────────────────────────────────

    def load_data(self, data_dir: str = DATA_DIR) -> Dict[str, pd.DataFrame]:
        """Load generated CSV data from disk."""
        data = {}
        for name in ["accounts", "transactions", "devices", "ips", "atm_withdrawals"]:
            path = os.path.join(data_dir, f"{name}.csv")
            if os.path.exists(path):
                data[name] = pd.read_csv(path)
                print(f"   📂 Loaded {name}: {len(data[name])} rows")
            else:
                print(f"   ⚠️  Missing: {path}")
                data[name] = pd.DataFrame()

        rings_path = os.path.join(data_dir, "mule_rings.json")
        if os.path.exists(rings_path):
            with open(rings_path, "r") as f:
                data["rings"] = json.load(f)
            print(f"   📂 Loaded {len(data['rings'])} mule rings")
        else:
            data["rings"] = []

        return data

    # ─── Build Graph ────────────────────────────────────────────

    def build(self, data: Optional[Dict] = None, data_dir: str = DATA_DIR) -> nx.MultiDiGraph:
        """Build the full Unified Entity Graph."""
        if data is None:
            data = self.load_data(data_dir)

        print("\n🔨 Building Unified Entity Graph...")

        self._add_account_nodes(data.get("accounts", pd.DataFrame()))
        self._add_device_nodes(data.get("devices", pd.DataFrame()))
        self._add_ip_nodes(data.get("ips", pd.DataFrame()))
        self._add_transaction_edges(data.get("transactions", pd.DataFrame()))
        self._add_atm_edges(data.get("atm_withdrawals", pd.DataFrame()))

        print(f"\n✅ Graph built: {self.nx_graph.number_of_nodes()} nodes, "
              f"{self.nx_graph.number_of_edges()} edges")

        # Build Neo4j if available
        if self.neo4j and self.neo4j.is_connected:
            self._build_neo4j(data)

        return self.nx_graph

    # ─── Account Nodes ──────────────────────────────────────────

    def _add_account_nodes(self, df: pd.DataFrame):
        if df.empty:
            return
        for _, row in df.iterrows():
            self.nx_graph.add_node(
                row["account_id"],
                entity_type="Account",
                jurisdiction=row.get("jurisdiction", ""),
                jurisdiction_risk_weight=float(row.get("jurisdiction_risk_weight", 0)),
                account_type=row.get("account_type", ""),
                is_mule=bool(row.get("is_mule", False)),
            )
        self._stats["accounts"] = len(df)
        print(f"   ✓ {len(df)} Account nodes added")

    # ─── Device Nodes + Edges ───────────────────────────────────

    def _add_device_nodes(self, df: pd.DataFrame):
        if df.empty:
            return
        unique_devices = df["device_id"].unique()
        for dev_id in unique_devices:
            self.nx_graph.add_node(dev_id, entity_type="Device")
        self._stats["devices"] = len(unique_devices)

        for _, row in df.iterrows():
            self.nx_graph.add_edge(
                row["account_id"], row["device_id"],
                edge_type="USED_DEVICE"
            )
            self._stats["edges"] += 1
        print(f"   ✓ {len(unique_devices)} Device nodes, {len(df)} USED_DEVICE edges")

    # ─── IP Nodes + Edges ──────────────────────────────────────

    def _add_ip_nodes(self, df: pd.DataFrame):
        if df.empty:
            return
        unique_ips = df["ip_address"].unique()
        for ip in unique_ips:
            self.nx_graph.add_node(ip, entity_type="IPAddress")
        self._stats["ips"] = len(unique_ips)

        for _, row in df.iterrows():
            self.nx_graph.add_edge(
                row["account_id"], row["ip_address"],
                edge_type="LOGGED_FROM"
            )
            self._stats["edges"] += 1
        print(f"   ✓ {len(unique_ips)} IP nodes, {len(df)} LOGGED_FROM edges")

    # ─── Transaction Edges ─────────────────────────────────────

    def _add_transaction_edges(self, df: pd.DataFrame):
        if df.empty:
            return
        for _, row in df.iterrows():
            self.nx_graph.add_edge(
                row["source_id"], row["target_id"],
                edge_type="TRANSFERRED_TO",
                transaction_id=row.get("transaction_id", ""),
                amount=float(row.get("amount", 0)),
                timestamp=row.get("timestamp", ""),
                channel_type=row.get("channel_type", ""),
                geo_location=row.get("geo_location", ""),
                is_suspicious=bool(row.get("is_suspicious", False)),
            )
            self._stats["edges"] += 1
        self._stats["transactions"] = len(df)
        print(f"   ✓ {len(df)} TRANSFERRED_TO edges")

    # ─── ATM Edges ─────────────────────────────────────────────

    def _add_atm_edges(self, df: pd.DataFrame):
        if df.empty:
            return
        unique_atms = df["atm_id"].unique()
        for atm_id in unique_atms:
            self.nx_graph.add_node(atm_id, entity_type="ATMTerminal")
        self._stats["atm_terminals"] = len(unique_atms)

        for _, row in df.iterrows():
            self.nx_graph.add_edge(
                row["account_id"], row["atm_id"],
                edge_type="WITHDREW_AT",
                amount=float(row.get("amount", 0)),
                timestamp=row.get("timestamp", ""),
                geo_location=row.get("geo_location", ""),
            )
            self._stats["edges"] += 1
        print(f"   ✓ {len(unique_atms)} ATM nodes, {len(df)} WITHDREW_AT edges")

    # ─── Neo4j Ingestion ───────────────────────────────────────

    def _build_neo4j(self, data: Dict):
        """Ingest the graph into Neo4j."""
        print("\n📡 Ingesting into Neo4j...")

        try:
            self.neo4j.setup_schema()
            self.neo4j.clear_database()

            # Accounts
            accounts = data["accounts"].to_dict("records")
            n = self.neo4j.execute_batch(
                """UNWIND $batch AS row
                   CREATE (a:Account {
                       account_id: row.account_id,
                       holder_name: row.holder_name,
                       jurisdiction: row.jurisdiction,
                       jurisdiction_risk_weight: row.jurisdiction_risk_weight,
                       account_type: row.account_type,
                       created_at: row.created_at,
                       is_mule: row.is_mule
                   })""",
                accounts
            )
            print(f"   ✓ {n} Account nodes ingested")

            # Devices
            devices = data["devices"].to_dict("records")
            self.neo4j.execute_batch(
                """UNWIND $batch AS row
                   MERGE (d:Device {device_id: row.device_id})
                   WITH d, row
                   MATCH (a:Account {account_id: row.account_id})
                   CREATE (a)-[:USED_DEVICE]->(d)""",
                devices
            )
            print(f"   ✓ Device nodes + edges ingested")

            # IPs
            ips = data["ips"].to_dict("records")
            self.neo4j.execute_batch(
                """UNWIND $batch AS row
                   MERGE (ip:IPAddress {address: row.ip_address})
                   WITH ip, row
                   MATCH (a:Account {account_id: row.account_id})
                   CREATE (a)-[:LOGGED_FROM]->(ip)""",
                ips
            )
            print(f"   ✓ IP nodes + edges ingested")

            # Transactions
            txns = data["transactions"].to_dict("records")
            self.neo4j.execute_batch(
                """UNWIND $batch AS row
                   MATCH (src:Account {account_id: row.source_id})
                   MATCH (dst:Account {account_id: row.target_id})
                   CREATE (src)-[:TRANSFERRED_TO {
                       transaction_id: row.transaction_id,
                       amount: row.amount,
                       timestamp: row.timestamp,
                       channel_type: row.channel_type,
                       geo_location: row.geo_location,
                       is_suspicious: row.is_suspicious
                   }]->(dst)""",
                txns
            )
            print(f"   ✓ Transaction edges ingested")

            # ATM Withdrawals
            atm = data["atm_withdrawals"].to_dict("records")
            self.neo4j.execute_batch(
                """UNWIND $batch AS row
                   MERGE (atm:ATMTerminal {atm_id: row.atm_id})
                   WITH atm, row
                   MATCH (a:Account {account_id: row.account_id})
                   CREATE (a)-[:WITHDREW_AT {
                       amount: row.amount,
                       timestamp: row.timestamp
                   }]->(atm)""",
                atm
            )
            print(f"   ✓ ATM edges ingested")

            stats = self.neo4j.get_graph_stats()
            print(f"\n✅ Neo4j ingestion complete: {stats}")

        except Exception as e:
            print(f"⚠️  Neo4j ingestion error: {e}")
            print("   Continuing with NetworkX graph only.")

    # ─── Getters ───────────────────────────────────────────────

    def get_networkx_graph(self) -> nx.MultiDiGraph:
        return self.nx_graph

    def get_stats(self) -> dict:
        return {
            **self._stats,
            "nx_nodes": self.nx_graph.number_of_nodes(),
            "nx_edges": self.nx_graph.number_of_edges(),
        }

    # ─── Real-time Incremental Updates ────────────────────────

    def upsert_account(self, account_id: str, **attrs):
        """Create account node if missing, else update provided attributes."""
        if account_id not in self.nx_graph:
            self.nx_graph.add_node(
                account_id,
                entity_type="Account",
                is_mule=False,
                jurisdiction_risk_weight=float(attrs.get("jurisdiction_risk_weight", 0.0)),
                **attrs,
            )
            self._stats["accounts"] += 1
        else:
            self.nx_graph.nodes[account_id].update(attrs)

    def upsert_device_link(self, account_id: str, device_id: str):
        """Add device node/link used for real-time account session updates."""
        if device_id not in self.nx_graph:
            self.nx_graph.add_node(device_id, entity_type="Device")
            self._stats["devices"] += 1

        self.nx_graph.add_edge(account_id, device_id, edge_type="USED_DEVICE")
        self._stats["edges"] += 1

    def upsert_ip_link(self, account_id: str, ip_address: str):
        """Add IP node/link used for real-time account session updates."""
        if ip_address not in self.nx_graph:
            self.nx_graph.add_node(ip_address, entity_type="IPAddress")
            self._stats["ips"] += 1

        self.nx_graph.add_edge(account_id, ip_address, edge_type="LOGGED_FROM")
        self._stats["edges"] += 1

    def add_transaction_live(
        self,
        source_id: str,
        target_id: str,
        transaction_id: str,
        amount: float,
        channel_type: str,
        timestamp: str,
        geo_location: Optional[str] = None,
        device_id: Optional[str] = None,
        ip_address: Optional[str] = None,
        is_suspicious: bool = False,
    ):
        """
        Incrementally update graph with one incoming transaction.
        Avoids full graph rebuild for real-time scoring.
        """
        self.upsert_account(source_id)
        self.upsert_account(target_id)

        self.nx_graph.add_edge(
            source_id,
            target_id,
            edge_type="TRANSFERRED_TO",
            transaction_id=transaction_id,
            amount=float(amount),
            timestamp=timestamp,
            channel_type=channel_type,
            geo_location=geo_location or "",
            is_suspicious=bool(is_suspicious),
            ingestion_mode="realtime",
        )
        self._stats["transactions"] += 1
        self._stats["edges"] += 1

        if device_id:
            self.upsert_device_link(source_id, device_id)
        if ip_address:
            self.upsert_ip_link(source_id, ip_address)

        # Optional Neo4j incremental write
        if self.neo4j and self.neo4j.is_connected:
            try:
                self.neo4j.execute_write(
                    """
                    MERGE (src:Account {account_id: $source_id})
                    ON CREATE SET src.is_mule = false
                    MERGE (dst:Account {account_id: $target_id})
                    ON CREATE SET dst.is_mule = false
                    CREATE (src)-[:TRANSFERRED_TO {
                        transaction_id: $transaction_id,
                        amount: $amount,
                        timestamp: $timestamp,
                        channel_type: $channel_type,
                        geo_location: $geo_location,
                        is_suspicious: $is_suspicious,
                        ingestion_mode: 'realtime'
                    }]->(dst)
                    """,
                    {
                        "source_id": source_id,
                        "target_id": target_id,
                        "transaction_id": transaction_id,
                        "amount": float(amount),
                        "timestamp": timestamp,
                        "channel_type": channel_type,
                        "geo_location": geo_location or "",
                        "is_suspicious": bool(is_suspicious),
                    },
                )

                if device_id:
                    self.neo4j.execute_write(
                        """
                        MATCH (a:Account {account_id: $account_id})
                        MERGE (d:Device {device_id: $device_id})
                        CREATE (a)-[:USED_DEVICE]->(d)
                        """,
                        {"account_id": source_id, "device_id": device_id},
                    )

                if ip_address:
                    self.neo4j.execute_write(
                        """
                        MATCH (a:Account {account_id: $account_id})
                        MERGE (ip:IPAddress {address: $ip_address})
                        CREATE (a)-[:LOGGED_FROM]->(ip)
                        """,
                        {"account_id": source_id, "ip_address": ip_address},
                    )
            except Exception as e:
                print(f"⚠️  Neo4j realtime upsert failed: {e}")
