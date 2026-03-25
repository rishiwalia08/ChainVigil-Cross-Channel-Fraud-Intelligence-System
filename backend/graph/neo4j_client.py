"""
ChainVigil — Neo4j Client Wrapper

Provides connection pooling, query execution, and schema management
for the Neo4j graph database.
"""

try:
    from neo4j import GraphDatabase
    HAS_NEO4J = True
except ImportError:
    HAS_NEO4J = False
    GraphDatabase = None

from typing import List, Dict, Any, Optional
from backend.config import NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD


class Neo4jClient:
    """Thread-safe Neo4j driver wrapper with connection management."""

    def __init__(
        self,
        uri: str = NEO4J_URI,
        user: str = NEO4J_USER,
        password: str = NEO4J_PASSWORD
    ):
        self._driver = None
        if not HAS_NEO4J:
            print("⚠️  neo4j package not installed. Running in NetworkX-only mode.")
            return
        try:
            self._driver = GraphDatabase.driver(uri, auth=(user, password))
        except Exception as e:
            print(f"⚠️  Neo4j driver init failed: {e}")
            return
        self._verify_connectivity()

    def _verify_connectivity(self):
        try:
            self._driver.verify_connectivity()
            print("✅ Neo4j connection established")
        except Exception as e:
            print(f"⚠️  Neo4j connection failed: {e}")
            print("   Falling back to in-memory mode (NetworkX)")
            self._driver = None

    @property
    def is_connected(self) -> bool:
        return self._driver is not None

    def close(self):
        if self._driver:
            self._driver.close()

    def execute_query(
        self, query: str, parameters: Optional[Dict] = None
    ) -> List[Dict[str, Any]]:
        """Execute a Cypher query and return results as list of dicts."""
        if not self.is_connected:
            raise ConnectionError("Neo4j is not connected")

        with self._driver.session() as session:
            result = session.run(query, parameters or {})
            return [record.data() for record in result]

    def execute_write(
        self, query: str, parameters: Optional[Dict] = None
    ) -> None:
        """Execute a write Cypher query."""
        if not self.is_connected:
            raise ConnectionError("Neo4j is not connected")

        with self._driver.session() as session:
            session.run(query, parameters or {})

    def execute_batch(
        self, query: str, batch_data: List[Dict], batch_size: int = 500
    ) -> int:
        """Execute a Cypher query in batches for bulk ingestion."""
        if not self.is_connected:
            raise ConnectionError("Neo4j is not connected")

        total = 0
        for i in range(0, len(batch_data), batch_size):
            batch = batch_data[i : i + batch_size]
            with self._driver.session() as session:
                session.run(query, {"batch": batch})
            total += len(batch)
        return total

    def setup_schema(self) -> None:
        """Create constraints and indexes from schema.cypher."""
        schema_queries = [
            # Constraints
            "CREATE CONSTRAINT account_id_unique IF NOT EXISTS FOR (a:Account) REQUIRE a.account_id IS UNIQUE",
            "CREATE CONSTRAINT device_id_unique IF NOT EXISTS FOR (d:Device) REQUIRE d.device_id IS UNIQUE",
            "CREATE CONSTRAINT ip_address_unique IF NOT EXISTS FOR (ip:IPAddress) REQUIRE ip.address IS UNIQUE",
            "CREATE CONSTRAINT atm_id_unique IF NOT EXISTS FOR (atm:ATMTerminal) REQUIRE atm.atm_id IS UNIQUE",
            "CREATE CONSTRAINT transaction_id_unique IF NOT EXISTS FOR (t:Transaction) REQUIRE t.transaction_id IS UNIQUE",
            # Indexes
            "CREATE INDEX account_mule_idx IF NOT EXISTS FOR (a:Account) ON (a.is_mule)",
            "CREATE INDEX account_jurisdiction_idx IF NOT EXISTS FOR (a:Account) ON (a.jurisdiction)",
            "CREATE INDEX transaction_timestamp_idx IF NOT EXISTS FOR (t:Transaction) ON (t.timestamp)",
            "CREATE INDEX transaction_channel_idx IF NOT EXISTS FOR (t:Transaction) ON (t.channel_type)",
        ]
        for query in schema_queries:
            try:
                self.execute_write(query)
            except Exception as e:
                print(f"   Schema query warning: {e}")

        print("✅ Neo4j schema setup complete")

    def clear_database(self) -> None:
        """Delete all nodes and relationships (use with caution!)."""
        self.execute_write("MATCH (n) DETACH DELETE n")
        print("🗑️  Database cleared")

    def get_node_count(self, label: Optional[str] = None) -> int:
        """Get count of nodes, optionally filtered by label."""
        query = f"MATCH (n{':' + label if label else ''}) RETURN count(n) as cnt"
        result = self.execute_query(query)
        return result[0]["cnt"] if result else 0

    def get_edge_count(self, rel_type: Optional[str] = None) -> int:
        """Get count of relationships, optionally filtered by type."""
        query = f"MATCH ()-[r{':' + rel_type if rel_type else ''}]->() RETURN count(r) as cnt"
        result = self.execute_query(query)
        return result[0]["cnt"] if result else 0

    def get_graph_stats(self) -> Dict[str, int]:
        """Get comprehensive graph statistics."""
        return {
            "accounts": self.get_node_count("Account"),
            "devices": self.get_node_count("Device"),
            "ips": self.get_node_count("IPAddress"),
            "atm_terminals": self.get_node_count("ATMTerminal"),
            "transactions": self.get_node_count("Transaction"),
            "transferred_to": self.get_edge_count("TRANSFERRED_TO"),
            "used_device": self.get_edge_count("USED_DEVICE"),
            "logged_from": self.get_edge_count("LOGGED_FROM"),
            "withdrew_at": self.get_edge_count("WITHDREW_AT"),
        }
