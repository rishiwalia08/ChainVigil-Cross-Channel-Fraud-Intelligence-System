// ChainVigil — Neo4j Schema Setup
// Run this against your Neo4j instance to create constraints and indexes

// ─── Constraints ─────────────────────────────────────────────────

CREATE CONSTRAINT account_id_unique IF NOT EXISTS
FOR (a:Account) REQUIRE a.account_id IS UNIQUE;

CREATE CONSTRAINT device_id_unique IF NOT EXISTS
FOR (d:Device) REQUIRE d.device_id IS UNIQUE;

CREATE CONSTRAINT ip_address_unique IF NOT EXISTS
FOR (ip:IPAddress) REQUIRE ip.address IS UNIQUE;

CREATE CONSTRAINT atm_id_unique IF NOT EXISTS
FOR (atm:ATMTerminal) REQUIRE atm.atm_id IS UNIQUE;

CREATE CONSTRAINT transaction_id_unique IF NOT EXISTS
FOR (t:Transaction) REQUIRE t.transaction_id IS UNIQUE;


// ─── Indexes for Performance ────────────────────────────────────

CREATE INDEX account_mule_idx IF NOT EXISTS
FOR (a:Account) ON (a.is_mule);

CREATE INDEX account_jurisdiction_idx IF NOT EXISTS
FOR (a:Account) ON (a.jurisdiction);

CREATE INDEX transaction_timestamp_idx IF NOT EXISTS
FOR (t:Transaction) ON (t.timestamp);

CREATE INDEX transaction_channel_idx IF NOT EXISTS
FOR (t:Transaction) ON (t.channel_type);


// ─── Node Schema Reference ──────────────────────────────────────
// Account:    account_id, holder_name, jurisdiction, jurisdiction_risk_weight,
//             account_type, created_at, is_mule
// Device:     device_id
// IPAddress:  address
// ATMTerminal: atm_id, geo_location
// Transaction: transaction_id, amount, timestamp, channel_type, geo_location,
//              is_suspicious

// ─── Edge Schema Reference ──────────────────────────────────────
// (Account)-[:TRANSFERRED_TO {amount, timestamp, channel_type}]->(Account)
// (Account)-[:USED_DEVICE]->(Device)
// (Account)-[:LOGGED_FROM]->(IPAddress)
// (Account)-[:WITHDREW_AT {amount, timestamp}]->(ATMTerminal)
// (Account)-[:INITIATED]->(Transaction)
// (Transaction)-[:RECEIVED_BY]->(Account)
