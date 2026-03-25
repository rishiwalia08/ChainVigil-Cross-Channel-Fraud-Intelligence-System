"""
ChainVigil — Synthetic Multi-Channel Transaction Data Generator

Generates realistic transaction data with embedded mule ring patterns:
 - 500+ accounts across multiple jurisdictions
 - 2500+ transactions across UPI / ATM / Web / MobileApp channels
 - 3-5 mule rings (clusters of 4-8 accounts) with high-velocity patterns
 - Shared devices & IPs among mule accounts
 - Temporal patterns mimicking real laundering behavior
"""

import os
import json
import random
import uuid
from datetime import datetime, timedelta
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd
from faker import Faker

from backend.config import (
    NUM_ACCOUNTS, NUM_TRANSACTIONS, NUM_MULE_RINGS,
    MULE_RING_SIZE_RANGE, CHANNELS, DATA_DIR
)

fake = Faker()
Faker.seed(42)
random.seed(42)
np.random.seed(42)


# ─── Jurisdiction Risk Profiles ─────────────────────────────────────

JURISDICTIONS = {
    "IN": 0.2,    # India – low base risk
    "US": 0.15,
    "UK": 0.1,
    "NG": 0.6,    # Nigeria – elevated
    "RU": 0.55,
    "CN": 0.4,
    "AE": 0.35,
    "PH": 0.5,
    "KE": 0.45,
    "BR": 0.3,
}

GEO_LOCATIONS = [
    "Mumbai", "Delhi", "Bangalore", "Hyderabad", "Chennai",
    "Lagos", "Abuja", "Moscow", "Dubai", "Manila",
    "Nairobi", "São Paulo", "London", "New York", "Shanghai"
]


def _generate_device_id() -> str:
    return f"DEV-{uuid.uuid4().hex[:10].upper()}"


def _generate_ip() -> str:
    return fake.ipv4_public()


def _generate_atm_id() -> str:
    return f"ATM-{random.choice(GEO_LOCATIONS).upper()[:3]}-{random.randint(1000, 9999)}"


# ─── Account Generation ────────────────────────────────────────────

def generate_accounts(n: int = NUM_ACCOUNTS) -> pd.DataFrame:
    """Generate n accounts with risk weights and metadata."""
    accounts = []
    for i in range(n):
        jurisdiction = random.choice(list(JURISDICTIONS.keys()))
        accounts.append({
            "account_id": f"ACC-{i:05d}",
            "holder_name": fake.name(),
            "jurisdiction": jurisdiction,
            "jurisdiction_risk_weight": JURISDICTIONS[jurisdiction],
            "account_type": random.choice(["SAVINGS", "CURRENT", "WALLET"]),
            "created_at": fake.date_between(start_date="-2y", end_date="-30d").isoformat(),
            "is_mule": False,  # Will be updated for mule rings
        })
    return pd.DataFrame(accounts)


# ─── Mule Ring Injection ───────────────────────────────────────────

def inject_mule_rings(
    accounts_df: pd.DataFrame,
    num_rings: int = NUM_MULE_RINGS
) -> Tuple[pd.DataFrame, List[Dict]]:
    """
    Mark clusters of accounts as mule rings and return ring metadata.
    Mule accounts get elevated jurisdiction risk and shared devices.
    """
    all_account_ids = accounts_df["account_id"].tolist()
    used_ids = set()
    rings = []

    for ring_idx in range(num_rings):
        ring_size = random.randint(*MULE_RING_SIZE_RANGE)
        available = [a for a in all_account_ids if a not in used_ids]
        if len(available) < ring_size:
            break

        ring_members = random.sample(available, ring_size)
        used_ids.update(ring_members)

        # Shared device & IP for mule ring (partial overlap)
        shared_devices = [_generate_device_id() for _ in range(max(1, ring_size // 3))]
        shared_ips = [_generate_ip() for _ in range(max(1, ring_size // 3))]

        ring_meta = {
            "ring_id": f"MULE_RING_{ring_idx:02d}",
            "members": ring_members,
            "shared_devices": shared_devices,
            "shared_ips": shared_ips,
            "hub_account": ring_members[0],  # First account is the hub
        }
        rings.append(ring_meta)

        # Mark mule accounts in dataframe
        mask = accounts_df["account_id"].isin(ring_members)
        accounts_df.loc[mask, "is_mule"] = True
        # Elevate jurisdiction risk for mule accounts
        accounts_df.loc[mask, "jurisdiction_risk_weight"] = accounts_df.loc[
            mask, "jurisdiction_risk_weight"
        ].apply(lambda x: min(1.0, x + random.uniform(0.15, 0.35)))

    return accounts_df, rings


# ─── Device & IP Mapping ──────────────────────────────────────────

def generate_device_ip_mapping(
    accounts_df: pd.DataFrame,
    rings: List[Dict]
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Generate device and IP mappings.
    Mule accounts share devices/IPs within their ring.
    Normal accounts get unique devices.
    """
    device_rows = []
    ip_rows = []
    ring_lookup = {}

    for ring in rings:
        for member in ring["members"]:
            ring_lookup[member] = ring

    for _, acc in accounts_df.iterrows():
        acc_id = acc["account_id"]
        if acc_id in ring_lookup:
            ring = ring_lookup[acc_id]
            # Mule: use shared device + possibly own device
            devices = [random.choice(ring["shared_devices"])]
            if random.random() < 0.3:
                devices.append(_generate_device_id())
            ips = [random.choice(ring["shared_ips"])]
            if random.random() < 0.3:
                ips.append(_generate_ip())
        else:
            # Normal: unique device and IP
            devices = [_generate_device_id()]
            ips = [_generate_ip()]

        for dev in devices:
            device_rows.append({"account_id": acc_id, "device_id": dev})
        for ip in ips:
            ip_rows.append({"account_id": acc_id, "ip_address": ip})

    return pd.DataFrame(device_rows), pd.DataFrame(ip_rows)


# ─── Transaction Generation ───────────────────────────────────────

def generate_transactions(
    accounts_df: pd.DataFrame,
    rings: List[Dict],
    n: int = NUM_TRANSACTIONS
) -> pd.DataFrame:
    """
    Generate transactions mixing normal behavior with mule patterns.
    ~60% normal, ~40% mule ring activity.
    """
    all_ids = accounts_df["account_id"].tolist()
    mule_ids = accounts_df[accounts_df["is_mule"]]["account_id"].tolist()
    normal_ids = accounts_df[~accounts_df["is_mule"]]["account_id"].tolist()

    transactions = []
    base_time = datetime.now() - timedelta(days=30)

    # ── Normal transactions (~60%) ──────────────────────────────
    n_normal = int(n * 0.6)
    for _ in range(n_normal):
        src = random.choice(all_ids)
        dst = random.choice([a for a in all_ids if a != src])
        ts = base_time + timedelta(
            days=random.randint(0, 29),
            hours=random.randint(0, 23),
            minutes=random.randint(0, 59),
            seconds=random.randint(0, 59)
        )
        transactions.append({
            "transaction_id": f"TXN-{uuid.uuid4().hex[:12].upper()}",
            "source_id": src,
            "target_id": dst,
            "amount": round(random.uniform(100, 50000), 2),
            "channel_type": random.choice(CHANNELS),
            "timestamp": ts.isoformat(),
            "geo_location": random.choice(GEO_LOCATIONS),
            "is_suspicious": False,
        })

    # ── Mule ring transactions (~40%) ───────────────────────────
    n_mule = n - n_normal
    txns_per_ring = n_mule // max(1, len(rings))

    for ring in rings:
        members = ring["members"]
        hub = ring["hub_account"]

        for _ in range(txns_per_ring):
            pattern = random.choice(["chain", "hub_spoke", "circular", "smurfing"])

            if pattern == "chain":
                # Rapid multi-hop chain: A→B→C→D within minutes
                chain_length = random.randint(3, min(7, len(members)))
                chain = random.sample(members, chain_length)
                chain_start = base_time + timedelta(
                    days=random.randint(0, 29),
                    hours=random.randint(0, 23),
                    minutes=random.randint(0, 59)
                )
                base_amount = round(random.uniform(10000, 100000), 2)
                for j in range(len(chain) - 1):
                    ts = chain_start + timedelta(minutes=random.randint(1, 3) * (j + 1))
                    transactions.append({
                        "transaction_id": f"TXN-{uuid.uuid4().hex[:12].upper()}",
                        "source_id": chain[j],
                        "target_id": chain[j + 1],
                        "amount": round(base_amount * random.uniform(0.85, 0.99), 2),
                        "channel_type": random.choice(CHANNELS),
                        "timestamp": ts.isoformat(),
                        "geo_location": random.choice(GEO_LOCATIONS),
                        "is_suspicious": True,
                    })

            elif pattern == "hub_spoke":
                # Hub receives from outside then distributes
                outside_src = random.choice(normal_ids) if normal_ids else random.choice(all_ids)
                ts = base_time + timedelta(
                    days=random.randint(0, 29),
                    hours=random.randint(0, 23)
                )
                total_amount = round(random.uniform(50000, 200000), 2)
                transactions.append({
                    "transaction_id": f"TXN-{uuid.uuid4().hex[:12].upper()}",
                    "source_id": outside_src,
                    "target_id": hub,
                    "amount": total_amount,
                    "channel_type": "UPI",
                    "timestamp": ts.isoformat(),
                    "geo_location": random.choice(GEO_LOCATIONS),
                    "is_suspicious": True,
                })
                # Hub distributes to ring members
                num_dists = random.randint(2, min(4, len(members) - 1))
                targets = [m for m in members if m != hub]
                for t in random.sample(targets, min(num_dists, len(targets))):
                    ts_out = ts + timedelta(minutes=random.randint(2, 15))
                    transactions.append({
                        "transaction_id": f"TXN-{uuid.uuid4().hex[:12].upper()}",
                        "source_id": hub,
                        "target_id": t,
                        "amount": round(total_amount / num_dists * random.uniform(0.8, 1.1), 2),
                        "channel_type": random.choice(CHANNELS),
                        "timestamp": ts_out.isoformat(),
                        "geo_location": random.choice(GEO_LOCATIONS),
                        "is_suspicious": True,
                    })

            elif pattern == "circular":
                # Circular flow: A→B→C→A
                cycle_len = random.randint(3, min(5, len(members)))
                cycle = random.sample(members, cycle_len)
                cycle_start = base_time + timedelta(
                    days=random.randint(0, 29),
                    hours=random.randint(0, 23)
                )
                amount = round(random.uniform(20000, 80000), 2)
                for j in range(cycle_len):
                    src = cycle[j]
                    dst = cycle[(j + 1) % cycle_len]
                    ts = cycle_start + timedelta(minutes=random.randint(2, 8) * (j + 1))
                    transactions.append({
                        "transaction_id": f"TXN-{uuid.uuid4().hex[:12].upper()}",
                        "source_id": src,
                        "target_id": dst,
                        "amount": round(amount * random.uniform(0.9, 1.0), 2),
                        "channel_type": random.choice(CHANNELS),
                        "timestamp": ts.isoformat(),
                        "geo_location": random.choice(GEO_LOCATIONS),
                        "is_suspicious": True,
                    })

            elif pattern == "smurfing":
                # Many small transactions just under threshold
                src = random.choice(members)
                dst = random.choice([m for m in members if m != src])
                smurf_start = base_time + timedelta(
                    days=random.randint(0, 29),
                    hours=random.randint(6, 22)
                )
                for s in range(random.randint(5, 12)):
                    ts = smurf_start + timedelta(minutes=random.randint(1, 5) * (s + 1))
                    transactions.append({
                        "transaction_id": f"TXN-{uuid.uuid4().hex[:12].upper()}",
                        "source_id": src,
                        "target_id": dst,
                        "amount": round(random.uniform(500, 9900), 2),  # Under 10k threshold
                        "channel_type": random.choice(CHANNELS),
                        "timestamp": ts.isoformat(),
                        "geo_location": random.choice(GEO_LOCATIONS),
                        "is_suspicious": True,
                    })

    random.shuffle(transactions)
    return pd.DataFrame(transactions)


# ─── ATM Withdrawal Generation ────────────────────────────────────

def generate_atm_withdrawals(
    accounts_df: pd.DataFrame,
    rings: List[Dict]
) -> pd.DataFrame:
    """Generate ATM withdrawal records. Mule accounts have more ATM activity."""
    withdrawals = []
    base_time = datetime.now() - timedelta(days=30)
    mule_ids = set(accounts_df[accounts_df["is_mule"]]["account_id"])

    for _, acc in accounts_df.iterrows():
        acc_id = acc["account_id"]
        # Mule accounts: more ATM withdrawals
        n_withdrawals = random.randint(3, 10) if acc_id in mule_ids else random.randint(0, 2)

        for _ in range(n_withdrawals):
            ts = base_time + timedelta(
                days=random.randint(0, 29),
                hours=random.randint(6, 23),
                minutes=random.randint(0, 59)
            )
            withdrawals.append({
                "account_id": acc_id,
                "atm_id": _generate_atm_id(),
                "amount": round(random.uniform(2000, 25000), 2),
                "timestamp": ts.isoformat(),
                "geo_location": random.choice(GEO_LOCATIONS),
            })

    return pd.DataFrame(withdrawals)


# ─── Master Generator ─────────────────────────────────────────────

def generate_all_data(
    num_accounts: int = NUM_ACCOUNTS,
    num_transactions: int = NUM_TRANSACTIONS,
    num_mule_rings: int = NUM_MULE_RINGS,
    save: bool = True,
) -> Dict[str, pd.DataFrame]:
    """Generate all synthetic data and optionally save to disk."""

    print(f"🏗️  Generating {num_accounts} accounts...")
    accounts_df = generate_accounts(num_accounts)

    print(f"💀 Injecting {num_mule_rings} mule rings...")
    accounts_df, rings = inject_mule_rings(accounts_df, num_mule_rings)
    mule_count = accounts_df["is_mule"].sum()
    print(f"   → {mule_count} mule accounts across {len(rings)} rings")

    print(f"📱 Generating device & IP mappings...")
    devices_df, ips_df = generate_device_ip_mapping(accounts_df, rings)

    print(f"💰 Generating ~{num_transactions} transactions...")
    transactions_df = generate_transactions(accounts_df, rings, num_transactions)
    print(f"   → {len(transactions_df)} total transactions generated")

    print(f"🏧 Generating ATM withdrawals...")
    atm_df = generate_atm_withdrawals(accounts_df, rings)
    print(f"   → {len(atm_df)} ATM withdrawals generated")

    data = {
        "accounts": accounts_df,
        "transactions": transactions_df,
        "devices": devices_df,
        "ips": ips_df,
        "atm_withdrawals": atm_df,
        "rings": rings,
    }

    if save:
        os.makedirs(DATA_DIR, exist_ok=True)
        accounts_df.to_csv(os.path.join(DATA_DIR, "accounts.csv"), index=False)
        transactions_df.to_csv(os.path.join(DATA_DIR, "transactions.csv"), index=False)
        devices_df.to_csv(os.path.join(DATA_DIR, "devices.csv"), index=False)
        ips_df.to_csv(os.path.join(DATA_DIR, "ips.csv"), index=False)
        atm_df.to_csv(os.path.join(DATA_DIR, "atm_withdrawals.csv"), index=False)

        with open(os.path.join(DATA_DIR, "mule_rings.json"), "w") as f:
            json.dump(rings, f, indent=2)

        print(f"\n✅ All data saved to {DATA_DIR}")

    return data


# ─── CLI Entry Point ──────────────────────────────────────────────

if __name__ == "__main__":
    generate_all_data()
