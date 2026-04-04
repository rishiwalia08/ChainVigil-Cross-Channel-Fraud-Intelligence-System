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
    Normal accounts are MOSTLY unique, but ~20% share devices/IPs to simulate
    family members sharing a phone, or office workers on the same network.
    This destroys the clean binary split of shared_device_count=0 for normals.
    """
    device_rows = []
    ip_rows = []
    ring_lookup = {}

    for ring in rings:
        for member in ring["members"]:
            ring_lookup[member] = ring

    # --- Build "innocent sharing" pools for normal accounts --------
    # ~20% of normal accounts are grouped into small clusters (2-3 accounts)
    # that share one device or IP, mimicking family/office scenarios.
    normal_ids = [
        acc["account_id"] for _, acc in accounts_df.iterrows()
        if acc["account_id"] not in ring_lookup
        and not str(acc["account_id"]).startswith("ACC-HN-")
    ]
    random.shuffle(normal_ids)
    n_shared_normals = int(len(normal_ids) * 0.20)
    shared_normal_ids = normal_ids[:n_shared_normals]

    # Group into pairs/triples
    normal_shared_device = {}  # acc_id -> shared_device_id
    normal_shared_ip = {}      # acc_id -> shared_ip
    i = 0
    while i < len(shared_normal_ids):
        cluster_size = random.randint(2, 3)
        cluster = shared_normal_ids[i:i + cluster_size]
        i += cluster_size
        if len(cluster) < 2:
            break
        shared_dev = _generate_device_id()
        shared_ip_addr = _generate_ip()
        for acc_id in cluster:
            if random.random() < 0.6:   # not all in cluster share the device
                normal_shared_device[acc_id] = shared_dev
            if random.random() < 0.5:   # not all share the IP
                normal_shared_ip[acc_id] = shared_ip_addr
    # ---------------------------------------------------------------

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
            # Normal: own device always
            devices = [_generate_device_id()]
            ips = [_generate_ip()]
            # Innocent sharing: inject shared device/IP for ~20% of normals
            if acc_id in normal_shared_device:
                devices.append(normal_shared_device[acc_id])
            if acc_id in normal_shared_ip:
                ips.append(normal_shared_ip[acc_id])

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

    Split: ~80% normal activity (incl. high-volume legit hubs), ~20% mule ring.
    Previously 60/40 split meant mules got ~4x transactions per-account vs normals.
    Now we flatten that ratio to ~2x by boosting normal volume and cutting mule share.
    """
    all_ids = accounts_df["account_id"].tolist()
    mule_ids = accounts_df[accounts_df["is_mule"]]["account_id"].tolist()
    normal_ids = [a for a in accounts_df[~accounts_df["is_mule"]]["account_id"].tolist()
                  if not a.startswith("ACC-HN-")]

    transactions = []
    base_time = datetime.now() - timedelta(days=30)

    # ── Normal transactions (~75%) ─────────────────────────────────
    # Use a weighted sampler so ~15% of normal accounts act as hubs
    # (merchants, payment aggregators) with higher transaction activity.
    n_normal = int(n * 0.75)
    hub_normals = random.sample(normal_ids, max(1, int(len(normal_ids) * 0.15)))
    hub_set = set(hub_normals)
    # Weight: hub accounts are 8x more likely to appear as source
    weights = [8.0 if a in hub_set else 1.0 for a in normal_ids]
    w_sum = sum(weights)
    weights = [w / w_sum for w in weights]

    for _ in range(n_normal):
        src = np.random.choice(normal_ids, p=weights)
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

    # ── Burst-settlement normal accounts ──────────────────────────
    # 25% of hub normals are designated "payment processors" / "trading desks"
    # that do rapid-fire batch settlements within 1–5 minute windows.
    # This injects genuine velocity ambiguity into the normal class so that
    # low avg_velocity is no longer a clean mule discriminator.
    n_burst_hubs = max(2, int(len(hub_normals) * 0.25))
    burst_hub_ids = random.sample(hub_normals, min(n_burst_hubs, len(hub_normals)))
    for burst_id in burst_hub_ids:
        n_bursts = random.randint(4, 10)   # 4-10 burst events over 30 days
        for _ in range(n_bursts):
            burst_start = base_time + timedelta(
                days=random.randint(0, 29),
                hours=random.randint(8, 21)
            )
            n_rapid = random.randint(3, 8)   # 3-8 rapid txns per burst
            burst_dsts = random.sample([a for a in all_ids if a != burst_id],
                                       min(n_rapid, len(all_ids) - 1))
            for j, dst_id in enumerate(burst_dsts):
                ts = burst_start + timedelta(
                    minutes=random.randint(1, 2) * (j + 1)  # 1-2 min gaps → low velocity
                )
                transactions.append({
                    "transaction_id": f"TXN-{uuid.uuid4().hex[:12].upper()}",
                    "source_id": burst_id,
                    "target_id": dst_id,
                    "amount": round(random.uniform(5000, 80000), 2),
                    "channel_type": random.choice(CHANNELS),
                    "timestamp": ts.isoformat(),
                    "geo_location": random.choice(GEO_LOCATIONS),
                    "is_suspicious": False,   # legit — payment processor behaviour
                })


    # ── Mule ring transactions (~20%) ──────────────────────────────
    # Reduced from 40% → 20% to narrow the per-account degree ratio
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

    # ── STEP 3 (enhanced): Dissolve mule ring isolation ────────────
    # The original 15% bridge rate was insufficient — mule rings still formed
    # dense isolated cliques that GNN community detection trivially identifies.
    #
    # Enhanced strategy:
    #   A) 50% of normal accounts → 1 transaction to a random mule (normal→mule)
    #   B) Each mule account → 2–5 transactions to random normals (mule→normal exits)
    # This forces approximately 40-60% of each mule's neighbors to be normal accounts.

    # A) Normal → Mule bridges (50% of normals)
    n_bridges = int(len(normal_ids) * 0.50)
    bridge_ids = random.sample(normal_ids, min(n_bridges, len(normal_ids)))
    for norm_id in bridge_ids:
        mule_target = random.choice(mule_ids)
        ts = base_time + timedelta(
            days=random.randint(0, 29),
            hours=random.randint(8, 22),
            minutes=random.randint(0, 59),
        )
        transactions.append({
            "transaction_id": f"TXN-{uuid.uuid4().hex[:12].upper()}",
            "source_id": norm_id,
            "target_id": mule_target,
            "amount": round(random.uniform(100, 8000), 2),
            "channel_type": random.choice(CHANNELS),
            "timestamp": ts.isoformat(),
            "geo_location": random.choice(GEO_LOCATIONS),
            "is_suspicious": False,  # labelled normal — creates ambiguity
        })

    # B) Mule → Normal "exit" bridges: each mule sends money to 2–5 normals
    for mule_id in mule_ids:
        n_exits = random.randint(2, 5)
        exit_targets = random.sample(normal_ids, min(n_exits, len(normal_ids)))
        for norm_target in exit_targets:
            ts = base_time + timedelta(
                days=random.randint(0, 29),
                hours=random.randint(8, 22),
                minutes=random.randint(0, 59),
            )
            transactions.append({
                "transaction_id": f"TXN-{uuid.uuid4().hex[:12].upper()}",
                "source_id": mule_id,
                "target_id": norm_target,
                "amount": round(random.uniform(200, 10000), 2),
                "channel_type": random.choice(CHANNELS),
                "timestamp": ts.isoformat(),
                "geo_location": random.choice(GEO_LOCATIONS),
                "is_suspicious": False,  # exit transactions appear normal
            })

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
        # STEP 5: Reduce dominance of ATM signal — overlap ranges slightly
        # Previously mules: 3-10, normals: 0-2. Now: 1-7 vs 0-3 (less separable)
        n_withdrawals = random.randint(1, 7) if acc_id in mule_ids else random.randint(0, 3)

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


# ─── STEP 4: Hard Negative Account Injection ───────────────────────

def inject_hard_negatives(
    accounts_df: pd.DataFrame,
    transactions_df: pd.DataFrame,
    n_hard_negatives: int = 40,  # scaled to match larger account pool
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    STEP 4 — Inject 'normal-looking fraud' accounts.

    These are LEGITIMATE accounts that mimic surface-level mule patterns:
    - Very high transaction volume (like mules)
    - Multi-channel activity
    - Not marked as mule (is_mule=False)

    Goal: force the model to learn deeper structural signals.
    """
    base_time = datetime.now() - timedelta(days=30)
    hard_neg_ids = [f"ACC-HN-{i:04d}" for i in range(n_hard_negatives)]
    hard_neg_rows = []
    new_txns = []

    for acc_id in hard_neg_ids:
        jurisdiction = random.choice(["IN", "US", "UK"])  # low-risk jurisdictions
        hard_neg_rows.append({
            "account_id": acc_id,
            "holder_name": fake.name(),
            "jurisdiction": jurisdiction,
            "jurisdiction_risk_weight": JURISDICTIONS[jurisdiction],
            "account_type": "CURRENT",
            "created_at": fake.date_between(start_date="-3y", end_date="-180d").isoformat(),
            "is_mule": False,  # Legitimate despite high volume
        })

        # Generate mule-like high-velocity transactions
        n_txns = random.randint(20, 40)
        for _ in range(n_txns):
            other = random.choice(accounts_df["account_id"].tolist())
            ts = base_time + timedelta(
                days=random.randint(0, 29),
                hours=random.randint(0, 23),
                minutes=random.randint(0, 59)
            )
            new_txns.append({
                "transaction_id": f"TXN-{uuid.uuid4().hex[:12].upper()}",
                "source_id": acc_id,
                "target_id": other,
                "amount": round(random.uniform(50000, 200000), 2),  # high-value
                "channel_type": random.choice(CHANNELS),
                "timestamp": ts.isoformat(),
                "geo_location": random.choice(GEO_LOCATIONS),
                "is_suspicious": False,
            })

    combined_accounts = pd.concat(
        [accounts_df, pd.DataFrame(hard_neg_rows)], ignore_index=True
    )
    combined_txns = pd.concat(
        [transactions_df, pd.DataFrame(new_txns)], ignore_index=True
    )
    print(f"   🧨 Hard negatives: injected {n_hard_negatives} high-volume legit accounts")
    return combined_accounts, combined_txns


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
    print(f"   → {len(transactions_df)} total transactions generated (incl. bridges)")

    # STEP 4: Inject hard negatives after transactions are generated
    print(f"🧨 Injecting hard negative samples...")
    accounts_df, transactions_df = inject_hard_negatives(accounts_df, transactions_df)

    # Device/IP mappings for hard negatives (unique per account — they're legit)
    hn_accounts = accounts_df[accounts_df["account_id"].str.startswith("ACC-HN-")]
    for _, acc in hn_accounts.iterrows():
        devices_df = pd.concat([
            devices_df,
            pd.DataFrame([{"account_id": acc["account_id"], "device_id": _generate_device_id()}])
        ], ignore_index=True)
        ips_df = pd.concat([
            ips_df,
            pd.DataFrame([{"account_id": acc["account_id"], "ip_address": _generate_ip()}])
        ], ignore_index=True)

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
