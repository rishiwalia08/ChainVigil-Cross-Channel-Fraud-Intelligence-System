import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


class FraudCaseNFTRegistry:
    """Minimal NFT-like case certificate registry (off-chain simulation)."""

    def __init__(self, file_path: str, salt: str = "chainvigil-nft-salt"):
        self.path = Path(file_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.salt = salt
        if not self.path.exists():
            self.path.write_text("[]")

    def _load(self):
        try:
            return json.loads(self.path.read_text())
        except Exception:
            return []

    def _save(self, data):
        self.path.write_text(json.dumps(data, indent=2))

    def mint_case_certificate(self, case_payload: Dict[str, Any]) -> Dict[str, Any]:
        now = datetime.now(timezone.utc).isoformat()
        raw = json.dumps(case_payload, sort_keys=True)
        token_id = hashlib.sha256(f"{self.salt}:{raw}".encode()).hexdigest()[:24]

        nft_obj = {
            "token_id": token_id,
            "minted_at": now,
            "standard": "SIMULATED_ERC721",
            "metadata": case_payload,
        }
        existing = self._load()
        existing.append(nft_obj)
        self._save(existing)
        return nft_obj
