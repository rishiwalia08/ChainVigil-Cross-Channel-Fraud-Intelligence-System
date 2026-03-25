import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List


class FraudLedger:
    """Simple hash-chained audit ledger for tamper-evident fraud decisions."""

    def __init__(self, file_path: str):
        self.path = Path(file_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            self.path.write_text("")

    def _read_all(self) -> List[Dict[str, Any]]:
        raw = self.path.read_text().strip()
        if not raw:
            return []
        blocks: List[Dict[str, Any]] = []
        for line in raw.splitlines():
            line = line.strip()
            if not line:
                continue
            try:
                blocks.append(json.loads(line))
            except json.JSONDecodeError:
                continue
        return blocks

    def _last_hash(self) -> str:
        blocks = self._read_all()
        return blocks[-1]["block_hash"] if blocks else "GENESIS"

    def append(self, payload: Dict[str, Any]) -> str:
        block = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "prev_hash": self._last_hash(),
            "payload": payload,
        }
        raw = json.dumps(block, sort_keys=True)
        block_hash = hashlib.sha256(raw.encode()).hexdigest()
        block["block_hash"] = block_hash
        with self.path.open("a") as f:
            f.write(json.dumps(block) + "\n")
        return block_hash

    def verify(self) -> Dict[str, Any]:
        blocks = self._read_all()
        if not blocks:
            return {"valid": True, "blocks": 0, "message": "ledger empty"}

        expected_prev = "GENESIS"
        for i, block in enumerate(blocks):
            stored_hash = block.get("block_hash", "")
            if block.get("prev_hash") != expected_prev:
                return {
                    "valid": False,
                    "blocks": len(blocks),
                    "failed_at_index": i,
                    "reason": "prev_hash mismatch",
                }

            to_hash = {
                "ts": block.get("ts"),
                "prev_hash": block.get("prev_hash"),
                "payload": block.get("payload"),
            }
            computed = hashlib.sha256(json.dumps(to_hash, sort_keys=True).encode()).hexdigest()
            if computed != stored_hash:
                return {
                    "valid": False,
                    "blocks": len(blocks),
                    "failed_at_index": i,
                    "reason": "block_hash mismatch",
                }
            expected_prev = stored_hash

        return {"valid": True, "blocks": len(blocks), "last_hash": expected_prev}

    def recent(self, limit: int = 20) -> List[Dict[str, Any]]:
        blocks = self._read_all()
        if limit <= 0:
            return []
        return list(reversed(blocks[-limit:]))
