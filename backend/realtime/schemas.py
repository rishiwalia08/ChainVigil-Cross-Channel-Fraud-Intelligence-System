from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class TransactionCheckRequest(BaseModel):
    transaction_id: str = Field(..., min_length=4)
    source_id: str
    target_id: str
    amount: float = Field(..., gt=0)
    channel_type: str = Field(default="UPI")
    timestamp: Optional[str] = None
    geo_location: Optional[str] = None
    device_id: Optional[str] = None
    ip_address: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)


class TransactionDecisionResponse(BaseModel):
    transaction_id: str
    gnn_score: float
    rule_score: float
    intel_score: float
    hybrid_score: float
    decision: str
    reasons: List[str]
    ledger_tx_id: str
    processed_at: str


class StreamSimulationRequest(BaseModel):
    num_transactions: int = Field(default=25, ge=1, le=2000)
    interval_ms: int = Field(default=0, ge=0, le=5000)
