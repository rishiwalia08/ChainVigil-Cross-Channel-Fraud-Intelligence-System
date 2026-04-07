"""
ChainVigil — Pydantic Schemas for API request/response models
"""
from pydantic import BaseModel, Field
from typing import List, Optional
from datetime import datetime
from enum import Enum


class ChannelType(str, Enum):
    UPI = "UPI"
    ATM = "ATM"
    WEB = "WEB"
    MOBILE_APP = "MOBILE_APP"


class RiskAction(str, Enum):
    MONITOR = "Monitor"
    FREEZE = "Freeze"
    ESCALATE = "Escalate"


# ─── Request Models ─────────────────────────────────────────────────

class GenerateDataRequest(BaseModel):
    num_accounts: int = Field(default=500, ge=50, le=5000)
    num_transactions: int = Field(default=2500, ge=100, le=50000)
    num_mule_rings: int = Field(default=5, ge=1, le=20)


class IngestRequest(BaseModel):
    data_path: Optional[str] = None  # If None, uses default generated data


class TrainRequest(BaseModel):
    epochs: int = Field(default=40, ge=10, le=1000)
    learning_rate: float = Field(default=0.005, gt=0, lt=1)


# ─── Response Models ────────────────────────────────────────────────

class AccountNode(BaseModel):
    account_id: str
    entity_type: str = "Account"
    jurisdiction_risk_weight: float
    is_mule: Optional[bool] = None


class TransactionEdge(BaseModel):
    source_id: str
    target_id: str
    timestamp: datetime
    channel_type: ChannelType
    amount: float
    geo_location: Optional[str] = None


class GraphStats(BaseModel):
    total_accounts: int
    total_transactions: int
    total_devices: int
    total_ips: int
    total_atm_terminals: int
    total_edges: int
    mule_rings_detected: int


class RiskScore(BaseModel):
    account_id: str
    mule_probability: float
    cluster_id: Optional[str] = None
    recommended_action: RiskAction
    top_features: List[str] = []


class AuditReport(BaseModel):
    account_id: str
    confidence_score: float
    top_features: List[str]
    cluster_id: Optional[str] = None
    xai_reasoning: str
    timestamp: datetime = Field(default_factory=datetime.now)


class FeatureAttribution(BaseModel):
    name: str
    importance: float
    rank: int


class DriverMeaning(BaseModel):
    feature: str
    importance: float
    meaning: str


class LLMMeta(BaseModel):
    source: str
    model: str


class AccountExplanationResponse(BaseModel):
    account_id: str
    confidence_score: float
    top_features: List[str]
    feature_attributions: List[FeatureAttribution]
    feature_values: dict
    xai_reasoning: str
    plain_english_summary: str = ""
    llm_meta: Optional[LLMMeta] = None
    key_driver_meanings: List[DriverMeaning] = Field(default_factory=list)
    suggested_actions: List[str] = Field(default_factory=list)


class PipelineStatus(BaseModel):
    stage: str
    status: str
    message: str
    details: Optional[dict] = None
