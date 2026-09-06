"""Pydantic request and response schemas for the FinGuard AI FastAPI layer."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class HealthResponse(BaseModel):
    """Schema for application health check endpoint."""

    status: str = Field(default="healthy", description="Application operational status")
    version: str = Field(default="0.1.0", description="API version")
    environment: str = Field(default="development", description="Current operating environment")


class TransactionResponse(BaseModel):
    """Schema for individual transaction details."""

    transaction_id: str = Field(..., description="Unique transaction identifier")
    timestamp: str = Field(..., description="ISO-8601 transaction timestamp")
    customer_id: str = Field(..., description="Originator customer ID")
    counterparty_id: str = Field(..., description="Beneficiary customer ID")
    transaction_type: str = Field(..., description="Transaction channel/type")
    amount: float = Field(..., description="Transaction monetary value")
    oldbalanceOrg: float = Field(..., description="Originator starting balance")
    newbalanceOrig: float = Field(..., description="Originator ending balance")
    oldbalanceDest: float = Field(..., description="Beneficiary starting balance")
    newbalanceDest: float = Field(..., description="Beneficiary ending balance")
    counterparty_country: str = Field(default="USA", description="Destination country")
    channel: str = Field(default="ONLINE_BANKING", description="Execution channel")


class TransactionListResponse(BaseModel):
    """Schema for paginated/limited transaction list query."""

    total_records: int = Field(..., description="Total records matching criteria")
    returned_records: int = Field(..., description="Number of records returned in response")
    transactions: List[TransactionResponse] = Field(default_factory=list)


class HighRiskTransactionItem(BaseModel):
    """Schema for a transaction identified as potentially high risk."""

    transaction_id: str
    customer_id: str
    amount: float
    transaction_type: str
    counterparty_country: str
    risk_level: str
    risk_score: float
    triggered_rules: List[str]


class RiskAnalysisResponse(BaseModel):
    """Schema for transaction dual-screening risk analysis results."""

    transaction_id: str
    final_risk_level: str
    final_risk_score: float
    is_flagged: bool
    rule_score: float
    rule_severity: str
    triggered_rules: List[str]
    ml_probability: float
    ml_risk_level: str
    top_ml_signals: List[str]
    explanation: str


class InvestigationRequest(BaseModel):
    """Schema for requesting an AI-assisted transaction investigation."""

    transaction_id: str = Field(..., min_length=1, description="Unique transaction ID to investigate (e.g. 'TX-1003')")


class InvestigationResponse(BaseModel):
    """Schema for structured AI investigation workflow output."""

    transaction_id: str
    workflow_status: str = Field(..., description="'COMPLETED' or 'FAILED'")
    final_risk_level: str = Field(default="UNKNOWN")
    final_risk_score: float = Field(default=0.0)
    is_flagged: bool = Field(default=False)
    triggered_rules: List[str] = Field(default_factory=list)
    customer_id: Optional[str] = None
    investigation_summary: Optional[str] = None
    recommended_action: str = Field(default="CLOSE_AS_FALSE_POSITIVE")
    error_message: Optional[str] = None


class ErrorResponse(BaseModel):
    """Standardized error response model."""

    detail: str = Field(..., description="Human-readable explanation of error")
