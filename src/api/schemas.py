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


class EvidenceItemResponse(BaseModel):
    """Schema for individual evidence items in investigation responses."""

    evidence_id: str = Field(..., description="Unique evidence identifier")
    evidence_type: str = Field(..., description="Category of evidence")
    source: str = Field(..., description="Originating data source or model")
    description: str = Field(..., description="Factual description of the evidence")
    related_transaction_id: str
    related_customer_id: Optional[str] = None
    supporting_data: Dict[str, Any] = Field(default_factory=dict)
    retrieval_timestamp: str
    relevance_explanation: str = ""
    availability: str = "AVAILABLE"
    confidence_or_score: Optional[float] = None
    confidence_type: Optional[str] = None


class RiskAnalysisResponse(BaseModel):
    """Schema for transaction dual-screening risk analysis results with explainability."""

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
    rule_explanations: List[Dict[str, Any]] = Field(default_factory=list, description="Structured explainability for each triggered rule")
    ml_explanation: Optional[Dict[str, Any]] = Field(default=None, description="Model transparency metadata and signal explanations")


class InvestigationRequest(BaseModel):
    """Schema for requesting an AI-assisted transaction investigation."""

    transaction_id: str = Field(..., min_length=1, description="Unique transaction ID to investigate (e.g. 'TX-1003')")


class InvestigationResponse(BaseModel):
    """Schema for structured AI investigation workflow output with evidence trace."""

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
    evidence_trace: List[EvidenceItemResponse] = Field(default_factory=list, description="Chronological audit trail of collected evidence")
    rule_explanations: List[Dict[str, Any]] = Field(default_factory=list, description="Structured rule explainability items")
    ml_explanation: Optional[Dict[str, Any]] = Field(default=None, description="ML anomaly explainability metadata")
    ai_interpretation: Optional[Dict[str, Any]] = Field(default=None, description="AI-assisted interpretation clearly separated from factual evidence")


class ErrorResponse(BaseModel):
    """Standardized error response model."""

    detail: str = Field(..., description="Human-readable explanation of error")
