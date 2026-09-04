"""Schemas and standardized response models for Model Context Protocol (MCP) tools."""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field


class MCPToolResponse(BaseModel):
    """Standardized response format returned by all MCP tools."""

    tool_name: str = Field(..., description="Name of the MCP tool invoked")
    status: str = Field(..., description="'success' or 'error'")
    data: Optional[Dict[str, Any]] = Field(default=None, description="Structured result data when successful")
    error_message: Optional[str] = Field(default=None, description="Clear description of error when status is 'error'")


# --- Tool Input Schemas (with validation) ---

class CustomerProfileInput(BaseModel):
    """Input parameters for the get_customer_profile tool."""

    customer_id: str = Field(..., min_length=1, description="Unique customer identifier (e.g., 'CUST-101')")


class TransactionHistoryInput(BaseModel):
    """Input parameters for the get_transaction_history tool."""

    customer_id: str = Field(..., min_length=1, description="Unique customer identifier (e.g., 'CUST-101')")
    limit: int = Field(default=10, ge=1, le=100, description="Maximum number of historical transactions to return (1-100)")


class TransactionDetailsInput(BaseModel):
    """Input parameters for the get_transaction_details tool."""

    transaction_id: str = Field(..., min_length=1, description="Unique transaction identifier (e.g., 'TX-1003')")


class TransactionRiskAnalysisInput(BaseModel):
    """Input parameters for the get_transaction_risk_analysis tool."""

    transaction_id: str = Field(..., min_length=1, description="Unique transaction identifier (e.g., 'TX-1003')")
