"""Implementation of concrete Model Context Protocol (MCP) investigation tools."""

from typing import Any, Dict, List, Optional
from pydantic import ValidationError

from src.data.loader import TransactionDataLoader, CustomerDataLoader
from src.risk_engine.assessor import TransactionRiskAssessor, CombinedRiskAssessment
from src.mcp_tools.schemas import (
    MCPToolResponse,
    CustomerProfileInput,
    TransactionHistoryInput,
    TransactionDetailsInput,
    TransactionRiskAnalysisInput,
)


class AMLInvestigationTools:
    """Core provider for MCP tools, wrapping existing data and risk engine functionality."""

    def __init__(
        self,
        tx_loader: Optional[TransactionDataLoader] = None,
        customer_loader: Optional[CustomerDataLoader] = None,
        risk_assessor: Optional[TransactionRiskAssessor] = None,
    ):
        self.tx_loader = tx_loader or TransactionDataLoader()
        self.customer_loader = customer_loader or CustomerDataLoader()
        self.risk_assessor = risk_assessor or TransactionRiskAssessor()

    def get_customer_profile(self, **kwargs) -> MCPToolResponse:
        """Retrieves verified KYC and demographic profile for a customer."""
        try:
            validated = CustomerProfileInput(**kwargs)
        except ValidationError as e:
            return MCPToolResponse(
                tool_name="get_customer_profile",
                status="error",
                error_message=f"Invalid input: {e.errors()[0]['msg']}",
            )

        cid = validated.customer_id.strip()
        profile = self.customer_loader.get_customer_by_id(cid)
        if not profile:
            return MCPToolResponse(
                tool_name="get_customer_profile",
                status="error",
                error_message=f"Customer with ID '{cid}' not found in KYC database.",
            )

        return MCPToolResponse(
            tool_name="get_customer_profile",
            status="success",
            data=profile,
        )

    def get_transaction_history(self, **kwargs) -> MCPToolResponse:
        """Retrieves recent transaction history and baseline summary statistics for a customer."""
        try:
            validated = TransactionHistoryInput(**kwargs)
        except ValidationError as e:
            return MCPToolResponse(
                tool_name="get_transaction_history",
                status="error",
                error_message=f"Invalid input: {e.errors()[0]['msg']}",
            )

        cid = validated.customer_id.strip()
        history = self.tx_loader.get_transactions_for_customer(cid, limit=validated.limit)

        if not history:
            return MCPToolResponse(
                tool_name="get_transaction_history",
                status="success",
                data={
                    "customer_id": cid,
                    "transaction_count": 0,
                    "limit_applied": validated.limit,
                    "transactions": [],
                    "summary_statistics": {
                        "total_volume": 0.0,
                        "average_amount": 0.0,
                        "unique_counterparties": 0,
                    },
                },
            )

        amounts = [float(tx["amount"]) for tx in history]
        counterparties = set(tx.get("counterparty_id") for tx in history if tx.get("counterparty_id"))
        total_vol = round(sum(amounts), 2)
        avg_amt = round(total_vol / len(amounts), 2)

        return MCPToolResponse(
            tool_name="get_transaction_history",
            status="success",
            data={
                "customer_id": cid,
                "transaction_count": len(history),
                "limit_applied": validated.limit,
                "transactions": history,
                "summary_statistics": {
                    "total_volume": total_vol,
                    "average_amount": avg_amt,
                    "unique_counterparties": len(counterparties),
                },
            },
        )

    def get_transaction_details(self, **kwargs) -> MCPToolResponse:
        """Retrieves complete metadata, amounts, and account balance snapshots for a transaction."""
        try:
            validated = TransactionDetailsInput(**kwargs)
        except ValidationError as e:
            return MCPToolResponse(
                tool_name="get_transaction_details",
                status="error",
                error_message=f"Invalid input: {e.errors()[0]['msg']}",
            )

        tx_id = validated.transaction_id.strip()
        tx = self.tx_loader.get_transaction_by_id(tx_id)
        if not tx:
            return MCPToolResponse(
                tool_name="get_transaction_details",
                status="error",
                error_message=f"Transaction with ID '{tx_id}' not found.",
            )

        return MCPToolResponse(
            tool_name="get_transaction_details",
            status="success",
            data=tx,
        )

    def get_transaction_risk_analysis(self, **kwargs) -> MCPToolResponse:
        """Evaluates a transaction through rules and ML, returning composite risk scores and indicators."""
        try:
            validated = TransactionRiskAnalysisInput(**kwargs)
        except ValidationError as e:
            return MCPToolResponse(
                tool_name="get_transaction_risk_analysis",
                status="error",
                error_message=f"Invalid input: {e.errors()[0]['msg']}",
            )

        tx_id = validated.transaction_id.strip()
        tx = self.tx_loader.get_transaction_by_id(tx_id)
        if not tx:
            return MCPToolResponse(
                tool_name="get_transaction_risk_analysis",
                status="error",
                error_message=f"Transaction with ID '{tx_id}' not found for risk analysis.",
            )

        # Retrieve customer context and recent history to inform risk evaluation
        cid = str(tx.get("customer_id", ""))
        customer = self.customer_loader.get_customer_by_id(cid) if cid else None
        history = self.tx_loader.get_transactions_for_customer(cid, limit=10) if cid else None

        # Execute dual-screening risk assessor
        assessment: CombinedRiskAssessment = self.risk_assessor.assess_transaction(
            tx=tx, customer=customer, history=history
        )

        return MCPToolResponse(
            tool_name="get_transaction_risk_analysis",
            status="success",
            data=assessment.model_dump(),
        )
