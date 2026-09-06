"""FastAPI application exposing FinGuard AI AML transaction monitoring and investigation endpoints."""

from typing import List, Optional
from fastapi import FastAPI, HTTPException, Query, status
from fastapi.middleware.cors import CORSMiddleware

from src.config import settings
from src.data.loader import TransactionDataLoader, CustomerDataLoader
from src.risk_engine.assessor import TransactionRiskAssessor
from src.workflow.graph import InvestigationWorkflow
from src.api.schemas import (
    HealthResponse,
    TransactionResponse,
    TransactionListResponse,
    HighRiskTransactionItem,
    RiskAnalysisResponse,
    InvestigationRequest,
    InvestigationResponse,
    ErrorResponse,
)

app = FastAPI(
    title="FinGuard AI — AML Transaction Monitoring Assistant",
    description="REST API providing AML transaction data access, risk assessment, and AI investigation workflows.",
    version="0.1.0",
)

# Enable CORS for frontend and external clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize service instances
tx_loader = TransactionDataLoader()
customer_loader = CustomerDataLoader()
risk_assessor = TransactionRiskAssessor()
investigation_workflow = InvestigationWorkflow()


@app.get(
    "/api/health",
    response_model=HealthResponse,
    tags=["System"],
    summary="Health check endpoint",
)
def health_check() -> HealthResponse:
    """Returns application health and operational status."""
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        environment=settings.ENVIRONMENT,
    )


@app.get(
    "/api/transactions",
    response_model=TransactionListResponse,
    tags=["Transactions"],
    summary="List transactions with optional risk filter and pagination limit",
)
def get_transactions(
    limit: int = Query(default=20, ge=1, le=500, description="Maximum number of transactions to return"),
    risk_level: Optional[str] = Query(
        default=None,
        description="Filter by risk rating ('LOW', 'MEDIUM', or 'HIGH')",
    ),
) -> TransactionListResponse:
    """Retrieves a list of transactions with optional risk tier filtering."""
    all_txs = tx_loader.get_all_transactions()

    if risk_level:
        target_level = risk_level.strip().upper()
        if target_level not in {"LOW", "MEDIUM", "HIGH"}:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid risk_level '{risk_level}'. Supported levels: 'LOW', 'MEDIUM', 'HIGH'.",
            )

        filtered_txs = []
        for tx in all_txs:
            cid = str(tx.get("customer_id", ""))
            cust = customer_loader.get_customer_by_id(cid) if cid else None
            hist = tx_loader.get_transactions_for_customer(cid, limit=10) if cid else None
            assessment = risk_assessor.assess_transaction(tx=tx, customer=cust, history=hist)
            if assessment.final_risk_level.upper() == target_level:
                filtered_txs.append(tx)

        sliced = filtered_txs[:limit]
        return TransactionListResponse(
            total_records=len(filtered_txs),
            returned_records=len(sliced),
            transactions=[TransactionResponse(**item) for item in sliced],
        )

    sliced = all_txs[:limit]
    return TransactionListResponse(
        total_records=len(all_txs),
        returned_records=len(sliced),
        transactions=[TransactionResponse(**item) for item in sliced],
    )


@app.get(
    "/api/transactions/high-risk",
    response_model=List[HighRiskTransactionItem],
    tags=["Transactions"],
    summary="List transactions flagged as high or elevated risk",
)
def get_high_risk_transactions(
    limit: int = Query(default=20, ge=1, le=100, description="Maximum number of high-risk transactions to return"),
) -> List[HighRiskTransactionItem]:
    """Evaluates transactions and returns records identified as high or medium risk, sorted by risk score."""
    all_txs = tx_loader.get_all_transactions()
    high_risk_items: List[HighRiskTransactionItem] = []

    for tx in all_txs:
        cid = str(tx.get("customer_id", ""))
        cust = customer_loader.get_customer_by_id(cid) if cid else None
        hist = tx_loader.get_transactions_for_customer(cid, limit=10) if cid else None
        assessment = risk_assessor.assess_transaction(tx=tx, customer=cust, history=hist)

        if assessment.is_flagged or assessment.final_risk_level in ["MEDIUM", "HIGH"]:
            high_risk_items.append(
                HighRiskTransactionItem(
                    transaction_id=str(tx["transaction_id"]),
                    customer_id=str(tx["customer_id"]),
                    amount=float(tx["amount"]),
                    transaction_type=str(tx["transaction_type"]),
                    counterparty_country=str(tx.get("counterparty_country", "USA")),
                    risk_level=assessment.final_risk_level,
                    risk_score=assessment.final_risk_score,
                    triggered_rules=assessment.triggered_rules,
                )
            )

    high_risk_items.sort(key=lambda item: item.risk_score, reverse=True)
    return high_risk_items[:limit]


@app.get(
    "/api/transactions/{transaction_id}",
    response_model=TransactionResponse,
    responses={404: {"model": ErrorResponse}},
    tags=["Transactions"],
    summary="Get single transaction details",
)
def get_transaction_by_id(transaction_id: str) -> TransactionResponse:
    """Retrieves detailed record for a specific transaction ID."""
    clean_id = transaction_id.strip()
    tx = tx_loader.get_transaction_by_id(clean_id)
    if not tx:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Transaction with ID '{clean_id}' not found.",
        )
    return TransactionResponse(**tx)


@app.get(
    "/api/transactions/{transaction_id}/risk",
    response_model=RiskAnalysisResponse,
    responses={404: {"model": ErrorResponse}},
    tags=["Risk Assessment"],
    summary="Evaluate transaction risk with rules and ML",
)
def get_transaction_risk(transaction_id: str) -> RiskAnalysisResponse:
    """Evaluates transaction risk using AML rules and ML classifier."""
    clean_id = transaction_id.strip()
    tx = tx_loader.get_transaction_by_id(clean_id)
    if not tx:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Transaction with ID '{clean_id}' not found for risk evaluation.",
        )

    cid = str(tx.get("customer_id", ""))
    cust = customer_loader.get_customer_by_id(cid) if cid else None
    hist = tx_loader.get_transactions_for_customer(cid, limit=10) if cid else None

    assessment = risk_assessor.assess_transaction(tx=tx, customer=cust, history=hist)
    return RiskAnalysisResponse(**assessment.model_dump())


@app.post(
    "/api/investigate",
    response_model=InvestigationResponse,
    responses={404: {"model": ErrorResponse}},
    tags=["Investigations"],
    summary="Execute end-to-end AI investigation workflow for a transaction",
)
def run_investigation(request: InvestigationRequest) -> InvestigationResponse:
    """Runs the LangGraph multi-step investigation workflow on the requested transaction."""
    clean_id = request.transaction_id.strip()
    tx = tx_loader.get_transaction_by_id(clean_id)
    if not tx:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Transaction with ID '{clean_id}' not found for investigation.",
        )

    result = investigation_workflow.investigate(clean_id)
    return InvestigationResponse(**result.model_dump())
