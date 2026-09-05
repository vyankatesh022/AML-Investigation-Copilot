"""Individual node execution functions for the AML investigation workflow."""

from typing import Any, Dict
from src.mcp_tools.server import FinGuardMCPServer
from src.rag.retriever import AMLPolicyRetriever
from src.workflow.llm import InvestigationLLMService
from src.workflow.state import InvestigationWorkflowState


class WorkflowNodes:
    """Encapsulates the discrete step execution functions of the investigation workflow."""

    def __init__(
        self,
        mcp_server: FinGuardMCPServer = None,
        retriever: AMLPolicyRetriever = None,
        llm_service: InvestigationLLMService = None,
    ):
        self.mcp_server = mcp_server or FinGuardMCPServer()
        self.retriever = retriever or AMLPolicyRetriever()
        self.llm_service = llm_service or InvestigationLLMService()

    def get_transaction_details(self, state: InvestigationWorkflowState) -> Dict[str, Any]:
        """Node 1: Retrieves core transaction metadata via MCP tool."""
        tx_id = state["transaction_id"]
        res = self.mcp_server.call_tool("get_transaction_details", {"transaction_id": tx_id})

        if res["status"] != "success" or not res.get("data"):
            return {
                "workflow_status": "FAILED",
                "error_message": res.get("error_message") or f"Transaction '{tx_id}' not found.",
            }

        return {
            "transaction_details": res["data"],
            "workflow_status": "IN_PROGRESS",
        }

    def get_risk_analysis(self, state: InvestigationWorkflowState) -> Dict[str, Any]:
        """Node 2: Executes dual-screening rules & ML risk evaluation via MCP tool."""
        tx_id = state["transaction_id"]
        res = self.mcp_server.call_tool("get_transaction_risk_analysis", {"transaction_id": tx_id})

        if res["status"] == "success" and res.get("data"):
            data = res["data"]
            return {
                "risk_analysis": data,
                "triggered_risk_indicators": data.get("triggered_rules", []),
            }

        return {
            "risk_analysis": {"final_risk_level": "UNKNOWN", "final_risk_score": 0.0},
            "triggered_risk_indicators": [],
        }

    def get_customer_profile(self, state: InvestigationWorkflowState) -> Dict[str, Any]:
        """Node 3: Retrieves customer KYC profile via MCP tool."""
        tx = state.get("transaction_details") or {}
        cid = tx.get("customer_id")

        if not cid:
            return {"customer_profile": None}

        res = self.mcp_server.call_tool("get_customer_profile", {"customer_id": cid})
        profile = res.get("data") if res["status"] == "success" else None
        return {"customer_profile": profile}

    def get_transaction_history(self, state: InvestigationWorkflowState) -> Dict[str, Any]:
        """Node 4: Retrieves historical customer transactions via MCP tool."""
        tx = state.get("transaction_details") or {}
        cid = tx.get("customer_id")

        if not cid:
            return {"transaction_history": [], "history_summary": {}}

        res = self.mcp_server.call_tool("get_transaction_history", {"customer_id": cid, "limit": 10})
        if res["status"] == "success" and res.get("data"):
            return {
                "transaction_history": res["data"].get("transactions", []),
                "history_summary": res["data"].get("summary_statistics", {}),
            }

        return {"transaction_history": [], "history_summary": {}}

    def retrieve_aml_policies(self, state: InvestigationWorkflowState) -> Dict[str, Any]:
        """Node 5: Queries the RAG knowledge base for applicable AML regulations."""
        tx = state.get("transaction_details") or {}
        indicators = state.get("triggered_risk_indicators", [])

        chunks = self.retriever.retrieve_for_transaction(tx=tx, risk_indicators=indicators, top_k=3)
        citations_text = self.retriever.format_citations(chunks)

        return {
            "retrieved_policies": chunks,
            "formatted_citations": citations_text,
        }

    def generate_investigation_summary(self, state: InvestigationWorkflowState) -> Dict[str, Any]:
        """Node 6: Synthesizes gathered evidence into an 8-part investigation report via LLM."""
        tx = state.get("transaction_details") or {}
        risk = state.get("risk_analysis") or {}
        customer = state.get("customer_profile") or {}
        history = state.get("transaction_history") or []
        history_summary = state.get("history_summary") or {}
        citations = state.get("formatted_citations") or "No regulatory citations available."

        summary = self.llm_service.generate_summary(
            tx=tx,
            risk_analysis=risk,
            customer=customer,
            history=history,
            history_summary=history_summary,
            citations=citations,
        )

        risk_level = risk.get("final_risk_level", "LOW")
        if risk_level == "HIGH" or "CRITICAL" in risk.get("rule_severity", ""):
            action = "ESCALATE_TO_SAR_COMMITTEE"
        elif risk_level == "MEDIUM":
            action = "REQUEST_FOR_INFORMATION"
        else:
            action = "CLOSE_AS_FALSE_POSITIVE"

        return {
            "investigation_summary": summary,
            "recommended_action": action,
            "workflow_status": "COMPLETED",
        }
