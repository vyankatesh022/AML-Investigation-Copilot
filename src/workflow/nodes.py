"""Individual node execution functions for the AML investigation workflow with evidence traceability."""

from typing import Any, Dict
from src.evidence.collector import EvidenceCollector
from src.evidence.models import EvidenceItem
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
        """Node 1: Retrieves core transaction metadata via MCP tool and records transaction evidence."""
        tx_id = state["transaction_id"]
        res = self.mcp_server.call_tool("get_transaction_details", {"transaction_id": tx_id})

        if res["status"] != "success" or not res.get("data"):
            return {
                "workflow_status": "FAILED",
                "error_message": res.get("error_message") or f"Transaction '{tx_id}' not found.",
            }

        collector = EvidenceCollector()
        collector.add_transaction_evidence(res["data"])

        return {
            "transaction_details": res["data"],
            "workflow_status": "IN_PROGRESS",
            "evidence_items": collector.to_dict_list(),
        }

    def get_risk_analysis(self, state: InvestigationWorkflowState) -> Dict[str, Any]:
        """Node 2: Executes dual-screening rules & ML risk evaluation via MCP tool and records rule/ML evidence."""
        tx_id = state["transaction_id"]
        tx = state.get("transaction_details") or {}
        cid = tx.get("customer_id")
        res = self.mcp_server.call_tool("get_transaction_risk_analysis", {"transaction_id": tx_id})

        prior_items = [EvidenceItem(**i) for i in state.get("evidence_items", [])]
        collector = EvidenceCollector(prior_items)

        if res["status"] == "success" and res.get("data"):
            data = res["data"]
            rule_explanations = data.get("rule_explanations", [])
            for idx, r_exp in enumerate(rule_explanations, start=1):
                collector.add_rule_evidence(tx_id=tx_id, rule_exp=r_exp, index=idx, customer_id=cid)

            ml_exp = data.get("ml_explanation")
            if ml_exp:
                collector.add_ml_evidence(tx_id=tx_id, ml_exp=ml_exp, customer_id=cid)

            return {
                "risk_analysis": data,
                "triggered_risk_indicators": data.get("triggered_rules", []),
                "rule_explanations": rule_explanations,
                "ml_explanation": ml_exp,
                "evidence_items": collector.to_dict_list(),
            }

        return {
            "risk_analysis": {"final_risk_level": "UNKNOWN", "final_risk_score": 0.0},
            "triggered_risk_indicators": [],
            "rule_explanations": [],
            "ml_explanation": None,
            "evidence_items": collector.to_dict_list(),
        }

    def get_customer_profile(self, state: InvestigationWorkflowState) -> Dict[str, Any]:
        """Node 3: Retrieves customer KYC profile via MCP tool and records KYC evidence."""
        tx = state.get("transaction_details") or {}
        tx_id = state.get("transaction_id", "UNKNOWN")
        cid = tx.get("customer_id")

        prior_items = [EvidenceItem(**i) for i in state.get("evidence_items", [])]
        collector = EvidenceCollector(prior_items)

        if not cid:
            collector.add_customer_evidence(tx_id=tx_id, customer=None, customer_id="UNKNOWN")
            return {"customer_profile": None, "evidence_items": collector.to_dict_list()}

        res = self.mcp_server.call_tool("get_customer_profile", {"customer_id": cid})
        profile = res.get("data") if res["status"] == "success" else None
        collector.add_customer_evidence(tx_id=tx_id, customer=profile, customer_id=cid)

        return {"customer_profile": profile, "evidence_items": collector.to_dict_list()}

    def get_transaction_history(self, state: InvestigationWorkflowState) -> Dict[str, Any]:
        """Node 4: Retrieves historical customer transactions via MCP tool and records history evidence."""
        tx = state.get("transaction_details") or {}
        tx_id = state.get("transaction_id", "UNKNOWN")
        cid = tx.get("customer_id")

        prior_items = [EvidenceItem(**i) for i in state.get("evidence_items", [])]
        collector = EvidenceCollector(prior_items)

        if not cid:
            collector.add_history_evidence(tx_id=tx_id, customer_id="UNKNOWN", history=[], history_summary={})
            return {"transaction_history": [], "history_summary": {}, "evidence_items": collector.to_dict_list()}

        res = self.mcp_server.call_tool("get_transaction_history", {"customer_id": cid, "limit": 10})
        if res["status"] == "success" and res.get("data"):
            tx_history = res["data"].get("transactions", [])
            hist_summary = res["data"].get("summary_statistics", {})
            collector.add_history_evidence(tx_id=tx_id, customer_id=cid, history=tx_history, history_summary=hist_summary)
            return {
                "transaction_history": tx_history,
                "history_summary": hist_summary,
                "evidence_items": collector.to_dict_list(),
            }

        collector.add_history_evidence(tx_id=tx_id, customer_id=cid, history=[], history_summary={})
        return {"transaction_history": [], "history_summary": {}, "evidence_items": collector.to_dict_list()}

    def retrieve_aml_policies(self, state: InvestigationWorkflowState) -> Dict[str, Any]:
        """Node 5: Queries the RAG knowledge base for applicable AML regulations and records policy evidence."""
        tx = state.get("transaction_details") or {}
        tx_id = state.get("transaction_id", "UNKNOWN")
        cid = tx.get("customer_id")
        indicators = state.get("triggered_risk_indicators", [])

        chunks = self.retriever.retrieve_for_transaction(tx=tx, risk_indicators=indicators, top_k=3)
        citations_text = self.retriever.format_citations(chunks)

        prior_items = [EvidenceItem(**i) for i in state.get("evidence_items", [])]
        collector = EvidenceCollector(prior_items)
        for idx, chunk in enumerate(chunks, start=1):
            collector.add_policy_evidence(tx_id=tx_id, chunk=chunk, index=idx, customer_id=cid)

        return {
            "retrieved_policies": chunks,
            "formatted_citations": citations_text,
            "evidence_items": collector.to_dict_list(),
        }

    def generate_investigation_summary(self, state: InvestigationWorkflowState) -> Dict[str, Any]:
        """Node 6: Synthesizes gathered evidence into an 8-part investigation report via LLM."""
        tx = state.get("transaction_details") or {}
        risk = state.get("risk_analysis") or {}
        customer = state.get("customer_profile") or {}
        history = state.get("transaction_history") or []
        history_summary = state.get("history_summary") or {}
        citations = state.get("formatted_citations") or "No regulatory citations available."
        evidence_items = state.get("evidence_items", [])

        summary = self.llm_service.generate_summary(
            tx=tx,
            risk_analysis=risk,
            customer=customer,
            history=history,
            history_summary=history_summary,
            citations=citations,
            evidence_items=evidence_items,
        )

        risk_level = risk.get("final_risk_level", "LOW")
        if risk_level == "HIGH" or "CRITICAL" in risk.get("rule_severity", ""):
            action = "ESCALATE_TO_SAR_COMMITTEE"
        elif risk_level == "MEDIUM":
            action = "REQUEST_FOR_INFORMATION"
        else:
            action = "CLOSE_AS_FALSE_POSITIVE"

        ai_interpretation = {
            "assessment_type": "AI-Assisted Preliminary Investigative Assessment",
            "evaluated_risk_level": risk_level,
            "recommended_action": action,
            "is_autonomous_decision": False,
            "disclaimer": (
                "This assessment is preliminary AI-assisted guidance for human compliance officers. "
                "It does NOT constitute a final AML, legal, or compliance determination."
            ),
            "evidence_count": len(evidence_items),
            "referenced_evidence_ids": [item.get("evidence_id") for item in evidence_items],
        }

        return {
            "investigation_summary": summary,
            "recommended_action": action,
            "workflow_status": "COMPLETED",
            "ai_interpretation": ai_interpretation,
        }
