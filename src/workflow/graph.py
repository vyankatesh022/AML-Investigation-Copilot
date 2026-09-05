"""Linear state graph orchestrating the end-to-end AML investigation workflow."""

from typing import Optional
from src.workflow.state import InvestigationWorkflowState, InvestigationResult
from src.workflow.nodes import WorkflowNodes


class InvestigationWorkflow:
    """State graph coordinating the investigation of a financial transaction."""

    def __init__(self, nodes: Optional[WorkflowNodes] = None):
        self.nodes = nodes or WorkflowNodes()

    def run(self, transaction_id: str) -> InvestigationWorkflowState:
        """Executes the multi-step investigation workflow for a given transaction ID."""
        clean_tx_id = str(transaction_id).strip()

        # Initial State
        state: InvestigationWorkflowState = {
            "transaction_id": clean_tx_id,
            "transaction_details": None,
            "risk_analysis": None,
            "customer_profile": None,
            "transaction_history": [],
            "history_summary": {},
            "triggered_risk_indicators": [],
            "retrieved_policies": [],
            "formatted_citations": "",
            "investigation_summary": None,
            "recommended_action": "CLOSE_AS_FALSE_POSITIVE",
            "workflow_status": "IN_PROGRESS",
            "error_message": None,
        }

        # Step 1: Get Transaction Details
        step1_update = self.nodes.get_transaction_details(state)
        state.update(step1_update)

        # Conditional Edge: If transaction not found, stop workflow safely
        if state.get("workflow_status") == "FAILED":
            return state

        # Step 2: Get Risk Analysis (Rules + ML)
        step2_update = self.nodes.get_risk_analysis(state)
        state.update(step2_update)

        # Step 3: Get Customer Profile (KYC)
        step3_update = self.nodes.get_customer_profile(state)
        state.update(step3_update)

        # Step 4: Get Transaction History
        step4_update = self.nodes.get_transaction_history(state)
        state.update(step4_update)

        # Step 5: Retrieve Relevant AML Policies (RAG)
        step5_update = self.nodes.retrieve_aml_policies(state)
        state.update(step5_update)

        # Step 6: Generate Investigation Summary (LLM)
        step6_update = self.nodes.generate_investigation_summary(state)
        state.update(step6_update)

        return state

    def investigate(self, transaction_id: str) -> InvestigationResult:
        """Convenience method that runs the workflow and returns a validated Pydantic model."""
        state = self.run(transaction_id=transaction_id)
        risk = state.get("risk_analysis") or {}
        tx = state.get("transaction_details") or {}

        return InvestigationResult(
            transaction_id=state["transaction_id"],
            workflow_status=state.get("workflow_status", "FAILED"),
            final_risk_level=risk.get("final_risk_level", "UNKNOWN"),
            final_risk_score=float(risk.get("final_risk_score", 0.0)),
            is_flagged=bool(risk.get("is_flagged", False)),
            triggered_rules=risk.get("triggered_rules", []),
            customer_id=tx.get("customer_id"),
            investigation_summary=state.get("investigation_summary"),
            recommended_action=state.get("recommended_action", "CLOSE_AS_FALSE_POSITIVE"),
            error_message=state.get("error_message"),
        )
