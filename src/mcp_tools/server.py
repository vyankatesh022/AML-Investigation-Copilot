"""Model Context Protocol (MCP) server implementation for FinGuard AI."""

import json
import sys
from typing import Any, Callable, Dict, List, Optional

from src.mcp_tools.schemas import (
    CustomerProfileInput,
    TransactionHistoryInput,
    TransactionDetailsInput,
    TransactionRiskAnalysisInput,
)
from src.mcp_tools.tools import AMLInvestigationTools


class FinGuardMCPServer:
    """Manages MCP tool registration, schema publication, and tool execution."""

    def __init__(self, tools_provider: Optional[AMLInvestigationTools] = None):
        self.provider = tools_provider or AMLInvestigationTools()
        self._registry: Dict[str, Dict[str, Any]] = {}
        self._register_default_tools()

    def _register_default_tools(self) -> None:
        """Registers the 4 primary AML investigation tools."""
        self.register_tool(
            name="get_customer_profile",
            description=(
                "Retrieves verified KYC and demographic profile for an account holder, "
                "including KYC risk tier, occupation, expected turnover, and PEP status."
            ),
            input_schema=CustomerProfileInput.model_json_schema(),
            handler=self.provider.get_customer_profile,
        )

        self.register_tool(
            name="get_transaction_history",
            description=(
                "Retrieves a customer's recent past transaction records and baseline statistics "
                "(total volume, average ticket size, frequent counterparties)."
            ),
            input_schema=TransactionHistoryInput.model_json_schema(),
            handler=self.provider.get_transaction_history,
        )

        self.register_tool(
            name="get_transaction_details",
            description=(
                "Retrieves complete transaction metadata, timestamp, execution channel, "
                "amounts, and originator/beneficiary balance snapshots."
            ),
            input_schema=TransactionDetailsInput.model_json_schema(),
            handler=self.provider.get_transaction_details,
        )

        self.register_tool(
            name="get_transaction_risk_analysis",
            description=(
                "Runs dual-screening (AML compliance rules and Scikit-Learn ML classifier) "
                "to evaluate transaction anomaly risk, returning scores and triggered red flags."
            ),
            input_schema=TransactionRiskAnalysisInput.model_json_schema(),
            handler=self.provider.get_transaction_risk_analysis,
        )

    def register_tool(
        self,
        name: str,
        description: str,
        input_schema: Dict[str, Any],
        handler: Callable[..., Any],
    ) -> None:
        """Registers a tool with its metadata, schema, and execution handler."""
        self._registry[name] = {
            "name": name,
            "description": description,
            "inputSchema": input_schema,
            "handler": handler,
        }

    def list_tools(self) -> List[Dict[str, Any]]:
        """Returns metadata and JSON schemas for all registered MCP tools."""
        return [
            {
                "name": meta["name"],
                "description": meta["description"],
                "inputSchema": meta["inputSchema"],
            }
            for meta in self._registry.values()
        ]

    def call_tool(self, name: str, arguments: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Dispatches a tool call with the provided arguments and returns structured response."""
        clean_name = str(name).strip()
        tool_meta = self._registry.get(clean_name)
        args = arguments or {}

        if not tool_meta:
            return {
                "tool_name": clean_name,
                "status": "error",
                "data": None,
                "error_message": f"Tool '{clean_name}' is not registered on this MCP server.",
            }

        try:
            handler = tool_meta["handler"]
            result = handler(**args)
            return result.model_dump()
        except Exception as exc:
            return {
                "tool_name": clean_name,
                "status": "error",
                "data": None,
                "error_message": f"Unexpected execution failure: {str(exc)}",
            }


def run_stdio_server():
    """Simple JSON-RPC stdio runner for local MCP invocation."""
    server = FinGuardMCPServer()
    print("FinGuard AI MCP Server active. Awaiting JSON tool calls on stdin...", file=sys.stderr)

    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            request = json.loads(line)
            tool_name = request.get("name")
            args = request.get("arguments", {})
            response = server.call_tool(name=tool_name, arguments=args)
            print(json.dumps(response))
            sys.stdout.flush()
        except Exception as e:
            print(json.dumps({"status": "error", "error_message": str(e)}))
            sys.stdout.flush()


if __name__ == "__main__":
    run_stdio_server()
