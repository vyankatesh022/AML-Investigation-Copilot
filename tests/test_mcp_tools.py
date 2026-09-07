"""Integration and unit tests for Model Context Protocol (MCP) investigation tools."""

import pytest
from src.mcp_tools.server import FinGuardMCPServer


@pytest.fixture
def mcp_server():
    return FinGuardMCPServer()


def test_server_tool_registration(mcp_server):
    """Server must register all 4 core AML investigation tools with valid JSON schemas."""
    tools = mcp_server.list_tools()
    assert len(tools) == 4
    tool_names = [t["name"] for t in tools]
    assert "get_customer_profile" in tool_names
    assert "get_transaction_history" in tool_names
    assert "get_transaction_details" in tool_names
    assert "get_transaction_risk_analysis" in tool_names

    for t in tools:
        assert "description" in t
        assert "inputSchema" in t
        assert t["inputSchema"]["type"] == "object"


def test_get_customer_profile_valid(mcp_server):
    """Valid customer ID returns full KYC profile."""
    res = mcp_server.call_tool("get_customer_profile", {"customer_id": "CUST-101"})
    assert res["status"] == "success"
    assert res["data"]["customer_id"] == "CUST-101"
    assert res["data"]["full_name"] == "Alice Johnson"
    assert res["data"]["kyc_risk_tier"] == "LOW"


def test_get_customer_profile_unknown(mcp_server):
    """Non-existent customer ID returns clear error response without crashing."""
    res = mcp_server.call_tool("get_customer_profile", {"customer_id": "CUST-UNKNOWN-999"})
    assert res["status"] == "error"
    assert "not found in KYC database" in res["error_message"]
    assert res["data"] is None


def test_get_customer_profile_missing_input(mcp_server):
    """Omitting required customer_id triggers input validation error."""
    res = mcp_server.call_tool("get_customer_profile", {})
    assert res["status"] == "error"
    assert "Invalid input" in res["error_message"]


def test_get_transaction_history_valid(mcp_server):
    """Valid customer ID returns sorted history and summary statistics."""
    res = mcp_server.call_tool("get_transaction_history", {"customer_id": "CUST-103", "limit": 3})
    assert res["status"] == "success"
    assert res["data"]["customer_id"] == "CUST-103"
    assert len(res["data"]["transactions"]) <= 3
    assert "summary_statistics" in res["data"]
    assert res["data"]["summary_statistics"]["total_volume"] > 0
    assert res["data"]["summary_statistics"]["average_amount"] > 0


def test_get_transaction_history_invalid_limit(mcp_server):
    """Non-positive limit (< 1) triggers validation error."""
    res = mcp_server.call_tool("get_transaction_history", {"customer_id": "CUST-101", "limit": 0})
    assert res["status"] == "error"
    assert "Invalid input" in res["error_message"]


def test_get_transaction_history_unknown_customer(mcp_server):
    """Unknown customer returns 0 transactions gracefully."""
    res = mcp_server.call_tool("get_transaction_history", {"customer_id": "CUST-NO-HISTORY-999"})
    assert res["status"] == "success"
    assert res["data"]["transaction_count"] == 0
    assert res["data"]["transactions"] == []


def test_get_transaction_details_valid(mcp_server):
    """Valid transaction ID returns complete transaction metadata."""
    res = mcp_server.call_tool("get_transaction_details", {"transaction_id": "TX-1003"})
    assert res["status"] == "success"
    assert res["data"]["transaction_id"] == "TX-1003"
    assert res["data"]["amount"] == 9500.00
    assert res["data"]["transaction_type"] == "TRANSFER"


def test_get_transaction_details_unknown(mcp_server):
    """Non-existent transaction ID returns error."""
    res = mcp_server.call_tool("get_transaction_details", {"transaction_id": "TX-DOES-NOT-EXIST"})
    assert res["status"] == "error"
    assert "not found" in res["error_message"]


def test_get_transaction_risk_analysis_valid(mcp_server):
    """Valid transaction ID evaluates risk through rules and ML model."""
    res = mcp_server.call_tool("get_transaction_risk_analysis", {"transaction_id": "TX-1003"})
    assert res["status"] == "success"
    assert res["data"]["transaction_id"] == "TX-1003"
    assert res["data"]["final_risk_level"] == "HIGH"
    assert res["data"]["is_flagged"] is True
    assert len(res["data"]["triggered_rules"]) > 0
    assert "ml_probability" in res["data"]
    assert "explanation" in res["data"]


def test_get_transaction_risk_analysis_unknown(mcp_server):
    """Risk analysis on missing transaction returns error."""
    res = mcp_server.call_tool("get_transaction_risk_analysis", {"transaction_id": "TX-FAKE-999"})
    assert res["status"] == "error"
    assert "not found for risk analysis" in res["error_message"]


def test_unregistered_tool_call(mcp_server):
    """Calling an unregistered tool returns clean error response."""
    res = mcp_server.call_tool("non_existent_tool", {"foo": "bar"})
    assert res["status"] == "error"
    assert "not registered on this MCP server" in res["error_message"]
