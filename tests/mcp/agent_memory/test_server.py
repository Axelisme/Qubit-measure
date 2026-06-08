"""Tests for the agent-memory MCP server wiring (tool generation + dispatch)."""

from zcu_tools.mcp.agent_memory import server
from zcu_tools.mcp.agent_memory.method_specs import METHOD_SPECS
from zcu_tools.mcp.agent_memory.store import MemoryStore

_EXPECTED = {
    "memory_recall",
    "memory_search",
    "memory_get",
    "memory_record",
    "memory_add_solution",
    "memory_update_solution",
    "memory_delete",
}


def test_dispatch_covers_every_method(tmp_path):
    dispatch = server.build_dispatch(MemoryStore(root=tmp_path, namespace="ns"))
    assert set(dispatch) == set(METHOD_SPECS)  # no spec without a handler, vice versa


def test_build_tools_exposes_seven_tools(tmp_path):
    tools = server.build_tools(MemoryStore(root=tmp_path, namespace="ns"))
    assert set(tools) == _EXPECTED
    for spec in tools.values():
        assert callable(spec["handler"])
        assert spec["inputSchema"]["type"] == "object"


def test_record_inputschema_required_fields(tmp_path):
    tools = server.build_tools(MemoryStore(root=tmp_path, namespace="ns"))
    required = set(tools["memory_record"]["inputSchema"].get("required", []))
    assert {"chip", "qub", "date", "exp_type", "outcome", "body"} <= required
    assert "data_ref" not in required and "solutions" not in required


def test_tool_handler_end_to_end(tmp_path):
    tools = server.build_tools(MemoryStore(root=tmp_path, namespace="ns"))
    created = tools["memory_record"]["handler"](
        {
            "chip": "Q1",
            "qub": "Q1",
            "date": "2026-06-08",
            "exp_type": ["t1"],
            "outcome": "success",
            "body": "t1 ok",
        }
    )
    assert created["id"] == "records/Q1/Q1/2026-06-08-t1"
    got = tools["memory_get"]["handler"]({"entry_id": created["id"]})
    assert got["body"] == "t1 ok"
    found = tools["memory_search"]["handler"]({"query": "t1"})
    assert any(r["id"] == created["id"] for r in found["results"])
