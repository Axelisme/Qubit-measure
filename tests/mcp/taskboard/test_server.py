"""Tests for the taskboard MCP server wiring (tool generation + dispatch smoke)."""

from __future__ import annotations

from zcu_tools.mcp.taskboard import server
from zcu_tools.mcp.taskboard.method_specs import METHOD_SPECS
from zcu_tools.mcp.taskboard.store import TaskboardStore

_EXPECTED_TOOLS = {
    "taskboard_claim",
    "taskboard_release",
    "taskboard_check",
    "taskboard_list",
    "taskboard_wait",
    "taskboard_touch",
    "taskboard_force_release",
}


def test_dispatch_covers_every_method_spec(tmp_path):
    store = TaskboardStore(
        json_path=tmp_path / "taskboard.json",
        md_path=tmp_path / "taskboard.md",
    )
    dispatch = server.build_dispatch(store)
    assert set(dispatch) == set(METHOD_SPECS)


def test_build_tools_exposes_seven_tools(tmp_path):
    store = TaskboardStore(
        json_path=tmp_path / "taskboard.json",
        md_path=tmp_path / "taskboard.md",
    )
    tools = server.build_tools(store)
    assert set(tools) == _EXPECTED_TOOLS
    for spec in tools.values():
        assert callable(spec["handler"])
        assert spec["inputSchema"]["type"] == "object"


def test_claim_tool_end_to_end(tmp_path):
    """Full round-trip via MCP tool handlers: claim → check → list → release."""
    store = TaskboardStore(
        json_path=tmp_path / "taskboard.json",
        md_path=tmp_path / "taskboard.md",
    )
    tools = server.build_tools(store)

    # claim
    result = tools["taskboard_claim"]["handler"](
        {"owner": "test-agent", "paths": ["lib/foo"], "task": "smoke test"}
    )
    assert result["status"] == "granted"
    claim_id = result["claim_id"]

    # check — should report conflict
    chk = tools["taskboard_check"]["handler"]({"paths": ["lib/foo"]})
    assert len(chk["conflicts"]) == 1

    # list — should appear in active
    lst = tools["taskboard_list"]["handler"]({})
    assert any(c["claim_id"] == claim_id for c in lst["active"])

    # wait on granted returns immediately
    w = tools["taskboard_wait"]["handler"]({"claim_id": claim_id, "timeout_s": 1.0})
    assert w["status"] == "granted"

    # release
    rel = tools["taskboard_release"]["handler"]({"claim_id": claim_id})
    assert rel["released_id"] == claim_id

    # check after release — no conflicts
    chk2 = tools["taskboard_check"]["handler"]({"paths": ["lib/foo"]})
    assert chk2["conflicts"] == []


def test_touch_tool(tmp_path):
    store = TaskboardStore(
        json_path=tmp_path / "taskboard.json",
        md_path=tmp_path / "taskboard.md",
    )
    tools = server.build_tools(store)
    r = tools["taskboard_claim"]["handler"](
        {"owner": "a", "paths": ["@gui/measure"], "task": "hold resource"}
    )
    t = tools["taskboard_touch"]["handler"]({"claim_id": r["claim_id"]})
    assert t["claim_id"] == r["claim_id"]


def test_force_release_tool(tmp_path):
    store = TaskboardStore(
        json_path=tmp_path / "taskboard.json",
        md_path=tmp_path / "taskboard.md",
    )
    tools = server.build_tools(store)
    r = tools["taskboard_claim"]["handler"](
        {"owner": "a", "paths": ["lib/x"], "task": "manual release test"}
    )
    fr = tools["taskboard_force_release"]["handler"]({"claim_id": r["claim_id"]})
    assert fr["released_id"] == r["claim_id"]
