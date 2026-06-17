"""Tests for the agent-memory MCP server wiring (tool generation + dispatch),
including the J.ARRAY round-trip that must not char-split a multi-element list."""

from zcu_tools.mcp.agent_memory import server
from zcu_tools.mcp.agent_memory.method_specs import METHOD_SPECS
from zcu_tools.mcp.agent_memory.store import MemoryStore

_EXPECTED = {
    "memory_recall",
    "memory_search",
    "memory_get",
    "memory_checklist_get",
    "memory_record_measurement",
    "memory_checklist_set",
    "memory_add_solution",
    "memory_update_solution",
    "memory_delete",
}


def _tools(tmp_path):
    return server.build_tools(MemoryStore(root=tmp_path, namespace="ns"))


def test_dispatch_covers_every_method(tmp_path):
    dispatch = server.build_dispatch(MemoryStore(root=tmp_path, namespace="ns"))
    assert set(dispatch) == set(METHOD_SPECS)  # no spec without a handler, vice versa


def test_build_tools_exposes_expected_tools(tmp_path):
    tools = _tools(tmp_path)
    assert set(tools) == _EXPECTED
    for spec in tools.values():
        assert callable(spec["handler"])
        assert spec["inputSchema"]["type"] == "object"


def test_record_inputschema_required_fields(tmp_path):
    tools = _tools(tmp_path)
    schema = tools["memory_record_measurement"]["inputSchema"]
    required = set(schema.get("required", []))
    assert {"chip", "qub", "date", "exp_type", "decision", "reason", "body"} <= required
    assert "figure_paths" not in required and "data_ref" not in required


def test_array_params_emit_typed_array_schema(tmp_path):
    # The J.ARRAY fix: array params must render as {"type":"array",...}, never an
    # untyped JSON schema the client could stringify (the old char-split bug).
    tools = _tools(tmp_path)
    rec_props = tools["memory_record_measurement"]["inputSchema"]["properties"]
    assert rec_props["exp_type"]["type"] == "array"
    assert rec_props["figure_paths"]["type"] == "array"
    clist_props = tools["memory_checklist_set"]["inputSchema"]["properties"]
    assert clist_props["items"]["type"] == "array"


def test_record_handler_end_to_end(tmp_path):
    tools = _tools(tmp_path)
    created = tools["memory_record_measurement"]["handler"](
        {
            "chip": "Q1",
            "qub": "Q1",
            "date": "2026-06-08",
            "exp_type": ["t1"],
            "decision": "accept",
            "reason": "clean decay",
            "body": "t1 = 80us",
        }
    )
    assert created["id"] == "records/Q1/Q1/2026-06-08-t1"
    got = tools["memory_get"]["handler"]({"entry_id": created["id"]})
    assert "t1 = 80us" in got["body"]
    found = tools["memory_search"]["handler"]({"query": "t1"})
    assert any(r["id"] == created["id"] for r in found["results"])


def test_array_param_multi_element_not_char_split(tmp_path):
    # A multi-element exp_type list must round-trip as a list of whole strings,
    # never as a list of single characters.
    tools = _tools(tmp_path)
    created = tools["memory_record_measurement"]["handler"](
        {
            "chip": "Q1",
            "qub": "Q1",
            "date": "2026-06-08",
            "exp_type": ["onetone/freq", "twotone/freq"],
            "decision": "accept",
            "reason": "r",
            "body": "b",
        }
    )
    got = tools["memory_get"]["handler"]({"entry_id": created["id"]})
    assert got["exp_type"] == ["onetone/freq", "twotone/freq"]


def test_checklist_set_get_via_tools(tmp_path):
    tools = _tools(tmp_path)
    tools["memory_checklist_set"]["handler"](
        {"exp_type": "onetone/freq", "items": ["dip clean", "window ok", "snr ok"]}
    )
    got = tools["memory_checklist_get"]["handler"]({"exp_type": "onetone/freq"})
    assert got["items"] == ["dip clean", "window ok", "snr ok"]


def test_recall_three_buckets_via_tools(tmp_path):
    tools = _tools(tmp_path)
    tools["memory_checklist_set"]["handler"](
        {"exp_type": "t1", "items": ["decay visible"]}
    )
    tools["memory_record_measurement"]["handler"](
        {
            "chip": "Q1",
            "qub": "Q1",
            "date": "2026-06-08",
            "exp_type": ["t1"],
            "decision": "accept",
            "reason": "r",
            "body": "b",
        }
    )
    tools["memory_add_solution"]["handler"](
        {
            "exp_type": "t1",
            "symptom": "noisy decay",
            "category": "analysis-heuristic",
            "body": "average more",
        }
    )
    out = tools["memory_recall"]["handler"]({"chip": "Q1", "qub": "Q1", "exp_type": "t1"})
    assert out["checklist"] == ["decay visible"]
    assert len(out["gotchas"]) == 1
    assert len(out["recent"]) == 1
