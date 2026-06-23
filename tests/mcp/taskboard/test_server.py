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


def test_claim_tool_end_to_end(tmp_path, monkeypatch):
    """Full round-trip via MCP tool handlers: claim → check → list → release.

    Identity is derived from host session env; delete those variables so this test
    exercises the deterministic owner-fallback path (check has no owner → reports
    the alien grant as a conflict) regardless of where it runs.
    """
    monkeypatch.delenv("CLAUDE_CODE_SESSION_ID", raising=False)
    monkeypatch.delenv("CODEX_THREAD_ID", raising=False)
    monkeypatch.delenv("AGENT_SESSION_ID", raising=False)
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


# ---------------------------------------------------------------------------
# Env-derived identity (CLAUDE_CODE_SESSION_ID) — server glue
# ---------------------------------------------------------------------------


def _build_tools_for_session(
    json_path, session_id, monkeypatch, env_name="CLAUDE_CODE_SESSION_ID"
):
    """Build a fresh tool table whose dispatch snapshots ``session_id`` from env,
    pointing at the shared ``json_path`` (one server process == one session)."""
    monkeypatch.delenv("CLAUDE_CODE_SESSION_ID", raising=False)
    monkeypatch.delenv("CODEX_THREAD_ID", raising=False)
    monkeypatch.delenv("AGENT_SESSION_ID", raising=False)
    monkeypatch.setenv(env_name, session_id)
    store = TaskboardStore(
        json_path=json_path,
        md_path=json_path.parent / "taskboard.md",
    )
    return server.build_tools(store)


def test_same_session_overlapping_claims_both_granted(tmp_path, monkeypatch):
    """Two different owners under the same agent session never block each other —
    identity is derived from the env, overriding the per-call owner label."""
    json_path = tmp_path / "taskboard.json"
    tools = _build_tools_for_session(json_path, "session-A", monkeypatch)

    r1 = tools["taskboard_claim"]["handler"](
        {"owner": "orchestrator", "paths": ["lib/foo"], "task": "A"}
    )
    r2 = tools["taskboard_claim"]["handler"](
        {"owner": "sub-agent", "paths": ["lib/foo"], "task": "B"}
    )
    assert r1["status"] == "granted"
    assert r2["status"] == "granted"
    assert r2["claim_id"] == r1["claim_id"]
    assert r2["conflicts"] == []
    assert r2["warnings"][0]["kind"] == "same_session_overlap"


def test_cross_session_overlapping_claims_pending(tmp_path, monkeypatch):
    """A different agent session (a separate server process) still contends on
    overlapping write paths — second claim is queued."""
    json_path = tmp_path / "taskboard.json"

    tools_a = _build_tools_for_session(json_path, "session-A", monkeypatch)
    r1 = tools_a["taskboard_claim"]["handler"](
        {"owner": "alice", "paths": ["lib/foo"], "task": "A"}
    )
    assert r1["status"] == "granted"

    tools_b = _build_tools_for_session(json_path, "session-B", monkeypatch)
    r2 = tools_b["taskboard_claim"]["handler"](
        {"owner": "bob", "paths": ["lib/foo"], "task": "B"}
    )
    assert r2["status"] == "pending"


def test_codex_thread_id_is_session_identity(tmp_path, monkeypatch):
    """Codex passes CODEX_THREAD_ID rather than CLAUDE_CODE_SESSION_ID."""
    json_path = tmp_path / "taskboard.json"
    tools = _build_tools_for_session(
        json_path, "codex-thread-A", monkeypatch, env_name="CODEX_THREAD_ID"
    )

    r1 = tools["taskboard_claim"]["handler"](
        {"owner": "parent", "paths": ["lib/foo"], "task": "A"}
    )
    r2 = tools["taskboard_claim"]["handler"](
        {"owner": "sub-agent", "paths": ["lib/foo"], "task": "B"}
    )
    assert r1["status"] == "granted"
    assert r2["status"] == "granted"
    assert r2["warnings"][0]["owner"] == "parent"


def test_no_session_id_falls_back_to_owner(tmp_path, monkeypatch):
    """With session env unset, identity falls back to owner: same owner
    overlap auto-grants (idempotent), different owner conflicts."""
    monkeypatch.delenv("CLAUDE_CODE_SESSION_ID", raising=False)
    monkeypatch.delenv("CODEX_THREAD_ID", raising=False)
    monkeypatch.delenv("AGENT_SESSION_ID", raising=False)
    store = TaskboardStore(
        json_path=tmp_path / "taskboard.json",
        md_path=tmp_path / "taskboard.md",
    )
    tools = server.build_tools(store)

    r1 = tools["taskboard_claim"]["handler"](
        {"owner": "alice", "paths": ["lib/foo"], "task": "A"}
    )
    r2 = tools["taskboard_claim"]["handler"](
        {"owner": "alice", "paths": ["lib/foo"], "task": "A2"}
    )
    assert r1["status"] == "granted"
    assert r2["claim_id"] == r1["claim_id"]  # same owner → idempotent re-claim

    r3 = tools["taskboard_claim"]["handler"](
        {"owner": "bob", "paths": ["lib/foo"], "task": "B"}
    )
    assert r3["status"] == "pending"  # different owner → conflict
