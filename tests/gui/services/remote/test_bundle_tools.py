"""MCP tab bundle and stage tool tests."""

from __future__ import annotations

from pathlib import Path

import pytest
from zcu_tools.mcp.measure import server as mcp_server


@pytest.fixture(autouse=True)
def _clear_mcp_policy_state():
    mcp_server._SESSION.clear_policy_state()
    yield
    mcp_server._SESSION.clear_policy_state()


# ---------------------------------------------------------------------------
# MCP 45: gui_tab_new is PURE — reverted to the auto-generated tab.new forwarder
# (returns just {tab_id}); the fan-out + guide fold moved to gui_tab_open.
# ---------------------------------------------------------------------------


def test_tab_new_is_pure_generated_forwarder():
    """gui_tab_new forwards tab.new and returns ONLY its result ({tab_id}) — it no
    longer fans out over tab.snapshot / list_paths nor folds a guide.

    Like test_mcp_wrappers_map_to_expected_rpc, the generated forwarder captures
    the guarded send_gui_rpc as a closure at import time, so monkeypatching the
    module attribute does not reach it. Re-generate with a recording send_fn (the
    same projection the real bridge builds) to assert it forwards ONLY tab.new.
    """
    from zcu_tools.gui.app.main.services.remote.method_specs import METHOD_SPECS
    from zcu_tools.mcp.core.bridge import generate_tools
    from zcu_tools.mcp.measure import server as mcp_server

    calls: list[tuple[str, dict]] = []

    def fake_send(method: str, params: dict, timeout_seconds: float = 30.0) -> dict:
        del timeout_seconds
        calls.append((method, params))
        return {"tab_id": "tw-1"}

    tools = generate_tools(
        mcp_server._CONFIG,
        METHOD_SPECS,
        mcp_server._MCP_EXPOSURE.non_generated_methods,
        fake_send,
    )
    out = tools["gui_tab_new"]["handler"]({"adapter_name": "fake/freq"})

    assert out == {"tab_id": "tw-1"}
    # Exactly one RPC — tab.new — and no fan-out reads or guide fetch.
    assert calls == [("tab.new", {"adapter_name": "fake/freq"})]


# ---------------------------------------------------------------------------
# MCP 45: gui_tab_run_start is PURE — no figure fold; the figure-fold helper is
# now exercised only via the stage tools + directly.
# ---------------------------------------------------------------------------


def test_run_start_finished_carries_figure(monkeypatch):
    """gui_tab_run_start FINISHED reply includes 'figure' — the run plot rendered to a
    temp PNG (the run's OWN visual result, MCP 46). 'figure' was removed in MCP 45
    and is now re-added as a base-tool fold (not a stage bundle fold)."""
    from tempfile import gettempdir

    from zcu_tools.mcp.measure import server as mcp_server

    # No tracked op for this key -> _start_op_with_short_wait takes the
    # "settled synchronously" branch (status='finished', runs the product).
    calls: list[str] = []

    def fake_send(method: str, params: dict, timeout_seconds: float = 30.0) -> dict:
        del timeout_seconds
        calls.append(method)
        if method == "tab.snapshot":
            # tab.snapshot always returns {tabs: [...]} (single tab → one element).
            return {"tabs": [{"interaction": {"has_run_result": True}}]}
        if method == "tab.get_current_figure":
            return {"bytes": 9, "saved_to": params["out_path"]}
        return {}

    monkeypatch.setattr(mcp_server, "send_gui_rpc", fake_send)
    out = mcp_server.TOOLS["gui_tab_run_start"]["handler"]({"tab_id": "rt-1"})

    assert out["status"] == "finished"
    # MCP 46: figure is the run's OWN visual result — present on FINISHED.
    assert out["figure"] == str(Path(gettempdir()) / "measure_fig_rt-1.png")
    assert "tab.get_current_figure" in calls


def test_run_start_pending_has_no_figure(monkeypatch):
    """A 'pending' gui_tab_run_start (slow run) must NOT include 'figure' — the plot
    only exists once the run settles (MCP 46)."""
    from zcu_tools.mcp.measure import server as mcp_server

    mcp_server._SESSION.operation_handles.update({"tab:slow-1": 5})

    def fake_send(method, params, timeout_seconds=30.0):
        del timeout_seconds, params
        if method == "operation.await":
            raise RuntimeError("GUI Error (timeout): still running")
        return {}

    monkeypatch.setattr(mcp_server, "send_gui_rpc", fake_send)
    out = mcp_server.TOOLS["gui_tab_run_start"]["handler"]({"tab_id": "slow-1"})

    assert out["status"] == "pending"
    assert "figure" not in out


def test_op_wait_finished_reports_status_only(monkeypatch):
    """The generic gui_op_wait FINISHED reply reports ONLY status (+waited_seconds) —
    NO figure fold (P2 / ADR-0026 §8). The run's visual product is read from the
    START finished reply or gui_tab_get_current_figure, not from the wait."""
    from zcu_tools.mcp.measure import server as mcp_server

    rendered: list[str] = []

    def fake_send(method, params, timeout_seconds=30.0):
        del timeout_seconds, params
        if method == "operation.await":
            return {"status": "finished", "reason": "completed"}
        if method == "tab.get_current_figure":
            rendered.append("called")
            return {"bytes": 9}
        return {}

    monkeypatch.setattr(mcp_server, "send_gui_rpc", fake_send)
    out = mcp_server.TOOLS["gui_op_wait"]["handler"]({"handle": 3})

    assert out["status"] == "finished"
    assert "figure" not in out
    # The wait never renders the figure — that is the START reply / getter's job.
    assert rendered == []


def test_fold_finished_figure_finished_folds_pending_does_not(monkeypatch):
    """The figure-fold helper renders + folds 'figure' on a FINISHED reply and is
    a no-op on a pending one (used by gui_tab_run / gui_tab_analyze_review)."""
    from tempfile import gettempdir

    from zcu_tools.mcp.measure import server as mcp_server

    rendered: list[str] = []

    def fake_send(method: str, params: dict, timeout_seconds: float = 30.0) -> dict:
        del timeout_seconds
        if method == "tab.get_current_figure":
            rendered.append(params["out_path"])
            return {"bytes": 9, "saved_to": params["out_path"]}
        return {}

    monkeypatch.setattr(mcp_server, "send_gui_rpc", fake_send)

    # FINISHED -> renders + folds the figure path.
    finished = mcp_server._fold_finished_figure("az-1", {"status": "finished"})
    assert finished["figure"] == str(Path(gettempdir()) / "measure_fig_az-1.png")
    assert len(rendered) == 1

    # A pending reply (status != finished) must NOT trigger a render.
    pending = {"status": "pending", "message": "still running"}
    folded = mcp_server._fold_finished_figure("az-1", pending)
    assert folded == {"status": "pending", "message": "still running"}
    assert "figure" not in folded
    assert len(rendered) == 1


def test_fold_finished_figure_swallows_render_error(monkeypatch):
    """A figure-render failure must not mask an otherwise-good finished reply:
    the reply still settles, with figure=None."""
    from zcu_tools.mcp.measure import server as mcp_server

    def fake_send(method: str, params: dict, timeout_seconds: float = 30.0) -> dict:
        del params, timeout_seconds
        if method == "tab.get_current_figure":
            raise RuntimeError("plotting hiccup")
        return {}

    monkeypatch.setattr(mcp_server, "send_gui_rpc", fake_send)
    out = mcp_server._fold_finished_figure("x-1", {"status": "finished"})
    assert out["status"] == "finished"
    assert out["figure"] is None


# ---------------------------------------------------------------------------
# MCP 44 Phase ③: gui_tab_analyze folds the writeback preview; pending does not
# ---------------------------------------------------------------------------


def test_fold_writeback_preview_pending_does_not_fold(monkeypatch):
    """An INTERACTIVE 'pending' analyze must NOT read the writeback preview (no
    draft has been produced yet)."""
    from zcu_tools.mcp.measure import server as mcp_server

    calls: list[str] = []

    def fake_send(method, params, timeout_seconds=30.0):
        del params, timeout_seconds
        calls.append(method)
        return {}

    monkeypatch.setattr(mcp_server, "send_gui_rpc", fake_send)
    pending = {"status": "pending", "message": "still picking"}
    out = mcp_server._fold_writeback_preview("az-1", pending)
    assert out == {"status": "pending", "message": "still picking"}
    assert "writeback_preview" not in out
    assert "tab.writeback_preview" not in calls


def test_fold_writeback_preview_swallows_failure(monkeypatch):
    """A tab.writeback_preview failure must not break an otherwise-good finished
    analyze reply — the key is simply omitted (mirrors the figure/guide folds)."""
    from zcu_tools.mcp.measure import server as mcp_server

    def fake_send(method, params, timeout_seconds=30.0):
        del params, timeout_seconds
        if method == "tab.writeback_preview":
            raise RuntimeError("preview hiccup")
        return {}

    monkeypatch.setattr(mcp_server, "send_gui_rpc", fake_send)
    out = mcp_server._fold_writeback_preview("az-1", {"status": "finished"})
    assert out["status"] == "finished"
    assert "writeback_preview" not in out


# ---------------------------------------------------------------------------
# MCP 44 / P4 Phase ④: gui_tab_writeback_apply stays pure (save folded into gui_tab_commit)
# ---------------------------------------------------------------------------


def test_writeback_apply_is_pure_generated_forwarder():
    """gui_tab_writeback_apply forwards tab.writeback_apply and returns ONLY its result
    ({applied_ids, written, context_version}) — it no longer takes save_data nor chains
    tab.save_data (that moved to gui_tab_commit).

    Generated forwarder captures send_gui_rpc as a closure at import time, so a
    module-attr monkeypatch does not reach it — re-generate with a recording
    send_fn (the same projection the real bridge builds) to assert it forwards ONLY
    tab.writeback_apply. The MCP schema must not expose save_data.
    """
    from zcu_tools.gui.app.main.services.remote.method_specs import METHOD_SPECS
    from zcu_tools.mcp.core.bridge import generate_tools
    from zcu_tools.mcp.measure import server as mcp_server

    # The agent-facing schema carries only tab_id (expected_versions is mcp_hidden;
    # save_data is gone).
    assert set(
        mcp_server.TOOLS["gui_tab_writeback_apply"]["inputSchema"]["properties"]
    ) == {"tab_id"}

    calls: list[tuple[str, dict]] = []

    def fake_send(method: str, params: dict, timeout_seconds: float = 30.0) -> dict:
        del timeout_seconds
        calls.append((method, params))
        return {"applied_ids": ["md-0", "ml-1"]}

    tools = generate_tools(
        mcp_server._CONFIG,
        METHOD_SPECS,
        mcp_server._MCP_EXPOSURE.non_generated_methods,
        fake_send,
    )
    out = tools["gui_tab_writeback_apply"]["handler"]({"tab_id": "t1"})

    assert out == {"applied_ids": ["md-0", "ml-1"]}
    assert calls == [("tab.writeback_apply", {"tab_id": "t1"})]


# ---------------------------------------------------------------------------
# Generic gui_op_poll is PURE status (P2 / ADR-0026 §8) — no figure fold.
# ---------------------------------------------------------------------------


def test_op_poll_finished_reports_status_only(monkeypatch):
    """A 'finished' gui_op_poll reply reports ONLY status — NO figure fold. The run's
    visual product is read from the START finished reply or gui_tab_get_current_figure."""
    from zcu_tools.mcp.measure import server as mcp_server

    calls: list[str] = []

    def fake_send(method, params, timeout_seconds=30.0):
        del timeout_seconds, params
        calls.append(method)
        if method == "operation.await":
            return {"status": "finished"}
        return {}

    monkeypatch.setattr(mcp_server, "send_gui_rpc", fake_send)
    out = mcp_server.TOOLS["gui_op_poll"]["handler"]({"handle": 7})
    assert out["status"] == "finished"
    assert "figure" not in out
    # The poll never renders the figure — that is the START reply / getter's job.
    assert "tab.get_current_figure" not in calls


# ---------------------------------------------------------------------------
# MCP 45 / P4 Phase ②: gui_tab_run(tab_id, edits) configures + runs an existing tab
# (no tab creation, no guide fold) and folds the figure + analyze-params
# ---------------------------------------------------------------------------


def _run_stage_fake_send(calls: list[tuple[str, dict]]):
    """A send_gui_rpc stub covering every RPC gui_tab_run fans out over.

    Records (method, params) in call order so a test can assert the sequence and
    the exact values forwarded. tab.run_start captures no operation_id here (the
    real send_gui_rpc does, but it is monkeypatched out), so
    _start_op_with_short_wait sees no handle and settles synchronously — exactly
    the fast-run path.
    """

    def fake_send(method: str, params: dict, timeout_seconds: float = 30.0) -> dict:
        del timeout_seconds
        calls.append((method, dict(params)))
        if method == "tab.snapshot":
            # has_run_result drives the run-finished tab summary. tab.snapshot
            # always returns {tabs: [...]} (single tab → one-element list).
            return {
                "tabs": [
                    {
                        "editor_id": "stage-ed",
                        "interaction": {"is_running": False, "has_run_result": True},
                    }
                ]
            }
        if method == "tab.set_cfg":
            # Stage2 batch setter: aggregate result across all edits.
            return {"valid": True, "removed": [], "added": []}
        if method == "tab.get_current_figure":
            return {"bytes": 9, "saved_to": params["out_path"]}
        if method == "tab.get_analyze_params":
            return {"analyze_params": {"smooth": 1}}
        # tab.run_start, anything else
        return {}

    return fake_send


def test_run_configures_runs_and_stops_before_analyze(monkeypatch):
    """gui_tab_run operates on the given tab_id: tab.set_cfg then tab.run_start,
    NEVER creating a tab and NEVER calling analyze. A finished reply carries the
    figure (from gui_tab_run_start's own FINISHED fold) + the stage-specific
    analyze-params spec."""
    from zcu_tools.mcp.measure import server as mcp_server

    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(mcp_server, "send_gui_rpc", _run_stage_fake_send(calls))

    out = mcp_server.TOOLS["gui_tab_run"]["handler"](
        {
            "tab_id": "stage-tab",
            "edits": [
                {"path": "reps", "value": 100},
                {"path": "sweep.gain.expts", "value": 5},
            ],
        }
    )

    methods = [m for m, _ in calls]
    # No tab creation (the tab already exists); edits via tab.set_cfg precede tab.run_start.
    assert "tab.new" not in methods
    assert methods.index("tab.set_cfg") < methods.index("tab.run_start")
    assert "tab.analyze" not in methods
    # tab.snapshot is still called by gui_tab_run_start's finished-reply fold
    # (_run_tab_summary); gui_tab_run no longer calls it for editor_id resolution.

    # Finished run reply carries the folded figure AND the analyze-params spec.
    assert out["status"] == "finished"
    assert "figure" in out
    assert out["figure"].endswith("measure_fig_stage-tab.png")
    assert out["analyze_params"] == {"smooth": 1}
    assert ("tab.get_analyze_params", {"tab_id": "stage-tab"}) in calls


def test_run_does_not_double_fold_figure(monkeypatch):
    """gui_tab_run must NOT call tab.get_current_figure a second time —
    the figure arrives already folded inside gui_tab_run_start's FINISHED reply (MCP 46).
    Exactly one tab.get_current_figure call is expected."""
    from zcu_tools.mcp.measure import server as mcp_server

    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(mcp_server, "send_gui_rpc", _run_stage_fake_send(calls))

    out = mcp_server.TOOLS["gui_tab_run"]["handler"]({"tab_id": "stage-tab"})

    assert out["status"] == "finished"
    assert "figure" in out
    figure_calls = [m for m, _ in calls if m == "tab.get_current_figure"]
    # Exactly one render: from gui_tab_run_start's fold — not a second from gui_tab_run.
    assert len(figure_calls) == 1


def test_run_edits_preserve_order_and_numbers(monkeypatch):
    """The ORDERED {path, value} list is forwarded to tab.set_cfg verbatim (order
    preserved so a $ref switch lands before its children); numeric values reach the
    wire as numbers (NOT stringified)."""
    from zcu_tools.mcp.measure import server as mcp_server

    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(mcp_server, "send_gui_rpc", _run_stage_fake_send(calls))

    mcp_server.TOOLS["gui_tab_run"]["handler"](
        {
            "tab_id": "stage-tab",
            "edits": [
                {"path": "sweep.type", "value": "gauss"},
                {"path": "reps", "value": 100},
                {"path": "gain", "value": 0.2},
            ],
        }
    )

    # gui_tab_run sends one tab.set_cfg call with the ordered batch edits.
    set_cfg_calls = [params for method, params in calls if method == "tab.set_cfg"]
    assert len(set_cfg_calls) == 1
    edits = set_cfg_calls[0]["edits"]
    # Order is preserved exactly as supplied (ref-switch first).
    assert edits == [
        {"path": "sweep.type", "value": "gauss"},
        {"path": "reps", "value": 100},
        {"path": "gain", "value": 0.2},
    ]
    by_path = {e["path"]: e["value"] for e in edits}
    assert isinstance(by_path["reps"], int)
    assert isinstance(by_path["gain"], float)


def test_run_without_edits_runs_current_cfg(monkeypatch):
    """Omitting 'edits' runs the tab's current cfg — no tab.set_cfg fires."""
    from zcu_tools.mcp.measure import server as mcp_server

    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(mcp_server, "send_gui_rpc", _run_stage_fake_send(calls))

    out = mcp_server.TOOLS["gui_tab_run"]["handler"]({"tab_id": "stage-tab"})

    assert "tab.set_cfg" not in [m for m, _ in calls]
    assert out["status"] == "finished"
    assert out["analyze_params"] == {"smooth": 1}


def test_run_rejects_map_edits(monkeypatch):
    """'edits' must be an ordered list — a {path: value} map is rejected (the map
    form lost ref-switch ordering)."""
    from zcu_tools.mcp.measure import server as mcp_server

    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(mcp_server, "send_gui_rpc", _run_stage_fake_send(calls))

    with pytest.raises(ValueError, match="ordered list"):
        mcp_server.TOOLS["gui_tab_run"]["handler"](
            {"tab_id": "stage-tab", "edits": {"reps": 100}}
        )


def test_run_pending_run_owes_figure_and_omits_analyze_params(monkeypatch):
    """A slow run degrades to {status:'pending', owed} and must NOT fold
    analyze_params (nothing settled to analyze yet); 'owed' names the deferred read."""
    from zcu_tools.mcp.measure import server as mcp_server

    mcp_server._SESSION.operation_handles.update({"tab:stage-tab": 5})
    calls: list[str] = []

    def fake_send(method, params, timeout_seconds=30.0):
        del timeout_seconds, params
        calls.append(method)
        if method == "operation.await":
            raise RuntimeError("GUI Error (timeout): still running")
        return {}

    monkeypatch.setattr(mcp_server, "send_gui_rpc", fake_send)
    out = mcp_server.TOOLS["gui_tab_run"]["handler"]({"tab_id": "stage-tab"})

    assert out["status"] == "pending"
    assert "figure" in out["owed"]
    assert "analyze_params" not in out
    assert "tab.get_analyze_params" not in calls


def test_fold_analyze_params_fetch_failure_is_swallowed(monkeypatch):
    """A tab.get_analyze_params failure must not mask an otherwise-good finished
    run reply: the reply settles, with analyze_params=None."""
    from zcu_tools.mcp.measure import server as mcp_server

    monkeypatch.setattr(mcp_server, "send_gui_rpc", lambda *a, **k: None)
    out = mcp_server._fold_analyze_params("stage-tab", {"status": "finished"})
    assert out["status"] == "finished"
    assert out["analyze_params"] is None


# ---------------------------------------------------------------------------
# MCP 45 / P4 Phase ①: gui_tab_open creates a tab + folds the editing context
# + the adapter guide (guide always sent by default; skip_guide=true to opt out)
# ---------------------------------------------------------------------------


def test_open_creates_tab_and_folds_context_and_guide(monkeypatch):
    """gui_tab_open creates the tab, fans out the two editing-context reads
    (snapshot for editor_id, tab.get_cfg for the settable cfg tree) and folds the
    adapter guide into one reply. The guide is included on every call by default —
    no server-side dedup; the cfg tree already carries the current values."""
    from zcu_tools.mcp.measure import server as mcp_server

    calls: list[tuple[str, dict]] = []
    tree = {"reps": 100, "sweep": {"freq": {"start": 1.0}}}

    def fake_send(method: str, params: dict, timeout_seconds: float = 30.0) -> dict:
        del timeout_seconds
        calls.append((method, dict(params)))
        return {
            "tab.new": {"tab_id": "tw-1"},
            # tab.snapshot always returns {tabs: [...]} (single tab → one element).
            "tab.snapshot": {"tabs": [{"editor_id": "ed-tw-1", "interaction": {}}]},
            "tab.get_cfg": {"tree": tree},
            "adapter.guide": {"guide": {"behavior": "measures X"}},
        }[method]

    monkeypatch.setattr(mcp_server, "send_gui_rpc", fake_send)
    out = mcp_server.TOOLS["gui_tab_open"]["handler"]({"adapter_name": "fake/freq"})

    # tab.new first (adapter_name verbatim), then the two reads keyed by the new
    # tab_id, then the adapter.guide fetch (always by default — no server-side gating).
    assert calls == [
        ("tab.new", {"adapter_name": "fake/freq"}),
        ("tab.snapshot", {"tab_id": "tw-1"}),
        ("tab.get_cfg", {"tab_id": "tw-1"}),
        ("adapter.guide", {"adapter_name": "fake/freq"}),
    ]
    assert out == {
        "tab_id": "tw-1",
        "adapter": "fake/freq",
        "editor_id": "ed-tw-1",
        "tree": tree,
        "guide": {"behavior": "measures X"},
    }


def test_open_second_call_default_still_sends_guide(monkeypatch):
    """A repeat gui_tab_open for the same adapter (without skip_guide) still
    returns the guide — there is no server-side dedup any more. The server cannot
    know whether the caller's context still has the guide (e.g. context-reset,
    sub-agent)."""
    from zcu_tools.mcp.measure import server as mcp_server

    def fake_send(method: str, params: dict, timeout_seconds: float = 30.0) -> dict:
        del params, timeout_seconds
        return {
            "tab.new": {"tab_id": "tw-1"},
            # tab.snapshot always returns {tabs: [...]} (single tab → one element).
            "tab.snapshot": {"tabs": [{"editor_id": "ed-1", "interaction": {}}]},
            "tab.get_cfg": {"tree": {"reps": 1}},
            "adapter.guide": {"guide": {"behavior": "measures X"}},
        }[method]

    monkeypatch.setattr(mcp_server, "send_gui_rpc", fake_send)

    first = mcp_server.TOOLS["gui_tab_open"]["handler"]({"adapter_name": "amp_rabi"})
    second = mcp_server.TOOLS["gui_tab_open"]["handler"]({"adapter_name": "amp_rabi"})
    assert first["guide"] == {"behavior": "measures X"}
    assert "guide_omitted" not in first
    # Second call — no skip_guide — also returns the full guide.
    assert second["guide"] == {"behavior": "measures X"}
    assert "guide_omitted" not in second


def test_open_skip_guide_omits_guide(monkeypatch):
    """skip_guide=true suppresses the adapter.guide RPC and returns
    guide_omitted: True. Callers use this only when they know the guide is
    already in their context (e.g. same adapter opened earlier this session)."""
    from zcu_tools.mcp.measure import server as mcp_server

    def fake_send(method: str, params: dict, timeout_seconds: float = 30.0) -> dict:
        del params, timeout_seconds
        return {
            "tab.new": {"tab_id": "tw-1"},
            "tab.snapshot": {"tabs": [{"editor_id": "ed-1", "interaction": {}}]},
            "tab.get_cfg": {"tree": {"reps": 1}},
            "adapter.guide": {"guide": {"behavior": "measures X"}},
        }[method]

    monkeypatch.setattr(mcp_server, "send_gui_rpc", fake_send)

    result = mcp_server.TOOLS["gui_tab_open"]["handler"](
        {"adapter_name": "amp_rabi", "skip_guide": True}
    )
    assert "guide" not in result
    assert result["guide_omitted"] is True


# ---------------------------------------------------------------------------
# MCP 45 / P4 Phase ③: gui_tab_analyze_review analyzes + folds summary + figure + writeback
# ---------------------------------------------------------------------------


def test_analyze_review_analyzes_and_folds_figure_and_writeback(monkeypatch):
    """gui_tab_analyze_review runs gui_tab_analyze, then folds the writeback preview
    onto a FINISHED FIT reply. 'figure' comes from gui_tab_analyze's own FINISHED fold
    (MCP 46); 'writeback_preview' is the stage-specific {has_draft, items} fold."""
    from tempfile import gettempdir

    from zcu_tools.mcp.measure import server as mcp_server

    # No tracked op -> the analyze short-wait settles synchronously (fast FIT).
    calls: list[tuple[str, dict]] = []

    def fake_send(method: str, params: dict, timeout_seconds: float = 30.0) -> dict:
        del timeout_seconds
        calls.append((method, dict(params)))
        if method == "tab.get_analyze_result":
            return {"summary": {"t1": 12.3}}
        if method == "tab.get_current_figure":
            return {"bytes": 9, "saved_to": params["out_path"]}
        if method == "tab.writeback_preview":
            return {"has_draft": True, "items": [{"id": "md-0", "target_name": "q_f"}]}
        return {}

    monkeypatch.setattr(mcp_server, "send_gui_rpc", fake_send)
    out = mcp_server.TOOLS["gui_tab_analyze_review"]["handler"]({"tab_id": "az-1"})

    assert out["status"] == "finished"
    assert out["summary"] == {"t1": 12.3}
    assert out["figure"] == str(Path(gettempdir()) / "measure_fig_az-1.png")
    # The full {has_draft, items} object is surfaced verbatim (not just the list).
    assert out["writeback_preview"] == {
        "has_draft": True,
        "items": [{"id": "md-0", "target_name": "q_f"}],
    }
    assert ("tab.analyze", {"tab_id": "az-1"}) in calls


def test_analyze_review_passes_updates_and_wait_seconds(monkeypatch):
    """'updates' and 'wait_seconds' flow through to the underlying analyze."""
    from zcu_tools.mcp.measure import server as mcp_server

    calls: list[tuple[str, dict]] = []

    def fake_send(method: str, params: dict, timeout_seconds: float = 30.0) -> dict:
        del timeout_seconds
        calls.append((method, dict(params)))
        if method == "tab.get_analyze_result":
            return {"summary": {}}
        if method == "tab.get_current_figure":
            return {"bytes": 9, "saved_to": params["out_path"]}
        if method == "tab.writeback_preview":
            return {"has_draft": False, "items": []}
        return {}

    monkeypatch.setattr(mcp_server, "send_gui_rpc", fake_send)
    mcp_server.TOOLS["gui_tab_analyze_review"]["handler"](
        {"tab_id": "az-1", "updates": {"smooth": 3}, "wait_seconds": 2.0}
    )

    analyze_params = [p for m, p in calls if m == "tab.analyze"][0]
    assert analyze_params == {"tab_id": "az-1", "updates": {"smooth": 3}}


def test_analyze_review_does_not_double_fold_figure(monkeypatch):
    """gui_tab_analyze_review must NOT call tab.get_current_figure a second time —
    the figure arrives already folded inside gui_tab_analyze's FINISHED reply (MCP 46).
    Exactly one tab.get_current_figure call is expected."""
    from zcu_tools.mcp.measure import server as mcp_server

    calls: list[tuple[str, dict]] = []

    def fake_send(method: str, params: dict, timeout_seconds: float = 30.0) -> dict:
        del timeout_seconds
        calls.append((method, dict(params)))
        if method == "tab.get_analyze_result":
            return {"summary": {"t1": 12.3}}
        if method == "tab.get_current_figure":
            return {"bytes": 9, "saved_to": params["out_path"]}
        if method == "tab.writeback_preview":
            return {"has_draft": False, "items": []}
        return {}

    monkeypatch.setattr(mcp_server, "send_gui_rpc", fake_send)
    out = mcp_server.TOOLS["gui_tab_analyze_review"]["handler"]({"tab_id": "az-1"})

    assert out["status"] == "finished"
    assert "figure" in out
    figure_calls = [m for m, _ in calls if m == "tab.get_current_figure"]
    # Exactly one render: from gui_tab_analyze's fold — not a second from analyze_review.
    assert len(figure_calls) == 1


def test_analyze_review_pending_interactive_owes_and_omits_folds(monkeypatch):
    """An INTERACTIVE analyze degrades to pending -> gui_tab_analyze_review folds
    NOTHING (no figure, no writeback preview) and names the deferred reads in 'owed'."""
    from zcu_tools.mcp.measure import server as mcp_server

    mcp_server._SESSION.operation_handles.update({"analyze:az-1": 9})
    calls: list[str] = []

    def fake_send(method, params, timeout_seconds=30.0):
        del params, timeout_seconds
        calls.append(method)
        if method == "operation.await":
            raise RuntimeError("GUI Error (timeout): user still picking")
        return {}

    monkeypatch.setattr(mcp_server, "send_gui_rpc", fake_send)
    out = mcp_server.TOOLS["gui_tab_analyze_review"]["handler"]({"tab_id": "az-1"})

    assert out["status"] == "pending"
    assert "owed" in out
    assert "figure" not in out
    assert "writeback_preview" not in out
    assert "tab.get_current_figure" not in calls
    assert "tab.writeback_preview" not in calls


# ---------------------------------------------------------------------------
# MCP 45 / P4 Phase ④: gui_tab_commit applies the writeback, optionally saving
# ---------------------------------------------------------------------------


def test_commit_default_applies_only(monkeypatch):
    """save defaults 'none': apply runs, no save, status committed, saved null."""
    from zcu_tools.mcp.measure import server as mcp_server

    calls: list[str] = []

    def fake_send(method, params, timeout_seconds=30.0):
        del params, timeout_seconds
        calls.append(method)
        if method == "tab.writeback_apply":
            return {
                "applied_ids": ["md-0", "ml-1"],
                "written": ["q_f"],
                "context_version": 5,
            }
        return {}

    monkeypatch.setattr(mcp_server, "send_gui_rpc", fake_send)
    out = mcp_server.TOOLS["gui_tab_commit"]["handler"]({"tab_id": "t1"})

    assert out == {
        "status": "committed",
        "applied_ids": ["md-0", "ml-1"],
        "written": ["q_f"],
        "context_version": 5,
        "saved": None,
    }
    assert "tab.save_data" not in calls and "tab.save_image" not in calls


def test_commit_save_data_chains_save_and_folds_result(monkeypatch):
    """save='data': apply then gui_tab_save(artifact='data'), folding the result."""
    from zcu_tools.mcp.measure import server as mcp_server

    calls: list[tuple[str, dict]] = []

    def fake_send(method, params, timeout_seconds=30.0):
        del timeout_seconds
        calls.append((method, dict(params)))
        if method == "tab.writeback_apply":
            return {"applied_ids": ["md-0"], "written": [], "context_version": 6}
        if method == "tab.save_data":
            return {"data_path": "/results/Q1/data_0001.hdf5"}
        return {}

    monkeypatch.setattr(mcp_server, "send_gui_rpc", fake_send)
    out = mcp_server.TOOLS["gui_tab_commit"]["handler"](
        {"tab_id": "t1", "save": "data"}
    )

    methods = [m for m, _ in calls]
    # apply runs first, tab.save_data second (apply is committed before the save).
    assert methods.index("tab.writeback_apply") < methods.index("tab.save_data")
    assert out["status"] == "committed"
    assert out["applied_ids"] == ["md-0"]
    assert out["saved"]["data_path"] == "/results/Q1/data_0001.hdf5"


def test_commit_save_failure_is_fail_soft(monkeypatch):
    """fail-soft: when the apply succeeds but the follow-up save raises (e.g. a DATA
    precondition), the applied_ids are NOT lost — status flips to 'partial' and the
    error lands in save_error."""
    from zcu_tools.mcp.measure import server as mcp_server

    def fake_send(method, params, timeout_seconds=30.0):
        del params, timeout_seconds
        if method == "tab.writeback_apply":
            return {"applied_ids": ["md-0"], "written": ["q_f"], "context_version": 7}
        if method == "tab.save_data":
            raise RuntimeError("no run result to save")
        return {}

    monkeypatch.setattr(mcp_server, "send_gui_rpc", fake_send)
    out = mcp_server.TOOLS["gui_tab_commit"]["handler"](
        {"tab_id": "t1", "save": "data"}
    )

    assert out["status"] == "partial"
    assert out["applied_ids"] == ["md-0"]  # the committed writeback is preserved
    assert out["saved"] is None
    assert "no run result" in out["save_error"]


def test_commit_rejects_bad_save_enum(monkeypatch):
    """An unknown 'save' value Fast-Fails before the apply runs."""
    from zcu_tools.mcp.measure import server as mcp_server

    calls: list[str] = []

    def fake_send(method, params, timeout_seconds=30.0):
        del params, timeout_seconds
        calls.append(method)
        return {}

    monkeypatch.setattr(mcp_server, "send_gui_rpc", fake_send)
    with pytest.raises(ValueError, match="save must be one of"):
        mcp_server.TOOLS["gui_tab_commit"]["handler"]({"tab_id": "t1", "save": "bogus"})
    # Validation precedes any wire call (no half-applied writeback).
    assert calls == []


def test_explicit_adapter_guide_tool_still_works():
    """gui_adapter_guide stays a generated forwarder mapping to adapter.guide —
    an explicit re-read that the first-use fold does not remove or alter.

    Like test_mcp_wrappers_map_to_expected_rpc, generated forwarders capture the
    guarded send_gui_rpc as a closure at import time, so monkeypatching the module
    attribute does not reach them. Re-generate with a recording send_fn (the same
    projection the real bridge builds) to assert the wrapper -> (method, params)
    mapping is intact.
    """
    from zcu_tools.gui.app.main.services.remote.method_specs import METHOD_SPECS
    from zcu_tools.mcp.core.bridge import generate_tools
    from zcu_tools.mcp.measure import server as mcp_server

    calls: list[tuple[str, dict]] = []

    def fake_send(method: str, params: dict, timeout_seconds: float = 30.0) -> dict:
        del timeout_seconds
        calls.append((method, params))
        return {"guide": {"behavior": "measures X"}}

    tools = generate_tools(
        mcp_server._CONFIG,
        METHOD_SPECS,
        mcp_server._MCP_EXPOSURE.non_generated_methods,
        fake_send,
    )
    out = tools["gui_adapter_guide"]["handler"]({"adapter_name": "amp_rabi"})
    assert out == {"guide": {"behavior": "measures X"}}
    assert calls == [("adapter.guide", {"adapter_name": "amp_rabi"})]
