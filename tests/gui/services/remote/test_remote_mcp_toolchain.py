"""Full MCP-facing remote toolchain coverage."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from zcu_tools.device.fake import FakeDeviceInfo
from zcu_tools.device.yoko import YOKOGS200Info
from zcu_tools.gui.app.main.event_bus import (
    DeviceSetupFinishedPayload,
    DeviceSetupStartedPayload,
    GuiEvent,
)
from zcu_tools.gui.app.main.services.device import (
    DeviceSetupSnapshot,
    SetupDeviceRequest,
)
from zcu_tools.gui.app.main.services.remote.dispatch import METHOD_REGISTRY
from zcu_tools.mcp.measure.server import TOOLS

from ._helpers import Fixture, call, open_client, recv_push


@pytest.fixture()
def fx(qapp):  # noqa: ARG001
    f = Fixture()
    f.start()
    yield f
    f.stop()


def test_event_requery_hints_point_to_registered_methods():
    assert "device.active_setup" in METHOD_REGISTRY
    assert "context.get_md_attr" in METHOD_REGISTRY
    assert "context.get_ml" in METHOD_REGISTRY


def test_device_setup_started_and_finished_push(fx):
    sock = open_client(fx.service.port)
    try:
        call(
            sock,
            "events.subscribe",
            {"events": ["device_setup_started", "device_setup_finished"]},
        )
        fx.bus.emit(
            GuiEvent.DEVICE_SETUP_STARTED, DeviceSetupStartedPayload(name="bias")
        )
        started = recv_push(sock, "device_setup_started")
        assert started["payload"] == {"name": "bias"}

        fx.bus.emit(
            GuiEvent.DEVICE_SETUP_FINISHED,
            DeviceSetupFinishedPayload(name="bias", outcome="finished"),
        )
        finished = recv_push(sock, "device_setup_finished")
        assert finished["payload"]["name"] == "bias"
        assert finished["payload"]["outcome"] == "finished"
    finally:
        sock.close()


def test_device_active_setup_names_the_device(fx):
    # active_setup now only names which device is setting up; live progress is
    # via device.setup_progress (ADR-0013 device↔run alignment).
    fx.ctrl.get_active_device_setup = MagicMock(  # type: ignore[method-assign]
        return_value=DeviceSetupSnapshot(device_name="bias")
    )
    sock = open_client(fx.service.port)
    try:
        resp = call(sock, "device.active_setup")
        assert resp["ok"] is True
        assert resp["result"]["active_setup"] == {"device_name": "bias"}
    finally:
        sock.close()


def test_operation_progress_device_setup_bars(fx):
    # operation.progress covers device setup too: live (token, ProgressBarModel).
    import time

    from zcu_tools.gui.app.main.pbar_host import ProgressBarModel

    m = ProgressBarModel(label="Ramp", total=10, start_time=time.monotonic())
    m.set_n(3)
    fx.ctrl.get_operation_progress = MagicMock(  # type: ignore[method-assign]
        return_value=((1, m),)
    )
    sock = open_client(fx.service.port)
    try:
        # operation.progress is unified across run + device setup, keyed by id.
        resp = call(sock, "operation.progress", {"operation_id": 7})
        assert resp["ok"] is True
        assert resp["result"]["active"] is True
        bar = resp["result"]["bars"][0]
        assert bar["token"] == 1
        assert bar["maximum"] == 10 and bar["value"] == 3
        assert bar["n"] == 3 and bar["total"] == 10
        assert "Ramp" in bar["format"]
        fx.ctrl.get_operation_progress.assert_called_with(7)
    finally:
        sock.close()


def test_operation_progress_idle_returns_empty(fx):
    fx.ctrl.get_operation_progress = MagicMock(return_value=())  # type: ignore[method-assign]
    sock = open_client(fx.service.port)
    try:
        resp = call(sock, "operation.progress", {"operation_id": 7})
        assert resp["ok"] is True
        assert resp["result"] == {"active": False, "bars": []}
    finally:
        sock.close()


def test_operation_progress_serializes_live_bars(fx):
    # get_operation_progress returns live (token, ProgressBarModel) pairs; the
    # wire layer reads the model's methods (computed at serialization time).
    import time

    from zcu_tools.gui.app.main.pbar_host import ProgressBarModel

    t = time.monotonic()
    m1 = ProgressBarModel(label="Rounds", total=100, start_time=t)
    m1.set_n(23)
    m2 = ProgressBarModel(label="Reps", total=500, start_time=t)
    m2.set_n(5)
    fx.ctrl.get_operation_progress = MagicMock(  # type: ignore[method-assign]
        return_value=((1, m1), (2, m2))
    )
    sock = open_client(fx.service.port)
    try:
        resp = call(sock, "operation.progress", {"operation_id": 7})
        assert resp["ok"] is True
        assert resp["result"]["active"] is True
        bars = {b["token"]: b for b in resp["result"]["bars"]}
        assert bars[1]["maximum"] == 100
        assert bars[1]["value"] == 23
        assert bars[1]["percent"] == 23.0
        assert bars[1]["n"] == 23 and bars[1]["total"] == 100
        assert "Rounds" in bars[1]["format"]
        assert bars[2]["percent"] == 1.0
    finally:
        sock.close()


def test_operation_progress_unknown_total_has_null_percent(fx):
    import time

    from zcu_tools.gui.app.main.pbar_host import ProgressBarModel

    m = ProgressBarModel(label="working", total=None, start_time=time.monotonic())
    fx.ctrl.get_operation_progress = MagicMock(  # type: ignore[method-assign]
        return_value=((1, m),)
    )
    sock = open_client(fx.service.port)
    try:
        resp = call(sock, "operation.progress", {"operation_id": 7})
        assert resp["ok"] is True
        assert resp["result"]["bars"][0]["percent"] is None
    finally:
        sock.close()


def test_device_setup_builds_request_from_live_info_and_updates(fx):
    fx.ctrl.get_device_info = MagicMock(  # type: ignore[method-assign]
        return_value=FakeDeviceInfo(address="none", value=0.0)
    )
    fx.ctrl.start_setup_device = MagicMock(return_value=7)  # type: ignore[method-assign]
    sock = open_client(fx.service.port)
    try:
        resp = call(sock, "device.setup", {"name": "bias", "updates": {"value": 1.5}})
        assert resp["ok"] is True
        req = fx.ctrl.start_setup_device.call_args.args[0]
        assert isinstance(req, SetupDeviceRequest)
        assert req.name == "bias"
        assert isinstance(req.info, FakeDeviceInfo)
        assert req.info.value == 1.5
        assert req.info.address == "none"
    finally:
        sock.close()


def test_device_setup_rejects_protected_info_update(fx):
    fx.ctrl.get_device_info = MagicMock(  # type: ignore[method-assign]
        return_value=FakeDeviceInfo(address="none", value=0.0)
    )
    sock = open_client(fx.service.port)
    try:
        resp = call(sock, "device.setup", {"name": "bias", "updates": {"type": "x"}})
        assert resp["ok"] is False
        assert resp["error"]["code"] == "invalid_params"
    finally:
        sock.close()


def test_device_setup_spec_lists_settable_fields_with_current(fx):
    fx.ctrl.get_device_info = MagicMock(  # type: ignore[method-assign]
        return_value=FakeDeviceInfo(address="none", value=2.5)
    )
    sock = open_client(fx.service.port)
    try:
        resp = call(sock, "device.setup_spec", {"name": "bias"})
        assert resp["ok"] is True
        fields = {f["name"]: f for f in resp["result"]["fields"]}
        # protected fields reported but settable=false
        assert fields["address"]["settable"] is False
        assert fields["type"]["settable"] is False
        # driver fields are settable, typed, carry the current value
        assert fields["value"]["settable"] is True
        assert fields["value"]["type"] == "float"
        assert fields["value"]["current"] == 2.5
        assert fields["rampstep"]["settable"] is True
    finally:
        sock.close()


def test_device_setup_spec_exposes_literal_choices(fx):
    # A driver with Literal enum fields (YOKO output/mode) → choices surfaced.
    fx.ctrl.get_device_info = MagicMock(  # type: ignore[method-assign]
        return_value=YOKOGS200Info(address="x", type="YOKOGS200")
    )
    sock = open_client(fx.service.port)
    try:
        resp = call(sock, "device.setup_spec", {"name": "flux"})
        assert resp["ok"] is True
        fields = {f["name"]: f for f in resp["result"]["fields"]}
        assert fields["output"]["type"] == "enum"
        assert fields["output"]["choices"] == ["on", "off"]
        assert fields["mode"]["choices"] == ["voltage", "current"]
    finally:
        sock.close()


def test_device_setup_spec_requires_live_info(fx):
    fx.ctrl.get_device_info = MagicMock(return_value=None)  # type: ignore[method-assign]
    sock = open_client(fx.service.port)
    try:
        resp = call(sock, "device.setup_spec", {"name": "ghost"})
        assert resp["ok"] is False
        assert resp["error"]["code"] == "precondition_failed"
    finally:
        sock.close()


# ---------------------------------------------------------------------------
# soc.info — read the connected SoC's hardware summary (wire v11)
# ---------------------------------------------------------------------------


def _install_real_mock_soccfg(fx) -> None:
    """Swap the fixture's MagicMock soccfg for a real QICK mock soccfg so
    description()/dump_cfg() return real content."""
    import dataclasses

    from zcu_tools.program.v2.mocksoc import make_mock_soccfg

    ctx = fx.state.exp_context
    fx.state.exp_context = dataclasses.replace(ctx, soccfg=make_mock_soccfg())


def test_soc_info_returns_description_and_cfg(fx):
    _install_real_mock_soccfg(fx)
    sock = open_client(fx.service.port)
    try:
        resp = call(sock, "soc.info")
        assert resp["ok"] is True
        result = resp["result"]
        assert "QICK" in result["description"]
        assert "signal generator" in result["description"]
        # structured cfg carries the DAC generators with their sample rate
        gens = result["cfg"]["gens"]
        assert gens and "fs" in gens[0]
        assert isinstance(result["is_mock"], bool)
    finally:
        sock.close()


def test_soc_info_requires_connected_soc(fx):
    import dataclasses

    ctx = fx.state.exp_context
    fx.state.exp_context = dataclasses.replace(ctx, soc=None, soccfg=None)
    sock = open_client(fx.service.port)
    try:
        resp = call(sock, "soc.info")
        assert resp["ok"] is False
        assert resp["error"]["code"] == "precondition_failed"
    finally:
        sock.close()


def test_device_mutation_error_path_is_precondition_failed(fx):
    sock = open_client(fx.service.port)
    try:
        resp = call(sock, "device.disconnect", {"name": "missing"})
        assert resp["ok"] is False
        assert resp["error"]["code"] == "precondition_failed"
    finally:
        sock.close()


def test_context_md_write_and_delete(fx):
    md = fx.state.exp_context.md
    sock = open_client(fx.service.port)
    try:
        resp = call(sock, "context.set_md_attr", {"key": "bias", "value": 0.25})
        assert resp["ok"] is True
        assert getattr(md, "bias") == 0.25

        resp = call(sock, "context.del_md_attr", {"key": "bias"}, rid="2")
        assert resp["ok"] is True
        assert not hasattr(md, "bias")
    finally:
        sock.close()


def test_context_ml_delete_delegates(fx):
    # ADR-0006: there is no raw-dict context.set_ml_* RPC anymore (ml entries are
    # built/edited via the editor session). Delete still delegates to the ctrl.
    fx.ctrl.del_ml_module = MagicMock()  # type: ignore[method-assign]
    fx.ctrl.del_ml_waveform = MagicMock()  # type: ignore[method-assign]
    sock = open_client(fx.service.port)
    try:
        assert call(sock, "context.del_ml_module", {"name": "m"})["ok"]
        fx.ctrl.del_ml_module.assert_called_once_with("m")

        assert call(sock, "context.del_ml_waveform", {"name": "w"}, rid="2")["ok"]
        fx.ctrl.del_ml_waveform.assert_called_once_with("w")
    finally:
        sock.close()


def test_mcp_tool_schemas_include_required_discovery_tools():
    expected = {
        "gui_adapter_list",
        "gui_connect_start",
        "gui_context_labels",
        "gui_context_active",
        "gui_context_use",
        "gui_context_new",
        "gui_save_data",
        "gui_save_image",
        "gui_device_connect",
        "gui_device_disconnect",
        "gui_device_setup",
        "gui_device_active_setup",
        "gui_device_active_operation",
        "gui_state_check",
    }
    assert expected <= set(TOOLS)
    for name, info in TOOLS.items():
        schema = info["inputSchema"]
        assert schema["type"] == "object", name
        for prop_name, prop_schema in schema.get("properties", {}).items():
            assert "type" in prop_schema, f"{name}.{prop_name}"
    assert TOOLS["gui_context_use"]["inputSchema"]["required"] == ["label"]
    assert TOOLS["gui_device_setup"]["inputSchema"]["required"] == [
        "name",
        "updates",
    ]


def _add_fake_tab(fx, tab_id: str) -> None:
    """Register a minimal Session so has_tab(tab_id) is True."""
    from zcu_tools.experiment.v2_gui.adapters.fake import FakeAdapter
    from zcu_tools.gui.app.main.state import Session

    adapter = FakeAdapter()
    cfg = adapter.make_default_cfg(fx.state.exp_context)
    fx.state.add_tab(
        tab_id, Session(adapter_name="fake", adapter=adapter, cfg_schema=cfg)
    )


def test_editor_set_field_blocked_while_owning_tab_runs(fx):
    """A tab cfg draft (editor session owned by tab_id) can't be edited while
    that tab runs — same guard the human gets via the disabled form (F11)."""
    tab_id = "tab-run"
    _add_fake_tab(fx, tab_id)
    cfg = fx.state.get_tab(tab_id).cfg_schema
    editor_id, _ = fx.ctrl.open_seeded_cfg_editor(cfg, gc=False, owner_key=tab_id)
    sock = open_client(fx.service.port)
    try:
        with patch.object(fx.ctrl, "get_running_tab_id", return_value=tab_id):
            resp = call(
                sock,
                "editor.set_field",
                {"editor_id": editor_id, "path": "reps", "value": 10},
            )
        assert resp["ok"] is False
        assert resp["error"]["code"] == "precondition_failed"
    finally:
        sock.close()


def test_tab_update_cfg_blocked_while_running(fx):
    tab_id = "tab-run"
    _add_fake_tab(fx, tab_id)
    sock = open_client(fx.service.port)
    try:
        with patch.object(fx.ctrl, "get_running_tab_id", return_value=tab_id):
            resp = call(sock, "tab.update_cfg", {"tab_id": tab_id, "raw": {}})
        assert resp["ok"] is False
        assert resp["error"]["code"] == "precondition_failed"
    finally:
        sock.close()


def test_mcp_wrappers_map_to_expected_rpc():
    # These three are GENERATED forwarders (not in _OVERRIDE_TOOLS): post-E4 they
    # capture the guarded send_gui_rpc as a closure (send_fn) at import time, so
    # patching mcp_server.send_gui_rpc no longer reaches them. Re-generate them
    # with a recording send_fn via the shared generate_tools — same projection
    # the real bridge builds — to assert the wrapper -> (method, params) mapping
    # without the guard mutating params.
    from zcu_tools.gui.app.main.services.remote.method_specs import METHOD_SPECS
    from zcu_tools.mcp.core.bridge import generate_tools
    from zcu_tools.mcp.measure import server as mcp_server

    calls: list[tuple[str, dict]] = []

    def fake_send(method: str, params: dict, timeout_seconds: float = 30.0) -> dict:
        del timeout_seconds
        calls.append((method, params))
        return {}

    tools = generate_tools(
        mcp_server._CONFIG, METHOD_SPECS, mcp_server._NON_GENERATED_METHODS, fake_send
    )

    tools["gui_context_use"]["handler"]({"label": "ctx1"})
    tools["gui_device_reconnect"]["handler"]({"name": "bias"})
    tools["gui_save_image"]["handler"]({"tab_id": "tab1", "image_path": "/tmp/a.png"})

    assert calls == [
        ("context.use", {"label": "ctx1"}),
        ("device.reconnect", {"name": "bias"}),
        ("save.image", {"tab_id": "tab1", "image_path": "/tmp/a.png"}),
    ]


def test_device_setup_wrapper_issues_setup_then_short_wait(monkeypatch):
    """gui_device_setup is not a 1:1 wrapper: it starts device.setup then waits
    briefly (operation.await) and reports a snapshot/handle (short-wait degrade)."""
    from zcu_tools.mcp.measure import server as mcp_server

    calls: list[tuple[str, dict]] = []

    def fake_send(method: str, params: dict, timeout_seconds: float = 30.0) -> dict:
        del timeout_seconds
        calls.append((method, params))
        return {}

    monkeypatch.setattr(mcp_server, "send_gui_rpc", fake_send)
    mcp_server.TOOLS["gui_device_setup"]["handler"](
        {"name": "bias", "updates": {"value": 1.0}}
    )

    # First call is always the start RPC with the agent's params verbatim.
    assert calls[0] == ("device.setup", {"name": "bias", "updates": {"value": 1.0}})


# ---------------------------------------------------------------------------
# Batch convenience tools (fan-out over single-field RPCs, fail-fast)
# ---------------------------------------------------------------------------


def test_set_fields_fans_out_in_order_and_returns_valid(monkeypatch):
    from zcu_tools.mcp.measure import server as mcp_server

    calls: list[tuple[str, dict]] = []

    def fake_send(method: str, params: dict, timeout_seconds: float = 30.0) -> dict:
        del timeout_seconds
        calls.append((method, params))
        return {"valid": True, "removed": [], "added": []}

    monkeypatch.setattr(mcp_server, "send_gui_rpc", fake_send)
    out = mcp_server.TOOLS["gui_editor_set_fields"]["handler"](
        {
            "editor_id": "ed1",
            "edits": [
                {"path": "reps", "value": 100},
                {"path": "sweep.gain.expts", "value": 5},
            ],
        }
    )

    # Each edit becomes one set_field in order; NO trailing editor.get (set does
    # not echo cfg content — that would force a lowering/eval pass).
    assert calls == [
        ("editor.set_field", {"editor_id": "ed1", "path": "reps", "value": 100}),
        (
            "editor.set_field",
            {"editor_id": "ed1", "path": "sweep.gain.expts", "value": 5},
        ),
    ]
    assert out["applied"] == 2
    assert out["valid"] is True
    assert "cfg" not in out


def test_set_fields_fail_fast_stops_and_reports_progress(monkeypatch):
    from zcu_tools.mcp.measure import server as mcp_server

    calls: list[str] = []

    def fake_send(method: str, params: dict, timeout_seconds: float = 30.0) -> dict:
        del timeout_seconds
        calls.append(params.get("path", method))
        if params.get("path") == "bad":
            raise RuntimeError("GUI Error (INVALID_PARAMS): unknown path 'bad'")
        return {}

    monkeypatch.setattr(mcp_server, "send_gui_rpc", fake_send)
    with pytest.raises(RuntimeError) as ei:
        mcp_server.TOOLS["gui_editor_set_fields"]["handler"](
            {
                "editor_id": "ed1",
                "edits": [
                    {"path": "ok", "value": 1},
                    {"path": "bad", "value": 2},
                    {"path": "never", "value": 3},
                ],
            }
        )

    # Stops at the failing edit; the third never fires; no editor.get on failure.
    assert calls == ["ok", "bad"]
    msg = str(ei.value)
    assert "edits[1]" in msg and "'bad'" in msg and "1 edit(s) already applied" in msg


def test_set_md_attrs_fans_out_in_order(monkeypatch):
    from zcu_tools.mcp.measure import server as mcp_server

    calls: list[tuple[str, dict]] = []

    def fake_send(method: str, params: dict, timeout_seconds: float = 30.0) -> dict:
        del timeout_seconds
        calls.append((method, params))
        return {}

    monkeypatch.setattr(mcp_server, "send_gui_rpc", fake_send)
    out = mcp_server.TOOLS["gui_context_set_md_attrs"]["handler"](
        {"attrs": [{"key": "r_f", "value": 5000.0}, {"key": "q_f", "value": 200.0}]}
    )

    assert calls == [
        ("context.set_md_attr", {"key": "r_f", "value": 5000.0}),
        ("context.set_md_attr", {"key": "q_f", "value": 200.0}),
    ]
    assert out["applied"] == 2


def test_batch_tools_reject_malformed_items_before_any_rpc(monkeypatch):
    from zcu_tools.mcp.measure import server as mcp_server

    calls: list[str] = []

    def fake_send(method: str, params: dict, timeout_seconds: float = 30.0) -> dict:
        del params, timeout_seconds
        calls.append(method)
        return {}

    monkeypatch.setattr(mcp_server, "send_gui_rpc", fake_send)

    # Missing 'value' in an edit / empty list — validated up front, no RPC fires.
    with pytest.raises(ValueError):
        mcp_server.TOOLS["gui_editor_set_fields"]["handler"](
            {"editor_id": "ed1", "edits": [{"path": "reps"}]}
        )
    with pytest.raises(ValueError):
        mcp_server.TOOLS["gui_context_set_md_attrs"]["handler"]({"attrs": []})
    assert calls == []


# ---------------------------------------------------------------------------
# Split startup + connect tools (generated from ParamSpec)
# ---------------------------------------------------------------------------


def test_startup_apply_optional_dirs_default_to_project_root(qapp):  # noqa: ARG001
    """Omitting result_dir/database_path fills the default per-qubit roots
    (<project_root>/result/<chip>/<qub>) via the RPC — anchored at the injected
    project root (the repo root), NOT cwd, so a .bat launcher that cd's into
    script/ still scopes defaults under the repo root. An agent gets a runnable
    project without knowing the path layout."""
    import os

    from ._helpers import Fixture

    root = os.path.join(os.sep, "tmp", "fake_repo_root")
    fx = Fixture(project_root=root)
    fx.start()
    try:
        captured = {}

        def _apply(req):
            captured["req"] = req
            return True

        fx.ctrl.apply_startup_project = MagicMock(side_effect=_apply)  # type: ignore[method-assign]
        sock = open_client(fx.service.port)
        try:
            resp = call(
                sock,
                "startup.apply",
                {"chip_name": "C", "qub_name": "Q", "res_name": "R"},
            )
            assert resp["ok"] is True
            req = captured["req"]
            assert req.chip_name == "C"
            # Anchored at the injected project root, NOT os.getcwd().
            assert req.result_dir == os.path.join(root, "result", "C", "Q")
            # database_path carries today's dated data folder (derive owns the date).
            from datetime import datetime

            yy, mm, dd = datetime.today().strftime("%Y-%m-%d").split("-")
            assert req.database_path == os.path.join(
                root, "Database", "C", "Q", yy, mm, f"Data_{mm}{dd}"
            )
        finally:
            sock.close()
    finally:
        fx.stop()


def test_startup_apply_missing_required_rejected(fx):
    sock = open_client(fx.service.port)
    try:
        resp = call(sock, "startup.apply", {"chip_name": "C", "qub_name": "Q"})
        assert resp["ok"] is False
        assert resp["error"]["code"] == "invalid_params"
    finally:
        sock.close()


def test_connect_start_remote_missing_ip_rejected(fx):
    sock = open_client(fx.service.port)
    try:
        resp = call(sock, "connect.start", {"kind": "remote"})
        assert resp["ok"] is False
        assert resp["error"]["code"] == "invalid_params"
    finally:
        sock.close()


def test_connect_start_mock_ok(fx):
    fx.ctrl.start_connect = MagicMock(return_value=3)  # type: ignore[method-assign]
    sock = open_client(fx.service.port)
    try:
        resp = call(sock, "connect.start", {"kind": "mock"})
        assert resp["ok"] is True
        fx.ctrl.start_connect.assert_called_once()
    finally:
        sock.close()


# ---------------------------------------------------------------------------
# Adapter spec queries (no tab needed)
# ---------------------------------------------------------------------------


def test_adapter_cfg_spec_lists_paths_without_tab(fx):
    sock = open_client(fx.service.port)
    try:
        resp = call(sock, "adapter.cfg_spec", {"adapter_name": "fake/freq"})
        assert resp["ok"] is True
        paths = {p["path"] for p in resp["result"]["paths"]}
        assert "sweep.freq.expts" in paths
        assert "reps" in paths
        # The simulated resonance moved to the adapter __init__ — not a cfg path.
        assert "model.freq" not in paths
        # a sweep edge carries integer/number type
        expts = next(
            p for p in resp["result"]["paths"] if p["path"] == "sweep.freq.expts"
        )
        assert expts["kind"] == "sweep_edge"
        assert expts["type"] == "integer"
    finally:
        sock.close()


def test_adapter_cfg_spec_lists_ref_only_not_variant_inner_fields(fx):
    """cfg_spec emits each ModuleRef's '.ref' selector + allowed choices, and
    does NOT descend into any variant's inner fields (no Cartesian-product
    blowup, no guessing the live default variant)."""
    sock = open_client(fx.service.port)
    try:
        resp = call(sock, "adapter.cfg_spec", {"adapter_name": "fake/freq"})
        by_path = {p["path"]: p for p in resp["result"]["paths"]}
        # The ref selector is present, carrying its allowed variant labels.
        assert "modules.readout.ref" in by_path
        ref = by_path["modules.readout.ref"]
        assert ref["kind"] == "moduleref_key"
        assert "Pulse Readout" in ref["choices"]
        assert "Direct Readout" in ref["choices"]
        # No variant inner fields leak through — neither label-keyed nor
        # chosen-style. The only modules.readout.* path is the ref itself.
        readout_paths = [p for p in by_path if p.startswith("modules.readout.")]
        assert readout_paths == ["modules.readout.ref"]
    finally:
        sock.close()


def test_adapter_cfg_spec_unknown_rejected(fx):
    sock = open_client(fx.service.port)
    try:
        resp = call(sock, "adapter.cfg_spec", {"adapter_name": "nope/nope"})
        assert resp["ok"] is False
        assert resp["error"]["code"] == "invalid_params"
    finally:
        sock.close()


def test_adapter_analyze_spec_reflects_params(fx):
    sock = open_client(fx.service.port)
    try:
        resp = call(sock, "adapter.analyze_spec", {"adapter_name": "fake/freq"})
        assert resp["ok"] is True
        params = {p["name"]: p for p in resp["result"]["params"]}
        assert params["model_type"]["choices"] == ["hm", "t", "auto"]
        assert params["fit_bg_slope"]["type"] == "bool"
    finally:
        sock.close()


def test_adapter_analyze_spec_empty_for_no_analysis(fx):
    # onetone/power_dep declares supports_analysis=False.
    sock = open_client(fx.service.port)
    try:
        resp = call(sock, "adapter.analyze_spec", {"adapter_name": "onetone/power_dep"})
        assert resp["ok"] is True
        assert resp["result"]["params"] == []
    finally:
        sock.close()


def test_adapter_guide_returns_five_fields(fx):
    sock = open_client(fx.service.port)
    try:
        resp = call(sock, "adapter.guide", {"adapter_name": "fake/freq"})
        assert resp["ok"] is True
        guide = resp["result"]["guide"]
        assert set(guide) == {
            "behavior",
            "expects_md",
            "expects_ml",
            "typical_writeback",
            "recommended",
        }
        # fake/freq overrides the guide — every field is non-empty prose.
        assert all(isinstance(v, str) and v for v in guide.values())
        assert "simulated" in guide["behavior"].lower()
        # expects_md names the concrete md keys it reads (orientation, not a
        # contract — but the guide is supposed to surface real key names).
        assert "r_f" in guide["expects_md"]
        assert "res_ch" in guide["expects_md"]
        assert "ro_waveform" in guide["expects_ml"]
    finally:
        sock.close()


def test_adapter_guide_unknown_rejected(fx):
    sock = open_client(fx.service.port)
    try:
        resp = call(sock, "adapter.guide", {"adapter_name": "nope/nope"})
        assert resp["ok"] is False
        assert resp["error"]["code"] == "invalid_params"
    finally:
        sock.close()


def test_base_adapter_guide_default_is_honest():
    # Every registered adapter now overrides guide(), so the honest default is
    # tested directly on BaseAdapter: an adapter with no guide says so plainly
    # rather than faking content.
    from zcu_tools.experiment.v2_gui.adapters.fake.stub import FakeAdapter

    guide = FakeAdapter.guide()
    assert guide.behavior == "(no guide written yet)"
    assert guide.expects_md == ""
    assert guide.recommended == ""


def test_every_registered_adapter_has_a_written_guide():
    # A new adapter that forgets to override guide() falls back to the honest
    # "(no guide written yet)" default — this test flags that so the gap is
    # caught at review time rather than shipping a blank Guide tab to users.
    from zcu_tools.experiment.v2_gui.registry import ADAPTERS

    missing = [
        name
        for name, cls in ADAPTERS.items()
        if cls.guide().behavior == "(no guide written yet)"
    ]
    assert not missing, f"adapters without a written guide: {missing}"


# ---------------------------------------------------------------------------
# Phase 120c-⑧ — run/analyze replies fold in figure_path (no extra screenshot
# call), saved to the cross-platform temp dir, base64-free.
# ---------------------------------------------------------------------------


def test_analyze_reply_includes_figure_path_when_figure_exists(monkeypatch):
    from zcu_tools.mcp.measure import server as mcp_server

    calls: list[tuple[str, dict]] = []

    def fake_send(method: str, params: dict, timeout_seconds: float = 30.0) -> dict:
        del timeout_seconds
        calls.append((method, params))
        if method == "tab.snapshot":
            return {"interaction": {"has_figure": True}}
        if method == "tab.figure_screenshot":
            return {"saved_to": params["out_path"]}
        return {}

    # analyze is async — gui_analyze starts it then awaits. Stub the await as a
    # completed operation so the synchronous-to-agent contract holds in the unit.
    monkeypatch.setattr(mcp_server, "send_gui_rpc", fake_send)
    monkeypatch.setattr(
        mcp_server,
        "_await_operation_by_key",
        lambda key, what, timeout: {"status": "finished"},
    )
    out = mcp_server.TOOLS["gui_analyze"]["handler"]({"tab_id": "fake-freq-1"})

    assert out["status"] == "finished"
    # figure_path points at a temp PNG keyed by tab_id (cross-platform tempdir).
    assert out["figure_path"].endswith("zcu_tools_figure_fake-freq-1.png")
    # analyze.start was issued, and the figure was rendered to that exact path.
    assert ("analyze.start", {"tab_id": "fake-freq-1"}) in calls
    shot = next(c for c in calls if c[0] == "tab.figure_screenshot")
    assert shot[1]["out_path"] == out["figure_path"]


def test_analyze_reply_omits_figure_path_when_no_figure(monkeypatch):
    from zcu_tools.mcp.measure import server as mcp_server

    def fake_send(method: str, params: dict, timeout_seconds: float = 30.0) -> dict:
        del timeout_seconds, params
        if method == "tab.snapshot":
            return {"interaction": {"has_figure": False}}
        return {}

    monkeypatch.setattr(mcp_server, "send_gui_rpc", fake_send)
    monkeypatch.setattr(
        mcp_server,
        "_await_operation_by_key",
        lambda key, what, timeout: {"status": "finished"},
    )
    out = mcp_server.TOOLS["gui_analyze"]["handler"]({"tab_id": "t1"})

    assert out["status"] == "finished"
    assert "figure_path" not in out


# ---------------------------------------------------------------------------
# Phase 120c-1 — non-blocking per-domain poll (replaces watching events).
# gui_run_poll maps a zero-timeout await onto finished/running/failed/
# no_operation, keyed on the semantic name (tab_id), no operation_id exposed.
# ---------------------------------------------------------------------------


def test_run_poll_running_when_op_in_flight(monkeypatch):
    from zcu_tools.mcp.measure import server as mcp_server

    monkeypatch.setattr(mcp_server, "_OP_BY_KEY", {"tab:t1": 7})

    def fake_send(method, params, timeout_seconds=30.0):
        del timeout_seconds
        if method == "operation.await":
            # zero-timeout await of an unfinished op -> wire TIMEOUT
            raise RuntimeError("GUI Error (timeout): not done")
        return {}

    monkeypatch.setattr(mcp_server, "send_gui_rpc", fake_send)
    out = mcp_server.TOOLS["gui_run_poll"]["handler"]({"tab_id": "t1"})
    assert out["status"] == "running"


def test_run_poll_finished_attaches_figure(monkeypatch):
    from zcu_tools.mcp.measure import server as mcp_server

    monkeypatch.setattr(mcp_server, "_OP_BY_KEY", {"tab:t1": 7})

    def fake_send(method, params, timeout_seconds=30.0):
        del timeout_seconds
        if method == "operation.await":
            return {"status": "finished"}
        if method == "tab.snapshot":
            return {"interaction": {"has_figure": True}}
        if method == "tab.figure_screenshot":
            return {"saved_to": params["out_path"]}
        return {}

    monkeypatch.setattr(mcp_server, "send_gui_rpc", fake_send)
    out = mcp_server.TOOLS["gui_run_poll"]["handler"]({"tab_id": "t1"})
    assert out["status"] == "finished"
    assert out["figure_path"].endswith("zcu_tools_figure_t1.png")


def test_run_poll_failed_does_not_raise(monkeypatch):
    from zcu_tools.mcp.measure import server as mcp_server

    monkeypatch.setattr(mcp_server, "_OP_BY_KEY", {"tab:t1": 7})

    def fake_send(method, params, timeout_seconds=30.0):
        del timeout_seconds, params
        if method == "operation.await":
            raise RuntimeError("GUI Error (precondition_failed): run blew up")
        return {}

    monkeypatch.setattr(mcp_server, "send_gui_rpc", fake_send)
    out = mcp_server.TOOLS["gui_run_poll"]["handler"]({"tab_id": "t1"})
    assert out["status"] == "failed"


def test_run_poll_no_operation_when_untracked(monkeypatch):
    from zcu_tools.mcp.measure import server as mcp_server

    monkeypatch.setattr(mcp_server, "_OP_BY_KEY", {})
    out = mcp_server.TOOLS["gui_run_poll"]["handler"]({"tab_id": "t1"})
    assert out["status"] == "no_operation"


# ---------------------------------------------------------------------------
# Phase 120c-3 — guard stale error names which resources moved, in agent
# language (no version numbers, no raw keyspace).
# ---------------------------------------------------------------------------


def test_describe_stale_keys_translates_to_agent_language():
    from zcu_tools.mcp.measure import server as mcp_server

    out = mcp_server._describe_stale_keys(
        ["context", "soc", "tab:abc123:cfg", "device:flux", "devices:__set__"]
    )
    assert "the active context (md/ml)" in out
    assert "the SoC connection" in out
    assert "this tab's cfg" in out
    assert "device 'flux'" in out
    assert "the set of devices (one added/removed)" in out
    # No raw key (uuid / keyspace) leaks through for known shapes.
    assert not any("abc123" in s for s in out)


def test_stale_error_message_names_changed_resources(monkeypatch):
    from zcu_tools.mcp.measure import server as mcp_server

    def fake_raw(method, params, timeout_seconds=30.0):
        del method, params, timeout_seconds
        return {
            "ok": False,
            "error": {
                "code": "precondition_failed",
                "message": "stale",
                "reason": "stale_version",
                "data": {"stale": ["context", "tab:t:cfg"]},
            },
        }

    monkeypatch.setattr(mcp_server._BRIDGE, "send_rpc_raw", fake_raw)
    monkeypatch.setattr(mcp_server, "_refresh_versions", lambda: None)
    with pytest.raises(RuntimeError) as ei:
        mcp_server.send_gui_rpc("run.start", {"tab_id": "t"})
    msg = str(ei.value)
    assert "the active context (md/ml)" in msg
    assert "this tab's cfg" in msg
