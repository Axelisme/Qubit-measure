"""Full MCP-facing remote toolchain coverage."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from zcu_tools.device.fake import FakeDeviceInfo
from zcu_tools.device.yoko import YOKOGS200Info
from zcu_tools.gui.app.main.services.remote.dispatch import METHOD_REGISTRY
from zcu_tools.gui.session.events import (
    DeviceSetupFinishedPayload,
    DeviceSetupStartedPayload,
)
from zcu_tools.gui.session.ports import OperationKind
from zcu_tools.gui.session.services.device import (
    ActiveDeviceOperation,
    DeviceSnapshot,
    DeviceStatus,
    SetupDeviceRequest,
)
from zcu_tools.mcp.measure.server import TOOLS

from ._helpers import Fixture, call, open_client, recv_push


@pytest.fixture()
def fx(qapp):  # noqa: ARG001
    f = Fixture()
    f.start()
    yield f
    f.stop()


def test_event_requery_hints_point_to_registered_methods():
    assert "device.active_operations" in METHOD_REGISTRY
    assert "context.md_get_attr" in METHOD_REGISTRY
    assert "context.ml_get" in METHOD_REGISTRY


def test_device_setup_started_and_finished_push(fx):
    sock = open_client(fx.service.port)
    try:
        call(
            sock,
            "events.subscribe",
            {"events": ["device_setup_started", "device_setup_finished"]},
        )
        fx.bus.emit(DeviceSetupStartedPayload(name="bias"))
        started = recv_push(sock, "device_setup_started")
        assert started["payload"] == {"name": "bias"}

        fx.bus.emit(
            DeviceSetupFinishedPayload(name="bias", outcome="finished"),
        )
        finished = recv_push(sock, "device_setup_finished")
        assert finished["payload"]["name"] == "bias"
        assert finished["payload"]["outcome"] == "finished"
    finally:
        sock.close()


def test_device_active_operations_enumerate_with_kind(fx):
    # Phase C: active_operations lists EVERY in-flight op, each tagged with its
    # kind + device_name so the agent knows which device and which operation.
    fx.ctrl.get_active_device_operations = MagicMock(  # type: ignore[method-assign]
        return_value=(
            ActiveDeviceOperation(
                device_name="bias",
                kind=OperationKind.DEVICE_SETUP,
                snapshot=DeviceSnapshot(
                    name="bias",
                    type_name="YOKOGS200",
                    address="addr1",
                    status=DeviceStatus.SETTING_UP,
                ),
            ),
            ActiveDeviceOperation(
                device_name="flux",
                kind=OperationKind.DEVICE_CONNECT,
                snapshot=DeviceSnapshot(
                    name="flux",
                    type_name="FakeDevice",
                    address="addr2",
                    status=DeviceStatus.CONNECTING,
                ),
            ),
        )
    )
    sock = open_client(fx.service.port)
    try:
        resp = call(sock, "device.active_operations")
        assert resp["ok"] is True
        assert resp["result"]["active_operations"] == [
            {
                "device_name": "bias",
                "kind": "device_setup",
                "name": "bias",
                "type_name": "YOKOGS200",
                "address": "addr1",
                "status": DeviceStatus.SETTING_UP.value,
                "error": None,
            },
            {
                "device_name": "flux",
                "kind": "device_connect",
                "name": "flux",
                "type_name": "FakeDevice",
                "address": "addr2",
                "status": DeviceStatus.CONNECTING.value,
                "error": None,
            },
        ]
    finally:
        sock.close()


def test_operation_progress_device_setup_bars(fx):
    # operation.progress covers device setup too: live (token, ProgressBarModel).
    import time

    from zcu_tools.gui.session.pbar_host import ProgressBarModel

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

    from zcu_tools.gui.session.pbar_host import ProgressBarModel

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

    from zcu_tools.gui.session.pbar_host import ProgressBarModel

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
    describe_soc()'s field access and dump_cfg() return real content."""
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
        # compact describe_soc table: header + per-channel generator/readout rows
        assert "QICK running on" in result["description"]
        assert "Generators" in result["description"]
        assert "Readouts" in result["description"]
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
        resp = call(sock, "context.md_set_attr", {"key": "bias", "value": 0.25})
        assert resp["ok"] is True
        assert getattr(md, "bias") == 0.25

        resp = call(sock, "context.md_del_attr", {"key": "bias"}, rid="2")
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
        assert call(sock, "context.ml_del_module", {"name": "m"})["ok"]
        fx.ctrl.del_ml_module.assert_called_once_with("m")

        assert call(sock, "context.ml_del_waveform", {"name": "w"}, rid="2")["ok"]
        fx.ctrl.del_ml_waveform.assert_called_once_with("w")
    finally:
        sock.close()


def test_save_post_image_delegates_to_controller(fx):
    """tab.save_post_image mirrors tab.save_image but targets the post-analysis figure;
    it delegates to Controller.save_post_image and returns the written path."""
    fx.ctrl.save_post_image = MagicMock(  # type: ignore[method-assign]
        return_value="/tmp/post.png"
    )
    sock = open_client(fx.service.port)
    try:
        resp = call(
            sock,
            "tab.save_post_image",
            {"tab_id": "tab1", "image_path": "/tmp/post.png"},
        )
        assert resp["ok"] is True
        assert resp["result"] == {"image_path": "/tmp/post.png"}
        fx.ctrl.save_post_image.assert_called_once_with("tab1", "/tmp/post.png")
    finally:
        sock.close()


def test_mcp_tool_schemas_include_required_discovery_tools():
    expected = {
        "gui_adapter_list",
        "gui_soc_connect",
        "gui_context_labels",
        "gui_context_active",
        "gui_context_use",
        "gui_context_new",
        "gui_tab_save_data",
        "gui_tab_save_image",
        "gui_device_connect",
        "gui_device_disconnect",
        "gui_device_setup",
        "gui_device_active_operations",
        "gui_state_check",
    }
    assert expected <= set(TOOLS)
    for name, info in TOOLS.items():
        schema = info["inputSchema"]
        assert schema["type"] == "object", name
        for prop_name, prop_schema in schema.get("properties", {}).items():
            # Most props pin a concrete "type"; a JsonType.JSON param renders
            # UNTYPED on purpose (any JSON value — omitting "type" stops the MCP
            # client coercing a number against a "string" member). So only assert
            # that a present "type" is a string, allowing the untyped JSON form.
            if "type" in prop_schema:
                assert isinstance(prop_schema["type"], str), f"{name}.{prop_name}"
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
    tools["gui_tab_save_image"]["handler"](
        {"tab_id": "tab1", "image_path": "/tmp/a.png"}
    )

    assert calls == [
        ("context.use", {"label": "ctx1"}),
        ("device.reconnect", {"name": "bias"}),
        ("tab.save_image", {"tab_id": "tab1", "image_path": "/tmp/a.png"}),
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
    out = mcp_server.TOOLS["gui_context_md_set_attrs"]["handler"](
        {"attrs": [{"key": "r_f", "value": 5000.0}, {"key": "q_f", "value": 200.0}]}
    )

    assert calls == [
        ("context.md_set_attr", {"key": "r_f", "value": 5000.0}),
        ("context.md_set_attr", {"key": "q_f", "value": 200.0}),
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
        mcp_server.TOOLS["gui_context_md_set_attrs"]["handler"]({"attrs": []})
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


def test_soc_connect_remote_missing_ip_rejected(fx):
    sock = open_client(fx.service.port)
    try:
        resp = call(sock, "soc.connect", {"kind": "remote"})
        assert resp["ok"] is False
        assert resp["error"]["code"] == "invalid_params"
    finally:
        sock.close()


def test_soc_connect_mock_returns_summary_directly(fx):
    # soc.connect is synchronous: the handler calls Controller.connect_sync then
    # reads back the SoC summary via get_soc_info — no operation_id / handle.
    fx.ctrl.connect_sync = MagicMock()  # type: ignore[method-assign]
    fx.ctrl.get_soc_info = MagicMock(  # type: ignore[method-assign]
        return_value={
            "description": "QICK mock board",
            "cfg": {},
            "is_mock": True,
        }
    )
    sock = open_client(fx.service.port)
    try:
        resp = call(sock, "soc.connect", {"kind": "mock"})
        assert resp["ok"] is True
        fx.ctrl.connect_sync.assert_called_once()
        # The reply carries the soc summary directly (description + is_mock), with
        # no operation_id (connect is no longer an async handle).
        assert resp["result"]["soc"] == {
            "description": "QICK mock board",
            "is_mock": True,
        }
        assert "operation_id" not in resp["result"]
    finally:
        sock.close()


# ---------------------------------------------------------------------------
# Adapter spec queries (no tab needed)
# ---------------------------------------------------------------------------


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
# Figure/screenshot consolidation (WIRE 24) — run/analyze replies NO LONGER fold
# figure_path; looking at a plot is a separate gui_tab_get_current_figure call.
# ---------------------------------------------------------------------------


def test_get_current_figure_omitted_out_path_writes_temp_file(monkeypatch):
    """Omitting out_path must drive the wire in out_path mode (synthesised temp
    path) and return {saved_to, bytes} with NO inline base64."""
    from tempfile import gettempdir

    from zcu_tools.mcp.measure import server as mcp_server

    calls: list[tuple[str, dict]] = []

    def fake_send(method: str, params: dict, timeout_seconds: float = 30.0) -> dict:
        del timeout_seconds
        calls.append((method, params))
        # Mirror the wire out_path branch: write nothing here, just echo saved_to.
        return {"bytes": 1234, "saved_to": params["out_path"]}

    monkeypatch.setattr(mcp_server, "send_gui_rpc", fake_send)
    out = mcp_server.TOOLS["gui_tab_get_current_figure"]["handler"](
        {"tab_id": "fake-freq-1"}
    )

    expected_path = str(Path(gettempdir()) / "measure_fig_fake-freq-1.png")
    assert out == {"bytes": 1234, "saved_to": expected_path}
    assert "png_b64" not in out
    # The convenience layer forwarded an out_path so the wire never returns base64.
    assert calls == [
        ("tab.get_current_figure", {"tab_id": "fake-freq-1", "out_path": expected_path})
    ]


def test_get_current_figure_explicit_out_path_is_forwarded(monkeypatch):
    from zcu_tools.mcp.measure import server as mcp_server

    calls: list[tuple[str, dict]] = []

    def fake_send(method: str, params: dict, timeout_seconds: float = 30.0) -> dict:
        del timeout_seconds
        calls.append((method, params))
        return {"bytes": 1234, "saved_to": params["out_path"]}

    monkeypatch.setattr(mcp_server, "send_gui_rpc", fake_send)
    out = mcp_server.TOOLS["gui_tab_get_current_figure"]["handler"](
        {"tab_id": "t1", "out_path": "/tmp/custom.png"}
    )

    assert out == {"bytes": 1234, "saved_to": "/tmp/custom.png"}
    assert calls == [
        ("tab.get_current_figure", {"tab_id": "t1", "out_path": "/tmp/custom.png"})
    ]


def test_screenshot_dialog_omitted_out_path_writes_temp_file(monkeypatch):
    """target=<dialog name> still works: decode + write a per-dialog temp PNG and
    return {saved_to, bytes} with NO inline base64 (the old dialog behaviour)."""
    import base64
    from tempfile import gettempdir

    from zcu_tools.mcp.measure import server as mcp_server

    raw = b"PNGDATA"
    calls: list[tuple[str, dict]] = []

    def fake_send(method: str, params: dict, timeout_seconds: float = 30.0) -> dict:
        del timeout_seconds
        calls.append((method, params))
        return {"png_b64": base64.b64encode(raw).decode("ascii"), "bytes": len(raw)}

    monkeypatch.setattr(mcp_server, "send_gui_rpc", fake_send)
    out = mcp_server.TOOLS["gui_debug_screenshot"]["handler"]({"target": "setup"})

    expected_path = str(Path(gettempdir()) / "measure_dialog_setup.png")
    assert out == {"bytes": len(raw), "saved_to": expected_path}
    assert "png_b64" not in out
    assert Path(expected_path).read_bytes() == raw
    # A dialog target forwards dialog.screenshot with only 'name'.
    assert calls == [("dialog.screenshot", {"name": "setup"})]


def test_screenshot_dialog_explicit_out_path(monkeypatch, tmp_path):
    import base64

    from zcu_tools.mcp.measure import server as mcp_server

    raw = b"X"
    target = tmp_path / "shot.png"

    def fake_send(method: str, params: dict, timeout_seconds: float = 30.0) -> dict:
        del timeout_seconds, method, params
        return {"png_b64": base64.b64encode(raw).decode("ascii"), "bytes": len(raw)}

    monkeypatch.setattr(mcp_server, "send_gui_rpc", fake_send)
    out = mcp_server.TOOLS["gui_debug_screenshot"]["handler"](
        {"target": "device", "out_path": str(target)}
    )

    assert out == {"bytes": len(raw), "saved_to": str(target)}
    assert target.read_bytes() == raw


def test_screenshot_window_writes_window_png(monkeypatch):
    """target='window' drives the view.screenshot wire method and writes the single
    measure_window.png temp file (the whole-window grab; no per-name)."""
    import base64
    from tempfile import gettempdir

    from zcu_tools.mcp.measure import server as mcp_server

    raw = b"WINDOWPNG"
    calls: list[tuple[str, dict]] = []

    def fake_send(method: str, params: dict, timeout_seconds: float = 30.0) -> dict:
        del timeout_seconds
        calls.append((method, params))
        return {"png_b64": base64.b64encode(raw).decode("ascii"), "bytes": len(raw)}

    monkeypatch.setattr(mcp_server, "send_gui_rpc", fake_send)
    out = mcp_server.TOOLS["gui_debug_screenshot"]["handler"]({"target": "window"})

    expected_path = str(Path(gettempdir()) / "measure_window.png")
    assert out == {"bytes": len(raw), "saved_to": expected_path}
    assert "png_b64" not in out
    assert Path(expected_path).read_bytes() == raw
    # The window target forwards view.screenshot with NO params.
    assert calls == [("view.screenshot", {})]


def test_debug_versions_dumps_resource_table(monkeypatch):
    """gui_debug_versions returns the full resources.versions table verbatim."""
    from zcu_tools.mcp.measure import server as mcp_server

    table = {"context": 3, "tab:t1:cfg": 7, "soc": 1}

    def fake_send(method: str, params: dict, timeout_seconds: float = 30.0) -> dict:
        del params, timeout_seconds
        assert method == "resources.versions"
        return {"versions": table}

    monkeypatch.setattr(mcp_server, "send_gui_rpc", fake_send)
    out = mcp_server.TOOLS["gui_debug_versions"]["handler"]({})
    assert out == {"versions": table}


def test_debug_operations_dumps_op_map_and_device_ops(monkeypatch):
    """gui_debug_operations returns the mcp-side _OP_BY_KEY map plus the wire
    device.active_operations list."""
    from zcu_tools.mcp.measure import server as mcp_server

    device_ops = [{"device_name": "flux", "kind": "device_setup", "name": "flux"}]

    def fake_send(method: str, params: dict, timeout_seconds: float = 30.0) -> dict:
        del params, timeout_seconds
        assert method == "device.active_operations"
        return {"active_operations": device_ops}

    monkeypatch.setattr(mcp_server, "send_gui_rpc", fake_send)
    monkeypatch.setattr(mcp_server, "_OP_BY_KEY", {"tab:t1": 42, "device:flux": 7})
    out = mcp_server.TOOLS["gui_debug_operations"]["handler"]({})
    assert out == {
        "by_key": {"tab:t1": 42, "device:flux": 7},
        "device_active_operations": device_ops,
    }


def test_server_instructions_present_three_tiers():
    """_SERVER_INSTRUCTIONS groups the tools under the three tiers (light check)."""
    from zcu_tools.mcp.measure import server as mcp_server

    text = mcp_server._SERVER_INSTRUCTIONS
    assert "RECOMMENDED" in text
    assert "ON-DEMAND" in text
    assert "DEV" in text
    # All three DEV tools are named in the instructions.
    assert "gui_debug_screenshot" in text
    assert "gui_debug_versions" in text
    assert "gui_debug_operations" in text


def _overview_fake_send(*, has_soc: bool):
    """A send_gui_rpc stub answering the read RPCs _assemble_overview fans out
    over, parameterised by SoC presence (soc.info only valid while connected)."""
    flags = {
        "state.has_project": {"value": True},
        "state.has_context": {"value": True},
        "state.has_active_context": {"value": False},
        "state.has_soc": {"value": has_soc},
        "project.info": {
            "chip_name": "Q5_2D",
            "qub_name": "Q1",
            "res_name": "R1",
            "result_dir": "/r",
            "database_path": "/db",
        },
        "context.active": {"label": "default"},
        "soc.info": {"description": "desc", "is_mock": True, "cfg": {}},
        "run.running_tab": {"tab_id": "t2"},
        "view.snapshot": {"active_tab_id": "t1", "context_label": "default"},
        "tab.snapshot": {
            "tabs": [
                {
                    "tab_id": "t1",
                    "adapter_name": "Freq",
                    "interaction": {"is_running": False},
                },
                {
                    "tab_id": "t2",
                    "adapter_name": "Rabi",
                    "interaction": {"is_running": True},
                },
            ]
        },
    }

    def fake_send(method: str, params: dict, timeout_seconds: float = 30.0) -> dict:
        del params, timeout_seconds
        if method not in flags:
            raise AssertionError(f"unexpected overview RPC: {method}")
        return flags[method]

    return fake_send


def test_overview_assembles_from_read_rpcs_with_project(monkeypatch):
    """gui_overview packs state / project / context / soc / tabs / running_tab /
    active_tab from existing reads; with a project applied, project uses
    long keys {chip_name, qub_name, res_name} matching the wire shape."""
    from zcu_tools.mcp.measure import server as mcp_server

    monkeypatch.setattr(mcp_server, "send_gui_rpc", _overview_fake_send(has_soc=True))
    out = mcp_server.TOOLS["gui_overview"]["handler"]({})

    assert out == {
        "state": {
            "has_project": True,
            "has_context": True,
            "has_active_context": False,
            "has_soc": True,
        },
        "project": {"chip_name": "Q5_2D", "qub_name": "Q1", "res_name": "R1"},
        "context": "default",
        "soc": {"connected": True, "is_mock": True},
        "tabs": [
            {"tab_id": "t1", "adapter": "Freq", "is_running": False},
            {"tab_id": "t2", "adapter": "Rabi", "is_running": True},
        ],
        "running_tab": "t2",
        "active_tab": "t1",
    }


def test_overview_project_is_null_when_no_project(monkeypatch):
    """project.info fast-fails no_project without a project, so the overview
    reports project=None and never queries project.info."""
    from zcu_tools.mcp.measure import server as mcp_server

    queried: list[str] = []
    base = _overview_fake_send(has_soc=True)

    def tracking(method: str, params: dict, timeout_seconds: float = 30.0) -> dict:
        queried.append(method)
        if method == "state.has_project":
            return {"value": False}
        return base(method, params, timeout_seconds)

    monkeypatch.setattr(mcp_server, "send_gui_rpc", tracking)
    out = mcp_server.TOOLS["gui_overview"]["handler"]({})

    assert out["project"] is None
    assert "project.info" not in queried


def test_overview_skips_soc_info_when_not_connected(monkeypatch):
    """soc.info fast-fails without a SoC, so is_mock stays None and soc.info is
    never queried when has_soc is False."""
    from zcu_tools.mcp.measure import server as mcp_server

    queried: list[str] = []
    base = _overview_fake_send(has_soc=False)

    def tracking(method: str, params: dict, timeout_seconds: float = 30.0) -> dict:
        queried.append(method)
        return base(method, params, timeout_seconds)

    monkeypatch.setattr(mcp_server, "send_gui_rpc", tracking)
    out = mcp_server.TOOLS["gui_overview"]["handler"]({})

    assert out["soc"] == {"connected": False, "is_mock": None}
    assert "soc.info" not in queried


def test_connect_folds_overview_into_reply(monkeypatch):
    """gui_connect returns {note, overview} so attaching alone gives the picture."""
    from zcu_tools.mcp.measure import server as mcp_server

    monkeypatch.setattr(mcp_server, "resolve_connect_port", lambda cfg, req: 8765)
    monkeypatch.setattr(mcp_server._BRIDGE, "connect", lambda port, token=None: "ok")
    monkeypatch.setattr(mcp_server, "_assemble_overview", lambda: {"sentinel": True})

    out = mcp_server.TOOLS["gui_connect"]["handler"]({})
    assert out == {"note": "ok", "overview": {"sentinel": True}}


def test_analyze_settled_returns_summary_and_figure(monkeypatch):
    """gui_tab_analyze FINISHED reply carries the fit summary AND the figure (analyze's
    OWN visual result — MCP 46). The writeback preview is NOT folded here (that fold
    lives in gui_tab_stage3 only)."""
    from tempfile import gettempdir

    from zcu_tools.mcp.measure import server as mcp_server

    # No tracked op for this key -> _start_op_with_short_wait settles synchronously.
    mcp_server._OP_BY_KEY.pop("analyze:fake-freq-1", None)
    calls: list[tuple[str, dict]] = []

    def fake_send(method: str, params: dict, timeout_seconds: float = 30.0) -> dict:
        del timeout_seconds
        calls.append((method, params))
        if method == "tab.get_analyze_result":
            return {"summary": {"t1": 5.0}}
        if method == "tab.get_current_figure":
            return {"bytes": 9, "saved_to": params["out_path"]}
        return {}

    monkeypatch.setattr(mcp_server, "send_gui_rpc", fake_send)
    out = mcp_server.TOOLS["gui_tab_analyze"]["handler"]({"tab_id": "fake-freq-1"})

    assert out["status"] == "finished"
    assert out["summary"] == {"t1": 5.0}
    # MCP 46: figure is analyze's OWN visual result — folded on FINISHED FIT.
    assert out["figure"] == str(Path(gettempdir()) / "measure_fig_fake-freq-1.png")
    # writeback_preview stays in gui_tab_stage3 — never in the base tool.
    assert "writeback_preview" not in out
    assert ("tab.analyze", {"tab_id": "fake-freq-1"}) in calls
    assert any(c[0] == "tab.get_current_figure" for c in calls)
    assert not any(c[0] == "tab.writeback_preview" for c in calls)


def test_analyze_degrades_to_pending_when_not_settled(monkeypatch):
    """An INTERACTIVE analysis never settles in the short wait -> pending.
    A pending reply must NOT include 'figure' (nothing settled yet — MCP 46)."""
    from zcu_tools.mcp.measure import server as mcp_server

    monkeypatch.setattr(mcp_server, "_OP_BY_KEY", {"analyze:t1": 9})

    def fake_send(method, params, timeout_seconds=30.0):
        del timeout_seconds, params
        if method == "operation.await":
            raise RuntimeError("GUI Error (timeout): user still picking")
        return {}

    monkeypatch.setattr(mcp_server, "send_gui_rpc", fake_send)
    out = mcp_server.TOOLS["gui_tab_analyze"]["handler"]({"tab_id": "t1"})
    assert out["status"] == "pending"
    # _fold_finished_figure is a no-op on non-finished status; figure must be absent.
    assert "figure" not in out


def test_analyze_poll_running_then_finished(monkeypatch):
    from zcu_tools.mcp.measure import server as mcp_server

    monkeypatch.setattr(mcp_server, "_OP_BY_KEY", {"analyze:t1": 9})

    # Still picking -> running.
    def picking(method, params, timeout_seconds=30.0):
        del timeout_seconds, params
        if method == "operation.await":
            raise RuntimeError("GUI Error (timeout): not done")
        return {}

    monkeypatch.setattr(mcp_server, "send_gui_rpc", picking)
    assert (
        mcp_server.TOOLS["gui_tab_analyze_poll"]["handler"]({"tab_id": "t1"})["status"]
        == "running"
    )

    # User clicked Done -> finished. No figure_path folded (consolidation WIRE 24).
    def done(method, params, timeout_seconds=30.0):
        del timeout_seconds, params
        if method == "operation.await":
            return {"status": "finished"}
        return {}

    monkeypatch.setattr(mcp_server, "send_gui_rpc", done)
    out = mcp_server.TOOLS["gui_tab_analyze_poll"]["handler"]({"tab_id": "t1"})
    assert out["status"] == "finished"
    assert "figure_path" not in out


# ---------------------------------------------------------------------------
# Phase 120c-1 — non-blocking per-domain poll (replaces watching events).
# gui_tab_run_poll maps a zero-timeout await onto finished/running/failed/
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
    out = mcp_server.TOOLS["gui_tab_run_poll"]["handler"]({"tab_id": "t1"})
    assert out["status"] == "running"


def test_run_poll_failed_does_not_raise(monkeypatch):
    from zcu_tools.mcp.measure import server as mcp_server

    monkeypatch.setattr(mcp_server, "_OP_BY_KEY", {"tab:t1": 7})

    def fake_send(method, params, timeout_seconds=30.0):
        del timeout_seconds, params
        if method == "operation.await":
            raise RuntimeError("GUI Error (precondition_failed): run blew up")
        return {}

    monkeypatch.setattr(mcp_server, "send_gui_rpc", fake_send)
    out = mcp_server.TOOLS["gui_tab_run_poll"]["handler"]({"tab_id": "t1"})
    assert out["status"] == "failed"


def test_run_poll_no_operation_when_untracked(monkeypatch):
    from zcu_tools.mcp.measure import server as mcp_server

    monkeypatch.setattr(mcp_server, "_OP_BY_KEY", {})
    out = mcp_server.TOOLS["gui_tab_run_poll"]["handler"]({"tab_id": "t1"})
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
    # send_gui_rpc lazily auto-connects on the first call; here we test the
    # stale-message translation on an already-connected bridge, so short-circuit
    # _ensure_connected by reporting a live socket.
    monkeypatch.setattr(
        type(mcp_server._BRIDGE), "is_connected", property(lambda _self: True)
    )
    with pytest.raises(RuntimeError) as ei:
        mcp_server.send_gui_rpc("tab.run_start", {"tab_id": "t"})
    msg = str(ei.value)
    assert "the active context (md/ml)" in msg
    assert "this tab's cfg" in msg


# ---------------------------------------------------------------------------
# MCP 45: gui_tab_new is PURE — reverted to the auto-generated tab.new forwarder
# (returns just {tab_id}); the fan-out + guide fold moved to gui_tab_stage1.
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
        mcp_server._CONFIG, METHOD_SPECS, mcp_server._NON_GENERATED_METHODS, fake_send
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
    mcp_server._OP_BY_KEY.pop("tab:rt-1", None)
    calls: list[str] = []

    def fake_send(method: str, params: dict, timeout_seconds: float = 30.0) -> dict:
        del timeout_seconds
        calls.append(method)
        if method == "tab.snapshot":
            return {"interaction": {"has_run_result": True}}
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

    monkeypatch.setattr(mcp_server, "_OP_BY_KEY", {"tab:slow-1": 5})

    def fake_send(method, params, timeout_seconds=30.0):
        del timeout_seconds, params
        if method == "operation.await":
            raise RuntimeError("GUI Error (timeout): still running")
        return {}

    monkeypatch.setattr(mcp_server, "send_gui_rpc", fake_send)
    out = mcp_server.TOOLS["gui_tab_run_start"]["handler"]({"tab_id": "slow-1"})

    assert out["status"] == "pending"
    assert "figure" not in out


def test_run_wait_finished_carries_figure(monkeypatch):
    """gui_tab_run_wait FINISHED reply includes 'figure' — the run plot rendered to a
    temp PNG (the run's OWN visual result, MCP 46). Non-finished statuses (timed_out,
    cancelled, no_operation) must NOT include 'figure'."""
    from tempfile import gettempdir

    from zcu_tools.mcp.measure import server as mcp_server

    monkeypatch.setattr(mcp_server, "_OP_BY_KEY", {"tab:wt-1": 3})

    def fake_send(method, params, timeout_seconds=30.0):
        del timeout_seconds
        if method == "operation.await":
            return {"status": "finished", "reason": "completed"}
        if method == "tab.get_current_figure":
            return {"bytes": 9, "saved_to": params["out_path"]}
        return {}

    monkeypatch.setattr(mcp_server, "send_gui_rpc", fake_send)
    out = mcp_server.TOOLS["gui_tab_run_wait"]["handler"]({"tab_id": "wt-1"})

    assert out["status"] == "finished"
    assert out["figure"] == str(Path(gettempdir()) / "measure_fig_wt-1.png")


def test_fold_finished_figure_finished_folds_pending_does_not(monkeypatch):
    """The figure-fold helper renders + folds 'figure' on a FINISHED reply and is
    a no-op on a pending one (used by gui_tab_stage2 / gui_tab_stage3)."""
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
# MCP 44 Phase ④: gui_tab_writeback_apply gains optional save_data (save_data now in gui_tab_stage4)
# ---------------------------------------------------------------------------


def test_writeback_apply_is_pure_generated_forwarder():
    """gui_tab_writeback_apply forwards tab.writeback_apply and returns ONLY its result
    ({applied_ids}) — it no longer takes save_data nor chains tab.save_data (that moved
    to gui_tab_stage4).

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
        mcp_server._CONFIG, METHOD_SPECS, mcp_server._NON_GENERATED_METHODS, fake_send
    )
    out = tools["gui_tab_writeback_apply"]["handler"]({"tab_id": "t1"})

    assert out == {"applied_ids": ["md-0", "ml-1"]}
    assert calls == [("tab.writeback_apply", {"tab_id": "t1"})]


# ---------------------------------------------------------------------------
# MCP 45: gui_tab_run_poll is PURE — no figure fold; returns just the poll status.
# ---------------------------------------------------------------------------


def test_run_poll_finished_carries_figure(monkeypatch):
    """A 'finished' gui_tab_run_poll reply includes 'figure' — the run plot rendered
    to a temp PNG (the run's OWN visual result, MCP 46)."""
    from tempfile import gettempdir

    from zcu_tools.mcp.measure import server as mcp_server

    monkeypatch.setattr(mcp_server, "_OP_BY_KEY", {"tab:t1": 7})
    calls: list[str] = []

    def fake_send(method, params, timeout_seconds=30.0):
        del timeout_seconds
        calls.append(method)
        if method == "operation.await":
            return {"status": "finished"}
        if method == "tab.get_current_figure":
            return {"bytes": 9, "saved_to": params["out_path"]}
        return {}

    monkeypatch.setattr(mcp_server, "send_gui_rpc", fake_send)
    out = mcp_server.TOOLS["gui_tab_run_poll"]["handler"]({"tab_id": "t1"})
    assert out["status"] == "finished"
    # MCP 46: figure is the run's OWN visual result — present on FINISHED poll.
    assert out["figure"] == str(Path(gettempdir()) / "measure_fig_t1.png")
    assert "tab.get_current_figure" in calls


# ---------------------------------------------------------------------------
# MCP 45 Phase ②: gui_tab_stage2(tab_id, edits) configures + runs an existing tab
# (no tab creation, no guide fold) and folds the figure + analyze-params
# ---------------------------------------------------------------------------


def _run_stage_fake_send(calls: list[tuple[str, dict]]):
    """A send_gui_rpc stub covering every RPC gui_tab_stage2 fans out over.

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
            # has_run_result drives the run-finished tab summary.
            return {
                "editor_id": "stage-ed",
                "interaction": {"is_running": False, "has_run_result": True},
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


def test_run_stage2_configures_runs_and_stops_before_analyze(monkeypatch):
    """gui_tab_stage2 operates on the given tab_id: tab.set_cfg then tab.run_start,
    NEVER creating a tab and NEVER calling analyze. A finished reply carries the
    figure (from gui_tab_run_start's own FINISHED fold) + the stage-specific
    analyze-params spec."""
    from zcu_tools.mcp.measure import server as mcp_server

    mcp_server._OP_BY_KEY.pop("tab:stage-tab", None)
    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(mcp_server, "send_gui_rpc", _run_stage_fake_send(calls))

    out = mcp_server.TOOLS["gui_tab_stage2"]["handler"](
        {
            "tab_id": "stage-tab",
            "edits": {"reps": 100, "sweep.gain.expts": 5},
        }
    )

    methods = [m for m, _ in calls]
    # No tab creation (the tab already exists); edits via tab.set_cfg precede tab.run_start.
    assert "tab.new" not in methods
    assert methods.index("tab.set_cfg") < methods.index("tab.run_start")
    assert "tab.analyze" not in methods
    # tab.snapshot is still called by gui_tab_run_start's finished-reply fold
    # (_run_tab_summary); stage2 no longer calls it for editor_id resolution.

    # Finished run reply carries the folded figure AND the analyze-params spec.
    assert out["status"] == "finished"
    assert "figure" in out
    assert out["figure"].endswith("measure_fig_stage-tab.png")
    assert out["analyze_params"] == {"smooth": 1}
    assert ("tab.get_analyze_params", {"tab_id": "stage-tab"}) in calls


def test_run_stage2_does_not_double_fold_figure(monkeypatch):
    """gui_tab_stage2 must NOT call tab.get_current_figure a second time —
    the figure arrives already folded inside gui_tab_run_start's FINISHED reply (MCP 46).
    Exactly one tab.get_current_figure call is expected."""
    from zcu_tools.mcp.measure import server as mcp_server

    mcp_server._OP_BY_KEY.pop("tab:stage-tab", None)
    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(mcp_server, "send_gui_rpc", _run_stage_fake_send(calls))

    out = mcp_server.TOOLS["gui_tab_stage2"]["handler"]({"tab_id": "stage-tab"})

    assert out["status"] == "finished"
    assert "figure" in out
    figure_calls = [m for m, _ in calls if m == "tab.get_current_figure"]
    # Exactly one render: from gui_tab_run_start's fold — not a second from stage2.
    assert len(figure_calls) == 1


def test_run_stage2_edits_numbers_stay_numbers(monkeypatch):
    """The {path: value} map is forwarded to tab.set_cfg as a list of {path, value}
    objects; numeric values reach the wire as numbers (NOT stringified)."""
    from zcu_tools.mcp.measure import server as mcp_server

    mcp_server._OP_BY_KEY.pop("tab:stage-tab", None)
    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(mcp_server, "send_gui_rpc", _run_stage_fake_send(calls))

    mcp_server.TOOLS["gui_tab_stage2"]["handler"](
        {"tab_id": "stage-tab", "edits": {"reps": 100, "gain": 0.2}}
    )

    # Stage2 sends one tab.set_cfg call with the batch edits.
    set_cfg_calls = [params for method, params in calls if method == "tab.set_cfg"]
    assert len(set_cfg_calls) == 1
    edits = {e["path"]: e["value"] for e in set_cfg_calls[0]["edits"]}
    assert edits == {"reps": 100, "gain": 0.2}
    assert isinstance(edits["reps"], int)
    assert isinstance(edits["gain"], float)


def test_run_stage2_without_edits_runs_current_cfg(monkeypatch):
    """Omitting 'edits' runs the tab's current cfg — no editor.set_field fires."""
    from zcu_tools.mcp.measure import server as mcp_server

    mcp_server._OP_BY_KEY.pop("tab:stage-tab", None)
    calls: list[tuple[str, dict]] = []
    monkeypatch.setattr(mcp_server, "send_gui_rpc", _run_stage_fake_send(calls))

    out = mcp_server.TOOLS["gui_tab_stage2"]["handler"]({"tab_id": "stage-tab"})

    assert "tab.set_cfg" not in [m for m, _ in calls]
    assert out["status"] == "finished"
    assert out["analyze_params"] == {"smooth": 1}


def test_run_stage2_pending_run_omits_analyze_params(monkeypatch):
    """A slow run degrades to {status:'pending'} and must NOT fold analyze_params
    (nothing settled to analyze yet)."""
    from zcu_tools.mcp.measure import server as mcp_server

    monkeypatch.setattr(mcp_server, "_OP_BY_KEY", {"tab:stage-tab": 5})
    calls: list[str] = []

    def fake_send(method, params, timeout_seconds=30.0):
        del timeout_seconds, params
        calls.append(method)
        if method == "operation.await":
            raise RuntimeError("GUI Error (timeout): still running")
        return {}

    monkeypatch.setattr(mcp_server, "send_gui_rpc", fake_send)
    out = mcp_server.TOOLS["gui_tab_stage2"]["handler"]({"tab_id": "stage-tab"})

    assert out["status"] == "pending"
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
# MCP 45 Phase ①: gui_tab_stage1 creates a tab + ALWAYS folds the editing context
# + the adapter guide (no first-use gating — you call stage1 to get a tab + guide)
# ---------------------------------------------------------------------------


def test_run_stage1_creates_tab_and_folds_context_and_guide(monkeypatch):
    """gui_tab_stage1 creates the tab, fans out the two editing-context reads
    (snapshot for editor_id, tab.get_cfg for the settable cfg tree) and ALWAYS
    folds the adapter guide into one reply. The cfg tree already carries
    the current values."""
    from zcu_tools.mcp.measure import server as mcp_server

    calls: list[tuple[str, dict]] = []
    tree = {"reps": 100, "sweep": {"freq": {"start": 1.0}}}

    def fake_send(method: str, params: dict, timeout_seconds: float = 30.0) -> dict:
        del timeout_seconds
        calls.append((method, dict(params)))
        return {
            "tab.new": {"tab_id": "tw-1"},
            "tab.snapshot": {"editor_id": "ed-tw-1", "interaction": {}},
            "tab.get_cfg": {"tree": tree},
            "adapter.guide": {"guide": {"behavior": "measures X"}},
        }[method]

    monkeypatch.setattr(mcp_server, "send_gui_rpc", fake_send)
    out = mcp_server.TOOLS["gui_tab_stage1"]["handler"]({"adapter_name": "fake/freq"})

    # tab.new first (adapter_name verbatim), then the two reads keyed by the new
    # tab_id, then the adapter.guide fetch (always — no first-use gating).
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


def test_run_stage1_dedupes_guide_per_adapter(monkeypatch):
    """The adapter guide rides the FIRST gui_tab_stage1 for an adapter in a
    session; a repeat call for the same adapter omits it and sets
    guide_omitted=True instead (the guide is static, so re-sending it each call
    is wasted tokens)."""
    from zcu_tools.mcp.measure import server as mcp_server

    monkeypatch.setattr(mcp_server, "_GUIDE_SENT", set())

    def fake_send(method: str, params: dict, timeout_seconds: float = 30.0) -> dict:
        del params, timeout_seconds
        return {
            "tab.new": {"tab_id": "tw-1"},
            "tab.snapshot": {"editor_id": "ed-1", "interaction": {}},
            "tab.get_cfg": {"tree": {"reps": 1}},
            "adapter.guide": {"guide": {"behavior": "measures X"}},
        }[method]

    monkeypatch.setattr(mcp_server, "send_gui_rpc", fake_send)

    first = mcp_server.TOOLS["gui_tab_stage1"]["handler"]({"adapter_name": "amp_rabi"})
    second = mcp_server.TOOLS["gui_tab_stage1"]["handler"]({"adapter_name": "amp_rabi"})
    assert first["guide"] == {"behavior": "measures X"}
    assert "guide_omitted" not in first
    assert "guide" not in second
    assert second["guide_omitted"] is True


# ---------------------------------------------------------------------------
# MCP 45 Phase ③: gui_tab_stage3 analyzes + folds summary + figure + writeback
# ---------------------------------------------------------------------------


def test_run_stage3_analyzes_and_folds_figure_and_writeback(monkeypatch):
    """gui_tab_stage3 runs gui_tab_analyze, then folds the writeback preview onto a
    FINISHED FIT reply. 'figure' comes from gui_tab_analyze's own FINISHED fold (MCP 46),
    not a separate stage3 fold; 'writeback_preview' is the stage-specific addition."""
    from tempfile import gettempdir

    from zcu_tools.mcp.measure import server as mcp_server

    # No tracked op -> the analyze short-wait settles synchronously (fast FIT).
    mcp_server._OP_BY_KEY.pop("analyze:az-1", None)
    calls: list[tuple[str, dict]] = []

    def fake_send(method: str, params: dict, timeout_seconds: float = 30.0) -> dict:
        del timeout_seconds
        calls.append((method, dict(params)))
        if method == "tab.get_analyze_result":
            return {"summary": {"t1": 12.3}}
        if method == "tab.get_current_figure":
            return {"bytes": 9, "saved_to": params["out_path"]}
        if method == "tab.writeback_preview":
            return {"items": [{"id": "md-0", "target_name": "q_f"}]}
        return {}

    monkeypatch.setattr(mcp_server, "send_gui_rpc", fake_send)
    out = mcp_server.TOOLS["gui_tab_stage3"]["handler"]({"tab_id": "az-1"})

    assert out["status"] == "finished"
    assert out["summary"] == {"t1": 12.3}
    assert out["figure"] == str(Path(gettempdir()) / "measure_fig_az-1.png")
    assert out["writeback_preview"] == [{"id": "md-0", "target_name": "q_f"}]
    assert ("tab.analyze", {"tab_id": "az-1"}) in calls


def test_run_stage3_does_not_double_fold_figure(monkeypatch):
    """gui_tab_stage3 must NOT call tab.get_current_figure a second time —
    the figure arrives already folded inside gui_tab_analyze's FINISHED reply (MCP 46).
    Exactly one tab.get_current_figure call is expected."""
    from zcu_tools.mcp.measure import server as mcp_server

    mcp_server._OP_BY_KEY.pop("analyze:az-1", None)
    calls: list[tuple[str, dict]] = []

    def fake_send(method: str, params: dict, timeout_seconds: float = 30.0) -> dict:
        del timeout_seconds
        calls.append((method, dict(params)))
        if method == "tab.get_analyze_result":
            return {"summary": {"t1": 12.3}}
        if method == "tab.get_current_figure":
            return {"bytes": 9, "saved_to": params["out_path"]}
        if method == "tab.writeback_preview":
            return {"items": []}
        return {}

    monkeypatch.setattr(mcp_server, "send_gui_rpc", fake_send)
    out = mcp_server.TOOLS["gui_tab_stage3"]["handler"]({"tab_id": "az-1"})

    assert out["status"] == "finished"
    assert "figure" in out
    figure_calls = [m for m, _ in calls if m == "tab.get_current_figure"]
    # Exactly one render: from gui_tab_analyze's fold — not a second from stage3.
    assert len(figure_calls) == 1


def test_run_stage3_pending_interactive_omits_folds(monkeypatch):
    """An INTERACTIVE analyze degrades to pending -> gui_tab_stage3 folds NOTHING
    (no figure, no writeback preview)."""
    from zcu_tools.mcp.measure import server as mcp_server

    monkeypatch.setattr(mcp_server, "_OP_BY_KEY", {"analyze:az-1": 9})
    calls: list[str] = []

    def fake_send(method, params, timeout_seconds=30.0):
        del params, timeout_seconds
        calls.append(method)
        if method == "operation.await":
            raise RuntimeError("GUI Error (timeout): user still picking")
        return {}

    monkeypatch.setattr(mcp_server, "send_gui_rpc", fake_send)
    out = mcp_server.TOOLS["gui_tab_stage3"]["handler"]({"tab_id": "az-1"})

    assert out["status"] == "pending"
    assert "figure" not in out
    assert "writeback_preview" not in out
    assert "tab.get_current_figure" not in calls
    assert "tab.writeback_preview" not in calls


# ---------------------------------------------------------------------------
# MCP 45 Phase ④: gui_tab_stage4 applies the writeback, optionally saving the data
# ---------------------------------------------------------------------------


def test_run_stage4_default_applies_only(monkeypatch):
    """save_data defaults false: apply runs, tab.save_data does NOT, no data_path."""
    from zcu_tools.mcp.measure import server as mcp_server

    calls: list[str] = []

    def fake_send(method, params, timeout_seconds=30.0):
        del params, timeout_seconds
        calls.append(method)
        if method == "tab.writeback_apply":
            return {"applied_ids": ["md-0", "ml-1"]}
        return {}

    monkeypatch.setattr(mcp_server, "send_gui_rpc", fake_send)
    out = mcp_server.TOOLS["gui_tab_stage4"]["handler"]({"tab_id": "t1"})

    assert out == {"applied_ids": ["md-0", "ml-1"]}
    assert "tab.save_data" not in calls
    assert "data_path" not in out


def test_run_stage4_save_data_chains_save_and_folds_path(monkeypatch):
    """save_data=true: apply then tab.save_data, folding the resolved data_path."""
    from zcu_tools.mcp.measure import server as mcp_server

    calls: list[tuple[str, dict]] = []

    def fake_send(method, params, timeout_seconds=30.0):
        del timeout_seconds
        calls.append((method, dict(params)))
        if method == "tab.writeback_apply":
            return {"applied_ids": ["md-0"]}
        if method == "tab.save_data":
            return {"data_path": "/results/Q1/data_0001.hdf5"}
        return {}

    monkeypatch.setattr(mcp_server, "send_gui_rpc", fake_send)
    out = mcp_server.TOOLS["gui_tab_stage4"]["handler"](
        {"tab_id": "t1", "save_data": True}
    )

    methods = [m for m, _ in calls]
    # apply runs first, tab.save_data second (apply is committed before the save).
    assert methods.index("tab.writeback_apply") < methods.index("tab.save_data")
    assert out["applied_ids"] == ["md-0"]
    assert out["data_path"] == "/results/Q1/data_0001.hdf5"
    assert ("tab.save_data", {"tab_id": "t1"}) in calls


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
        mcp_server._CONFIG, METHOD_SPECS, mcp_server._NON_GENERATED_METHODS, fake_send
    )
    out = tools["gui_adapter_guide"]["handler"]({"adapter_name": "amp_rabi"})
    assert out == {"guide": {"behavior": "measures X"}}
    assert calls == [("adapter.guide", {"adapter_name": "amp_rabi"})]
