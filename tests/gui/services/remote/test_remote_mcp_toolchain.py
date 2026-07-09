"""Full MCP-facing remote toolchain coverage."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
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
    ConnectDeviceRequest,
    DeviceEntry,
    DeviceSnapshot,
    DeviceStatus,
    DisconnectDeviceRequest,
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
    fx.service.device_control.get_active_device_operations = MagicMock(  # type: ignore[method-assign]
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
                token=11,
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
                token=12,
            ),
        )
    )
    sock = open_client(fx.service.port)
    try:
        resp = call(sock, "device.active_operations")
        assert resp["ok"] is True
        # P2: the reply key is 'operations'; each entry carries its 'handle' (the
        # op token) and drops the duplicate snapshot.name (device_name is the key).
        assert resp["result"]["operations"] == [
            {
                "handle": 11,
                "device_name": "bias",
                "kind": "device_setup",
                "type_name": "YOKOGS200",
                "address": "addr1",
                "status": DeviceStatus.SETTING_UP.value,
                "error": None,
            },
            {
                "handle": 12,
                "device_name": "flux",
                "kind": "device_connect",
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
        side_effect=AssertionError("operation.progress must use operation_control")
    )
    fx.service.operation_control.get_operation_progress = MagicMock(  # type: ignore[method-assign]
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
        fx.service.operation_control.get_operation_progress.assert_called_once_with(7)
        fx.ctrl.get_operation_progress.assert_not_called()
    finally:
        sock.close()


def test_operation_progress_idle_returns_empty(fx):
    fx.service.operation_control.get_operation_progress = MagicMock(return_value=())  # type: ignore[method-assign]
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
    fx.service.operation_control.get_operation_progress = MagicMock(  # type: ignore[method-assign]
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
    fx.service.operation_control.get_operation_progress = MagicMock(  # type: ignore[method-assign]
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
    fx.service.device_control.get_device_info = MagicMock(  # type: ignore[method-assign]
        return_value=FakeDeviceInfo(address="none", value=0.0)
    )
    fx.service.device_control.start_setup_device = MagicMock(return_value=7)  # type: ignore[method-assign]
    sock = open_client(fx.service.port)
    try:
        resp = call(sock, "device.setup", {"name": "bias", "updates": {"value": 1.5}})
        assert resp["ok"] is True
        req = fx.service.device_control.start_setup_device.call_args.args[0]
        assert isinstance(req, SetupDeviceRequest)
        assert req.name == "bias"
        assert isinstance(req.info, FakeDeviceInfo)
        assert req.info.value == 1.5
        assert req.info.address == "none"
    finally:
        sock.close()


def test_device_setup_rejects_protected_info_update(fx):
    fx.service.device_control.get_device_info = MagicMock(  # type: ignore[method-assign]
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
    fx.service.device_control.get_device_info = MagicMock(  # type: ignore[method-assign]
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
    fx.service.device_control.get_device_info = MagicMock(  # type: ignore[method-assign]
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
    fx.service.device_control.get_device_info = MagicMock(return_value=None)  # type: ignore[method-assign]
    sock = open_client(fx.service.port)
    try:
        resp = call(sock, "device.setup_spec", {"name": "ghost"})
        assert resp["ok"] is False
        assert resp["error"]["code"] == "precondition_failed"
    finally:
        sock.close()


def test_device_setup_spec_uses_device_control_facet(fx):
    ctrl_get = MagicMock(side_effect=AssertionError("broad controller used"))
    setattr(fx.ctrl, "get_device_info", ctrl_get)
    facet_get = MagicMock(return_value=FakeDeviceInfo(address="none", value=2.5))
    fx.service.device_control.get_device_info = facet_get  # type: ignore[method-assign]

    sock = open_client(fx.service.port)
    try:
        resp = call(sock, "device.setup_spec", {"name": "bias"})
        assert resp["ok"] is True
        fields = {f["name"]: f for f in resp["result"]["fields"]}
        assert fields["value"]["current"] == 2.5
        ctrl_get.assert_not_called()
        facet_get.assert_called_once_with("bias")
    finally:
        sock.close()


class _PoisonController:
    def __getattr__(self, name: str) -> object:
        raise AssertionError(f"broad controller used for {name}")


def _dispatch_with_device_control(
    method: str, params: dict[str, object], device_control: MagicMock
) -> dict[str, object]:
    adapter = SimpleNamespace(
        ctrl=_PoisonController(),
        device_control=device_control,
    )
    return dict(METHOD_REGISTRY[method].handler(adapter, params))


def test_device_handlers_dispatch_only_through_device_control_facet():
    dev = MagicMock()
    dev.start_connect_device.return_value = 101
    dev.start_disconnect_device.return_value = 102
    dev.start_reconnect_device.return_value = 103
    dev.start_setup_device.return_value = 104
    dev.get_device_info.return_value = FakeDeviceInfo(address="none", value=0.0)
    dev.get_active_device_operations.return_value = (
        ActiveDeviceOperation(
            device_name="bias",
            kind=OperationKind.DEVICE_SETUP,
            snapshot=DeviceSnapshot(
                name="bias",
                type_name="FakeDevice",
                address="none",
                status=DeviceStatus.SETTING_UP,
            ),
            token=105,
        ),
    )
    dev.list_devices.return_value = [
        DeviceEntry("bias", "FakeDevice", DeviceStatus.CONNECTED.value)
    ]
    dev.get_device_snapshot.return_value = DeviceSnapshot(
        name="bias",
        type_name="FakeDevice",
        address="none",
        status=DeviceStatus.CONNECTED,
        info=FakeDeviceInfo(address="none", value=1.0),
    )

    assert _dispatch_with_device_control(
        "device.connect",
        {
            "type_name": "FakeDevice",
            "name": "bias",
            "address": "none",
        },
        dev,
    ) == {"operation_id": 101}
    assert isinstance(dev.start_connect_device.call_args.args[0], ConnectDeviceRequest)

    assert _dispatch_with_device_control(
        "device.disconnect", {"name": "bias"}, dev
    ) == {"operation_id": 102}
    assert isinstance(
        dev.start_disconnect_device.call_args.args[0], DisconnectDeviceRequest
    )

    assert _dispatch_with_device_control("device.reconnect", {"name": "bias"}, dev) == {
        "operation_id": 103
    }
    assert _dispatch_with_device_control("device.forget", {"name": "bias"}, dev) == {
        "forgotten": "bias"
    }

    assert _dispatch_with_device_control(
        "device.setup", {"name": "bias", "updates": {"value": 2.0}}, dev
    ) == {"operation_id": 104}
    assert isinstance(dev.start_setup_device.call_args.args[0], SetupDeviceRequest)

    assert "fields" in _dispatch_with_device_control(
        "device.setup_spec", {"name": "bias"}, dev
    )
    assert _dispatch_with_device_control(
        "device.cancel_operation", {"name": "bias"}, dev
    ) == {"ok": True, "cancelled": True}
    assert _dispatch_with_device_control("device.active_operations", {}, dev) == {
        "operations": [
            {
                "handle": 105,
                "device_name": "bias",
                "kind": "device_setup",
                "type_name": "FakeDevice",
                "address": "none",
                "status": "setting_up",
                "error": None,
            }
        ]
    }
    assert _dispatch_with_device_control("device.list", {}, dev) == {
        "devices": [{"name": "bias", "type_name": "FakeDevice", "status": "connected"}]
    }
    snapshot = _dispatch_with_device_control("device.snapshot", {"name": "bias"}, dev)[
        "snapshot"
    ]
    assert isinstance(snapshot, dict)
    assert snapshot["info"]["value"] == 1.0


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
        # The structured cfg is opt-in now (include_cfg=true); the common path pays
        # nothing for it.
        resp = call(sock, "soc.info", {"include_cfg": True})
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
    # built/edited via the editor session). Delete still delegates to the context
    # control facet, not the wider controller.
    fx.service.context_control.del_ml_module = MagicMock()  # type: ignore[method-assign]
    fx.service.context_control.del_ml_waveform = MagicMock()  # type: ignore[method-assign]
    sock = open_client(fx.service.port)
    try:
        assert call(sock, "context.ml_del_module", {"name": "m"})["ok"]
        fx.service.context_control.del_ml_module.assert_called_once_with("m")

        assert call(sock, "context.ml_del_waveform", {"name": "w"}, rid="2")["ok"]
        fx.service.context_control.del_ml_waveform.assert_called_once_with("w")
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
    # P1 renamed/merged the context + state surface: context.active + context.labels
    # fold into gui_context_list; gui_context_new/_use -> gui_context_create/_switch;
    # gui_state_check is retired (the readiness flags live in gui_overview now). The
    # save tools merged into the single gui_tab_save (artifact + figure selectors).
    expected = {
        "gui_adapter_list",
        "gui_soc_connect",
        "gui_context_list",
        "gui_context_switch",
        "gui_context_create",
        "gui_tab_save",
        "gui_device_connect",
        "gui_device_disconnect",
        # P2 renamed gui_device_setup -> gui_device_apply and
        # gui_device_active_operations -> gui_device_list_operations.
        "gui_device_apply",
        "gui_device_list_operations",
        "gui_overview",
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
    assert TOOLS["gui_context_switch"]["inputSchema"]["required"] == ["label"]
    assert TOOLS["gui_device_apply"]["inputSchema"]["required"] == [
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
        mcp_server._CONFIG,
        METHOD_SPECS,
        mcp_server._MCP_EXPOSURE.non_generated_methods,
        fake_send,
    )

    # gui_context_switch / gui_device_snapshot are GENERATED forwarders. P2 retired
    # the standalone gui_device_reconnect tool (reconnect folded into
    # gui_device_connect, E4), so device.reconnect is wire-only — no generated
    # forwarder to assert here.
    tools["gui_context_switch"]["handler"]({"label": "ctx1"})
    tools["gui_device_snapshot"]["handler"]({"name": "bias"})

    assert calls == [
        ("context.use", {"label": "ctx1"}),
        ("device.snapshot", {"name": "bias"}),
    ]
    # The reconnect tool is gone (folded into gui_device_connect).
    assert "gui_device_reconnect" not in tools


def test_device_setup_wrapper_issues_setup_then_short_wait(monkeypatch):
    """gui_device_apply is not a 1:1 wrapper: it starts device.setup then waits
    briefly (operation.await) and reports a snapshot/handle (short-wait degrade).
    P2 renamed gui_device_setup -> gui_device_apply (the wire method stays
    device.setup)."""
    from zcu_tools.mcp.measure import server as mcp_server

    calls: list[tuple[str, dict]] = []

    def fake_send(method: str, params: dict, timeout_seconds: float = 30.0) -> dict:
        del timeout_seconds
        calls.append((method, params))
        return {}

    monkeypatch.setattr(mcp_server, "send_gui_rpc", fake_send)
    mcp_server.TOOLS["gui_device_apply"]["handler"](
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
    out = mcp_server.TOOLS["gui_editor_set"]["handler"](
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
        mcp_server.TOOLS["gui_editor_set"]["handler"](
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
    out = mcp_server.TOOLS["gui_context_md_write"]["handler"](
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
        mcp_server.TOOLS["gui_editor_set"]["handler"](
            {"editor_id": "ed1", "edits": [{"path": "reps"}]}
        )
    with pytest.raises(ValueError):
        mcp_server.TOOLS["gui_context_md_write"]["handler"]({"attrs": []})
    assert calls == []


# ---------------------------------------------------------------------------
# Split startup + connect tools (generated from ParamSpec)
# ---------------------------------------------------------------------------


def test_startup_apply_resolves_generated_scope_under_project_root(qapp, tmp_path):  # noqa: ARG001
    """Omitting scope_id uses the generated per-qubit result scope under the
    injected project root, not cwd. The RPC returns the resolved paths."""
    import os

    from ._helpers import Fixture

    root = str(tmp_path / "fake_repo_root")
    fx = Fixture(project_root=root)
    fx.start()
    try:
        sock = open_client(fx.service.port)
        try:
            resp = call(
                sock,
                "startup.apply",
                {"chip_name": "C", "qub_name": "Q", "res_name": "R"},
            )
            assert resp["ok"] is True
            result = resp["result"]
            assert result["chip_name"] == "C"
            # Anchored at the injected project root, NOT os.getcwd().
            assert result["result_dir"] == os.path.join(root, "result", "C", "Q")
            assert result["params_path"] == os.path.join(
                root, "result", "C", "Q", "params.json"
            )
            # database_path carries today's dated data folder (derive owns the date).
            from datetime import datetime

            yy, mm, dd = datetime.today().strftime("%Y-%m-%d").split("-")
            assert result["database_path"] == os.path.join(
                root, "Database", "C", "Q", yy, mm, f"Data_{mm}{dd}"
            )
            assert os.path.exists(result["params_path"])
        finally:
            sock.close()
    finally:
        fx.stop()


def test_result_scope_list_reports_discovered_params(qapp, tmp_path):  # noqa: ARG001
    import json

    from ._helpers import Fixture

    params_path = (
        tmp_path / "fake_repo_root" / "result" / "ChipA" / "Q1" / "params.json"
    )
    params_path.parent.mkdir(parents=True)
    params_path.write_text(
        json.dumps({"project": {"chip_name": "ChipA", "qubit_name": "Q1"}}),
        encoding="utf8",
    )
    fx = Fixture(project_root=str(tmp_path / "fake_repo_root"))
    fx.start()
    try:
        sock = open_client(fx.service.port)
        try:
            resp = call(sock, "result_scope.list", {})
            assert resp["ok"] is True
            scopes = resp["result"]["scopes"]
            assert len(scopes) == 1
            assert scopes[0]["chip_name"] == "ChipA"
            assert scopes[0]["qub_name"] == "Q1"
            assert scopes[0]["params_path"] == str(params_path.resolve())
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
    # Every registered adapter defines local guide_text, so the honest default is
    # tested directly on BaseAdapter: an adapter with no guide says so plainly
    # rather than faking content.
    from zcu_tools.experiment.v2_gui.adapters.fake.stub import FakeAdapter

    guide = FakeAdapter.guide()
    assert guide.behavior == "(no guide written yet)"
    assert guide.expects_md == ""
    assert guide.recommended == ""


def test_every_registered_adapter_has_a_written_guide():
    # A new adapter that forgets guide_text falls back to the honest
    # "(no guide written yet)" default — this test flags that so the gap is
    # caught at review time rather than shipping a blank Guide tab to users.
    from zcu_tools.experiment.v2_gui.registry import ADAPTERS

    missing = [
        name
        for name, cls in ADAPTERS.items()
        if cls.guide().behavior == "(no guide written yet)"
    ]
    assert not missing, f"adapters without a written guide: {missing}"


def test_analyze_settled_returns_summary_and_figure(monkeypatch):
    """gui_tab_analyze FINISHED reply carries the fit summary AND the figure (analyze's
    OWN visual result — MCP 46). The writeback preview is NOT folded here (that fold
    lives in gui_tab_analyze_review only)."""
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
    out = mcp_server.TOOLS["gui_tab_analyze_start"]["handler"](
        {"tab_id": "fake-freq-1"}
    )

    assert out["status"] == "finished"
    assert out["summary"] == {"t1": 5.0}
    # MCP 46: figure is analyze's OWN visual result — folded on FINISHED FIT.
    assert out["figure"] == str(Path(gettempdir()) / "measure_fig_fake-freq-1.png")
    # writeback_preview stays in gui_tab_analyze_review — never in the base tool.
    assert "writeback_preview" not in out
    assert ("tab.analyze", {"tab_id": "fake-freq-1"}) in calls
    assert any(c[0] == "tab.get_current_figure" for c in calls)
    assert not any(c[0] == "tab.writeback_preview" for c in calls)


def test_analyze_degrades_to_pending_when_not_settled(monkeypatch):
    """An INTERACTIVE analysis never settles in the short wait -> pending.
    A pending reply must NOT include 'figure' (nothing settled yet — MCP 46)."""
    from zcu_tools.mcp.measure import server as mcp_server

    mcp_server._SESSION.operation_handles.clear()
    mcp_server._SESSION.operation_handles.update({"analyze:t1": 9})

    def fake_send(method, params, timeout_seconds=30.0):
        del timeout_seconds, params
        if method == "operation.await":
            raise RuntimeError("GUI Error (timeout): user still picking")
        return {}

    monkeypatch.setattr(mcp_server, "send_gui_rpc", fake_send)
    out = mcp_server.TOOLS["gui_tab_analyze_start"]["handler"]({"tab_id": "t1"})
    assert out["status"] == "pending"
    # _fold_finished_figure is a no-op on non-finished status; figure must be absent.
    assert "figure" not in out


def test_analyze_poll_running_then_finished(monkeypatch):
    # P2 (ADR-0026 §8): the per-op gui_tab_analyze_poll is retired; the agent drives
    # the generic gui_op_poll with the handle the START reply folded (here handle=9).
    from zcu_tools.mcp.measure import server as mcp_server

    # Still picking -> running.
    def picking(method, params, timeout_seconds=30.0):
        del timeout_seconds, params
        if method == "operation.await":
            raise RuntimeError("GUI Error (timeout): not done")
        return {"active": False, "bars": []}

    monkeypatch.setattr(mcp_server, "send_gui_rpc", picking)
    assert (
        mcp_server.TOOLS["gui_op_poll"]["handler"]({"handle": 9})["status"] == "running"
    )

    # User clicked Done -> finished. The generic poll reports ONLY status — no
    # figure fold (the product is read from the START finished reply or a getter).
    def done(method, params, timeout_seconds=30.0):
        del timeout_seconds, params
        if method == "operation.await":
            return {"status": "finished"}
        return {}

    monkeypatch.setattr(mcp_server, "send_gui_rpc", done)
    out = mcp_server.TOOLS["gui_op_poll"]["handler"]({"handle": 9})
    assert out["status"] == "finished"
    assert "figure" not in out
    assert "figure_path" not in out


# ---------------------------------------------------------------------------
# Generic non-blocking poll (P2 / ADR-0026 §8): gui_op_poll(handle) maps a
# zero-timeout await onto finished/running/failed/cancelled, driven by the handle
# a START reply folded (the per-op gui_tab_run_poll is retired). no_operation is
# the helper's null-handle branch (the public tool always carries a handle).
# ---------------------------------------------------------------------------


def test_run_poll_running_when_op_in_flight(monkeypatch):
    from zcu_tools.mcp.measure import server as mcp_server

    def fake_send(method, params, timeout_seconds=30.0):
        del timeout_seconds, params
        if method == "operation.await":
            # zero-timeout await of an unfinished op -> wire TIMEOUT
            raise RuntimeError("GUI Error (timeout): not done")
        return {"active": False, "bars": []}

    monkeypatch.setattr(mcp_server, "send_gui_rpc", fake_send)
    out = mcp_server.TOOLS["gui_op_poll"]["handler"]({"handle": 7})
    assert out["status"] == "running"


def test_run_poll_failed_does_not_raise(monkeypatch):
    from zcu_tools.mcp.measure import server as mcp_server

    def fake_send(method, params, timeout_seconds=30.0):
        del timeout_seconds, params
        if method == "operation.await":
            raise RuntimeError("GUI Error (precondition_failed): run blew up")
        return {}

    monkeypatch.setattr(mcp_server, "send_gui_rpc", fake_send)
    out = mcp_server.TOOLS["gui_op_poll"]["handler"]({"handle": 7})
    assert out["status"] == "failed"


def test_run_poll_no_operation_when_handle_absent():
    # no_operation is the by-handle helper's null branch (a missing/already-reaped
    # handle); the public gui_op_poll tool always carries a handle, so the branch is
    # exercised directly on the helper.
    from zcu_tools.mcp.measure import server as mcp_server

    out = mcp_server._poll_operation_by_handle(None, "operation")
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
