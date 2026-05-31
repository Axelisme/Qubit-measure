"""Full MCP-facing remote toolchain coverage."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from zcu_tools.device.fake import FakeDeviceInfo
from zcu_tools.gui.event_bus import DeviceSetupChangedPayload, GuiEvent
from zcu_tools.gui.services.device import DeviceSetupSnapshot, SetupDeviceRequest
from zcu_tools.gui.services.device_progress import ProgressEntrySnapshot
from zcu_tools.gui.services.remote.dispatch import METHOD_REGISTRY
from zcu_tools.gui.services.remote.mcp_server import TOOLS

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


def test_device_setup_changed_push_requeries_active_setup(fx):
    sock = open_client(fx.service.port)
    try:
        call(sock, "events.subscribe", {"events": ["device_setup_changed"]})
        fx.bus.emit(
            GuiEvent.DEVICE_SETUP_CHANGED,
            DeviceSetupChangedPayload(active_setup=None),
        )
        msg = recv_push(sock, "device_setup_changed")
        assert msg["payload"] == {"requery": ["device.active_setup"]}
    finally:
        sock.close()


def test_device_active_setup_serializes_scalar_progress(fx):
    progress = (
        ProgressEntrySnapshot(token=1, format="Ramp %v/%m", maximum=10, value=3),
    )
    fx.ctrl.get_active_device_setup = MagicMock(  # type: ignore[method-assign]
        return_value=DeviceSetupSnapshot(device_name="bias", progress=progress)
    )
    sock = open_client(fx.service.port)
    try:
        resp = call(sock, "device.active_setup")
        assert resp["ok"] is True
        assert resp["result"]["active_setup"] == {
            "device_name": "bias",
            "progress": [
                {"token": 1, "format": "Ramp %v/%m", "maximum": 10, "value": 3}
            ],
        }
    finally:
        sock.close()


def test_run_progress_idle_returns_empty(fx):
    fx.ctrl.get_run_progress = MagicMock(return_value=())  # type: ignore[method-assign]
    sock = open_client(fx.service.port)
    try:
        resp = call(sock, "run.progress")
        assert resp["ok"] is True
        assert resp["result"] == {"active": False, "bars": []}
    finally:
        sock.close()


def test_run_progress_serializes_scalar_bars(fx):
    bars = (
        ProgressEntrySnapshot(token=1, format="Rounds 23/100", maximum=100, value=23),
        ProgressEntrySnapshot(token=2, format="Reps 5/500", maximum=500, value=5),
    )
    fx.ctrl.get_run_progress = MagicMock(return_value=bars)  # type: ignore[method-assign]
    sock = open_client(fx.service.port)
    try:
        resp = call(sock, "run.progress")
        assert resp["ok"] is True
        assert resp["result"] == {
            "active": True,
            "bars": [
                {
                    "token": 1,
                    "format": "Rounds 23/100",
                    "maximum": 100,
                    "value": 23,
                    "percent": 23.0,
                },
                {
                    "token": 2,
                    "format": "Reps 5/500",
                    "maximum": 500,
                    "value": 5,
                    "percent": 1.0,
                },
            ],
        }
    finally:
        sock.close()


def test_run_progress_unknown_total_has_null_percent(fx):
    bars = (ProgressEntrySnapshot(token=1, format="working", maximum=0, value=0),)
    fx.ctrl.get_run_progress = MagicMock(return_value=bars)  # type: ignore[method-assign]
    sock = open_client(fx.service.port)
    try:
        resp = call(sock, "run.progress")
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
    # ADR-0011: there is no raw-dict context.set_ml_* RPC anymore (ml entries are
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
        "gui_session_persist",
        "gui_session_restore",
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
    """Register a minimal TabState so has_tab(tab_id) is True."""
    from zcu_tools.experiment.v2_gui.adapters.fake import FakeAdapter
    from zcu_tools.gui.state import TabState

    adapter = FakeAdapter()
    cfg = adapter.make_default_cfg(fx.state.exp_context)
    fx.state.add_tab(
        tab_id, TabState(adapter_name="fake", adapter=adapter, cfg_schema=cfg)
    )


def test_cfg_set_field_blocked_while_running(fx):
    tab_id = "tab-run"
    _add_fake_tab(fx, tab_id)
    sock = open_client(fx.service.port)
    try:
        with patch.object(fx.ctrl, "get_running_tab_id", return_value=tab_id):
            resp = call(
                sock, "cfg.set_field", {"tab_id": tab_id, "path": "reps", "value": 10}
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


def test_mcp_wrappers_map_to_expected_rpc(monkeypatch):
    from zcu_tools.gui.services.remote import mcp_server

    calls: list[tuple[str, dict]] = []

    def fake_send(method: str, params: dict, timeout_seconds: float = 30.0) -> dict:
        del timeout_seconds
        calls.append((method, params))
        return {}

    monkeypatch.setattr(mcp_server, "send_gui_rpc", fake_send)

    mcp_server.TOOLS["gui_context_use"]["handler"]({"label": "ctx1"})
    mcp_server.TOOLS["gui_device_reconnect"]["handler"]({"name": "bias"})
    mcp_server.TOOLS["gui_save_image"]["handler"](
        {"tab_id": "tab1", "image_path": "/tmp/a.png"}
    )

    assert calls == [
        ("context.use", {"label": "ctx1"}),
        ("device.reconnect", {"name": "bias"}),
        ("save.image", {"tab_id": "tab1", "image_path": "/tmp/a.png"}),
    ]


def test_device_setup_wrapper_issues_setup_then_short_wait(monkeypatch):
    """gui_device_setup is not a 1:1 wrapper: it starts device.setup then waits
    briefly (operation.await) and reports a snapshot/handle (short-wait degrade)."""
    from zcu_tools.gui.services.remote import mcp_server

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
# Split startup + connect tools (generated from ParamSpec)
# ---------------------------------------------------------------------------


def test_startup_apply_optional_dirs_default_to_empty(fx):
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
        assert req.result_dir == ""  # omitted -> empty (DRAFT context)
        assert req.database_path == ""
    finally:
        sock.close()


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
        assert "model.freq" in paths
        # a sweep edge carries integer/number type
        expts = next(
            p for p in resp["result"]["paths"] if p["path"] == "sweep.freq.expts"
        )
        assert expts["kind"] == "sweep_edge"
        assert expts["type"] == "integer"
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
