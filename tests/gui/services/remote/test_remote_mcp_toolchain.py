"""Full MCP-facing remote toolchain coverage."""

from __future__ import annotations

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


def test_save_data_delegates_to_save_control(fx):
    fx.ctrl.save_data = MagicMock(  # type: ignore[method-assign]
        side_effect=AssertionError("tab.save_data must use save_control")
    )
    fx.service.save_control.save_data = MagicMock(  # type: ignore[method-assign]
        return_value="/tmp/data.hdf5"
    )
    sock = open_client(fx.service.port)
    try:
        resp = call(
            sock,
            "tab.save_data",
            {"tab_id": "tab1", "data_path": "/tmp/data.h5", "comment": "note"},
        )
        assert resp["ok"] is True
        assert resp["result"] == {"data_path": "/tmp/data.hdf5"}
        fx.service.save_control.save_data.assert_called_once_with(
            "tab1", "/tmp/data.h5", comment="note"
        )
        fx.ctrl.save_data.assert_not_called()
    finally:
        sock.close()


def test_save_image_delegates_to_save_control(fx):
    fx.ctrl.save_image = MagicMock(  # type: ignore[method-assign]
        side_effect=AssertionError("tab.save_image must use save_control")
    )
    fx.service.save_control.save_image = MagicMock(  # type: ignore[method-assign]
        return_value="/tmp/image.png"
    )
    fx.service.save_control.save_post_image = MagicMock(  # type: ignore[method-assign]
        return_value="/tmp/post.png"
    )
    sock = open_client(fx.service.port)
    try:
        resp = call(
            sock,
            "tab.save_image",
            {"tab_id": "tab1", "subtab_id": "analysis", "image_path": "/tmp/image.png"},
        )
        assert resp["ok"] is True
        assert resp["result"]["image_path"] == "/tmp/image.png"
        fx.service.save_control.save_image.assert_called_once_with(
            "tab1", "/tmp/image.png"
        )
        fx.ctrl.save_image.assert_not_called()
        resp2 = call(
            sock,
            "tab.save_image",
            {
                "tab_id": "tab1",
                "subtab_id": "post_analysis",
                "image_path": "/tmp/post.png",
            },
        )
        assert resp2["ok"] is True
        assert resp2["result"]["image_path"] == "/tmp/post.png"
        fx.service.save_control.save_post_image.assert_called_once_with(
            "tab1", "/tmp/post.png"
        )
    finally:
        sock.close()


def test_save_post_image_delegates_to_save_control(fx):
    """tab.save_post_image wire method is removed (clean break); save_image with subtab post_analysis routes to save_post_image internally."""
    from zcu_tools.gui.app.main.services.remote.method_specs import METHOD_SPECS

    assert "tab.save_post_image" not in METHOD_SPECS
    assert "tab.save_post_image" not in [m for m in METHOD_SPECS]


def test_save_result_delegates_to_save_control(fx):
    """tab.save_result wire method is removed (clean break); use tab.save_data and tab.save_image separately."""
    from zcu_tools.gui.app.main.services.remote.method_specs import METHOD_SPECS

    assert "tab.save_result" not in METHOD_SPECS


def test_save_set_paths_delegates_to_save_control(fx):
    """tab.save_set_paths wire method is removed (no combined setter); use separate save_data/save_image."""
    from zcu_tools.gui.app.main.services.remote.method_specs import METHOD_SPECS

    assert "tab.save_set_paths" not in METHOD_SPECS


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
