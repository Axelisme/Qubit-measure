"""Full MCP-facing remote toolchain coverage."""

from __future__ import annotations

import threading
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from typing import ClassVar
from unittest.mock import MagicMock

import numpy as np
import pytest
from zcu_tools.device.fake import FakeDeviceInfo
from zcu_tools.device.yoko import YOKOGS200Info
from zcu_tools.experiment.context import RunContext
from zcu_tools.gui.app.measure.remote.dispatch import METHOD_REGISTRY
from zcu_tools.gui.expected_error import FailedPreconditionError
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
from zcu_tools.plotting.plots import Plots

from tests.gui.app.measure._synthetic import SampleAdapter, SampleParams

from ._helpers import Fixture, call, open_client, recv_push


@pytest.fixture()
def fx(qapp):
    f = Fixture()
    f.start()
    yield f
    f.stop()


def test_analyze_params_wire_describes_live_adapter_before_run(fx):
    tab_id = fx.ctrl.new_tab("fake")
    sock = open_client(fx.service.port)
    try:
        primary = call(sock, "tab.get_analyze_params", {"tab_id": tab_id})
        assert primary["ok"] is True
        assert primary["result"]["analyze_params"] is None
        assert primary["result"]["definitions"] == [
            {
                "name": "threshold",
                "type": "float",
                "label": "Threshold",
                "decimals": 2,
                "default": 0.5,
            }
        ]
        post = call(sock, "tab.get_post_analyze_params", {"tab_id": tab_id})
        assert post["ok"] is True
        assert post["result"]["post_analyze_params"] is None
        assert isinstance(post["result"]["definitions"], list)
    finally:
        sock.close()


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


def test_gui_started_analyze_handle_is_indexed_and_awaited_over_remote(
    fx, monkeypatch: pytest.MonkeyPatch
) -> None:
    import threading

    from tests.gui.app.measure._synthetic import SampleAdapter, SampleParams

    tab_id = fx.ctrl.new_tab("fake")
    sock = open_client(fx.service.port)
    entered = threading.Event()
    release = threading.Event()
    original_analyze = SampleAdapter.analyze

    def held_analyze(self, request, *, plots: Plots):
        entered.set()
        if not release.wait(4):
            raise TimeoutError("fake analysis release was not signalled")
        return original_analyze(self, request, plots=plots)

    try:
        run_id = fx.ctrl.start_run(tab_id)  # GUI path, not an MCP start.
        run_result = call(
            sock, "operation.await", {"operation_id": run_id, "timeout": 2}
        )
        assert run_result["ok"] is True
        assert run_result["result"]["status"] == "finished"
        monkeypatch.setattr(SampleAdapter, "analyze", held_analyze)
        analysis_id = fx.ctrl.analyze(tab_id, SampleParams())
        assert entered.wait(1)
        active = call(sock, "operation.active")
        assert active["ok"] is True
        assert active["result"]["operations"] == [
            {"op": analysis_id, "tab": tab_id, "kind": "analyze"}
        ]
        pending = call(
            sock, "operation.await", {"operation_id": analysis_id, "timeout": 0}
        )
        assert pending["result"] == {"reason": "timeout"}
        release.set()
        completed = call(
            sock, "operation.await", {"operation_id": analysis_id, "timeout": 2}
        )
        assert completed["result"] == {"reason": "completed", "status": "finished"}
        assert call(sock, "operation.active")["result"]["operations"] == []
    finally:
        release.set()
        sock.close()


def _completed_run(fx, sock, tab_id):
    operation_id = fx.ctrl.start_run(tab_id)
    terminal = call(
        sock, "operation.await", {"operation_id": operation_id, "timeout": 2}
    )["result"]
    assert terminal["status"] == "finished"
    return operation_id


@pytest.mark.parametrize("replacement", ["run", "wrong", "pane", "tokens", None])
def test_run_preview_requires_the_original_source_before_rendering(
    fx, monkeypatch, replacement
):
    monkeypatch.setattr(
        SampleAdapter,
        "capabilities",
        replace(SampleAdapter.capabilities, load_data=True),
    )
    tab = fx.ctrl.new_tab("fake")
    with open_client(fx.service.port) as sock:
        original = _completed_run(fx, sock, tab)
        if replacement == "run":
            _completed_run(fx, sock, tab)
        elif replacement == "wrong":
            original += 1000
        params = {"tab_id": tab, "subtab_id": "run", "run_operation_id": original}
        if replacement == "pane":
            params["subtab_id"] = "analysis"
        elif replacement == "tokens":
            params["operation_id"] = original
        fx.view.take_figure_screenshot_for_subtab.reset_mock()
        reply = call(sock, "tab.get_figure", params)
        if replacement is None:
            assert reply["ok"]
            assert reply["result"]["png_b64"]
            fx.view.take_figure_screenshot_for_subtab.assert_called_once_with(
                tab, "run"
            )
        else:
            assert not reply["ok"]
            assert reply["error"]["code"] == (
                "invalid_params"
                if replacement in ("pane", "tokens")
                else "precondition_failed"
            )
            if replacement in ("run", "load", "wrong"):
                assert reply["error"]["reason"] == "result_superseded"
            fx.view.take_figure_screenshot_for_subtab.assert_not_called()


def test_run_snapshot_identifies_the_operation_that_published_its_result(fx):
    tab = fx.ctrl.new_tab("fake")
    with open_client(fx.service.port) as sock:
        for _ in range(2):
            run = _completed_run(fx, sock, tab)
            snapshot = call(sock, "tab.snapshot", {"tab_id": tab})["result"]["tabs"][0]
            assert snapshot["result_state"]["available"] is True
            assert snapshot["result_state"]["source_operation_id"] == run


@pytest.mark.parametrize("replacement", ["run", "unknown", "no_result"])
def test_analyze_rejects_a_superseded_run_without_changing_the_pane(
    fx, monkeypatch, replacement
):
    monkeypatch.setattr(
        SampleAdapter,
        "capabilities",
        replace(SampleAdapter.capabilities, load_data=True),
    )
    tab = fx.ctrl.new_tab("fake")
    with open_client(fx.service.port) as sock:
        original = 1000
        if replacement != "no_result":
            original = _completed_run(fx, sock, tab)
            if replacement == "run":
                _completed_run(fx, sock, tab)
            else:
                original += 1000
            prior = call(sock, "tab.analyze", {"tab_id": tab})["result"]["operation_id"]
            assert (
                call(sock, "operation.await", {"operation_id": prior, "timeout": 2})[
                    "result"
                ]["status"]
                == "finished"
            )
        before = call(sock, "tab.snapshot", {"tab_id": tab})["result"]
        params_before = call(sock, "tab.get_analyze_params", {"tab_id": tab})["result"]
        result_before = call(sock, "tab.get_analyze_result", {"tab_id": tab})["result"]
        fx.view.select_tab_pane.reset_mock()
        reply = call(
            sock,
            "tab.analyze",
            {
                "tab_id": tab,
                "run_operation_id": original,
                "updates": {"threshold": 0.7},
            },
        )
        if reply["ok"]:
            call(
                sock,
                "operation.await",
                {"operation_id": reply["result"]["operation_id"], "timeout": 2},
            )
        assert reply["ok"] is False
        assert reply["error"]["code"] == "precondition_failed"
        assert reply["error"]["reason"] == "result_superseded"
        assert call(sock, "tab.snapshot", {"tab_id": tab})["result"] == before
        assert (
            call(sock, "tab.get_analyze_params", {"tab_id": tab})["result"]
            == params_before
        )
        assert (
            call(sock, "tab.get_analyze_result", {"tab_id": tab})["result"]
            == result_before
        )
        fx.view.select_tab_pane.assert_not_called()
        assert call(sock, "operation.active")["result"]["operations"] == []


def test_original_run_can_be_analyzed_without_bypassing_seen_guards(fx, monkeypatch):
    original_run = SampleAdapter.run

    def known_run(self, request, schema, *, context):
        record = original_run(self, request, schema, context=context)
        return replace(record, result=replace(record.result, data=np.array([1.0, 3.0])))

    monkeypatch.setattr(SampleAdapter, "run", known_run)
    tab = fx.ctrl.new_tab("fake")
    with open_client(fx.service.port) as sock:
        run = _completed_run(fx, sock, tab)
        analyzed = call(sock, "tab.analyze", {"tab_id": tab, "run_operation_id": run})[
            "result"
        ]["operation_id"]
        assert (
            call(sock, "operation.await", {"operation_id": analyzed, "timeout": 2})[
                "result"
            ]["status"]
            == "finished"
        )
        assert (
            call(sock, "tab.get_analyze_result", {"tab_id": tab})["result"]["summary"][
                "peak"
            ]
            == 3.0
        )


@pytest.mark.parametrize("keep_partial", [True, False])
def test_gui_cancel_run_survives_eventless_remote_wait(
    fx, monkeypatch: pytest.MonkeyPatch, keep_partial: bool
) -> None:

    tab_id = fx.ctrl.new_tab("fake")
    sock = open_client(fx.service.port)
    entered = threading.Event()
    release = threading.Event()
    original_run = SampleAdapter.run

    def held_run(self, request, schema, *, context: RunContext):
        entered.set()
        if not release.wait(4):
            raise TimeoutError("fake run release was not signalled")
        if not keep_partial:
            raise InterruptedError("stopped before any acquisition")
        return original_run(self, request, schema, context=context)

    try:
        _completed_run(fx, sock, tab_id)
        monkeypatch.setattr(SampleAdapter, "run", held_run)
        run_id = fx.ctrl.start_run(tab_id)
        assert entered.wait(1)
        running = call(sock, "tab.snapshot", {"tab_id": tab_id})["result"]["tabs"][0]
        assert running["result_state"]["available"] is False
        assert running["result_state"]["source_operation_id"] is None
        assert fx.ctrl.cancel_run()
        release.set()
        reply = call(sock, "operation.await", {"operation_id": run_id, "timeout": 2})
        assert reply["result"] == {
            "reason": "completed",
            "status": "cancelled",
        }
        snapshot = call(sock, "tab.snapshot", {"tab_id": tab_id})["result"]["tabs"][0]
        assert snapshot["result_state"]["available"] is keep_partial
        assert snapshot["result_state"]["source_operation_id"] == (
            run_id if keep_partial else None
        )
    finally:
        release.set()
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
        assert bar["eta_s"] is not None and bar["eta_s"] >= 0
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
        assert resp["result"] == {"active": False, "bars": [], "elapsed_s": None}
    finally:
        sock.close()


def test_operation_progress_elapsed_uses_operation_clock_not_bar_age(
    fx, monkeypatch: pytest.MonkeyPatch
):
    import time

    from zcu_tools.gui.session.pbar_host import ProgressBarModel

    tab_id = fx.ctrl.new_tab("fake")
    operation_id = fx.ctrl.start_run(tab_id)
    bar = ProgressBarModel(
        label="late progress", total=2, start_time=time.monotonic() - 3600
    )
    monkeypatch.setattr(
        fx.service.operation_control,
        "get_operation_progress",
        MagicMock(return_value=((1, bar),)),
    )
    sock = open_client(fx.service.port)
    try:
        resp = call(sock, "operation.progress", {"operation_id": operation_id})
        assert resp["ok"] is True
        assert resp["result"]["active"] is True
        assert 0 <= resp["result"]["elapsed_s"] < 3600
        assert bar.elapsed() >= 3600
        # Do not leave the fake run's worker/diagnostic queued for the next test.
        settled = call(
            sock, "operation.await", {"operation_id": operation_id, "timeout": 3}
        )
        assert settled["ok"] and settled["result"]["reason"] == "completed"
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


def test_device_setup_builds_request_from_complete_cfg(fx):
    fx.service.device_control.get_device_info = MagicMock(
        return_value=FakeDeviceInfo(address="none", value=0.0)
    )
    fx.service.device_control.start_setup_device = MagicMock(return_value=7)
    sock = open_client(fx.service.port)
    try:
        resp = call(
            sock,
            "device.setup",
            {"name": "bias", "cfg": {"address": "none", "value": 1.5}},
        )
        assert resp["ok"] is True
        req = fx.service.device_control.start_setup_device.call_args.args[0]
        assert isinstance(req, SetupDeviceRequest)
        assert req.name == "bias"
        assert isinstance(req.info, FakeDeviceInfo)
        assert req.info.value == 1.5
        assert req.info.address == "none"
    finally:
        sock.close()


def test_device_setup_rejects_wrong_info_model_type(fx):
    fx.service.device_control.get_device_info = MagicMock(
        return_value=FakeDeviceInfo(address="none", value=0.0)
    )
    sock = open_client(fx.service.port)
    try:
        resp = call(
            sock,
            "device.setup",
            {"name": "bias", "cfg": {"address": "none", "type": "x"}},
        )
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
    fx.service.device_control.get_device_info = MagicMock(
        return_value=YOKOGS200Info(address="x", type="YOKOGS200", mode="voltage")
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
    fx.ctrl.get_device_info = ctrl_get
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
    def resources_versions(self) -> dict[str, int]:
        return {}

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
    dev.get_device_unit.return_value = "A"
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
        "operation_id": 103,
    }
    assert _dispatch_with_device_control("device.forget", {"name": "bias"}, dev) == {
        "forgotten": "bias",
    }

    assert _dispatch_with_device_control(
        "device.setup", {"name": "bias", "cfg": {"address": "none", "value": 2.0}}, dev
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
    assert snapshot["unit"] == "A"


# ---------------------------------------------------------------------------
# soc.info — read the connected SoC's hardware summary (wire v11)
# ---------------------------------------------------------------------------


def _install_real_mock_soccfg(fx) -> None:
    """Swap the fixture's MagicMock soccfg for a real QICK mock soccfg so
    describe_soc()'s field access and dump_cfg() return real content."""
    import dataclasses

    from zcu_tools.program.v2.mocksoc import make_mock_soccfg

    ctx = fx.state.session_env
    fx.state.session_env = dataclasses.replace(ctx, soccfg=make_mock_soccfg())


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

    ctx = fx.state.session_env
    fx.state.session_env = dataclasses.replace(ctx, soc=None, soccfg=None)
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


@pytest.mark.parametrize("source", ["primary", "run"])
def test_post_control_rejects_replaced_source_without_starting(fx, source):
    tab = fx.ctrl.new_tab("fake")
    with open_client(fx.service.port) as sock:
        run = _completed_run(fx, sock, tab)
        primary = call(sock, "tab.analyze", {"tab_id": tab})["result"]["operation_id"]
        assert (
            call(sock, "operation.await", {"operation_id": primary, "timeout": 2})[
                "result"
            ]["status"]
            == "finished"
        )
        before = call(sock, "tab.snapshot", {"tab_id": tab})["result"]
        with pytest.raises(FailedPreconditionError) as rejected:
            fx.ctrl.run_analyze_control.start_post_analyze(
                tab,
                SampleParams(),
                operation_id=primary + 1000 if source == "primary" else primary,
                run_operation_id=run + 1000 if source == "run" else run,
            )
        assert rejected.value.reason_code == "result_superseded"
        assert call(sock, "tab.snapshot", {"tab_id": tab})["result"] == before


@pytest.mark.parametrize("source", ["primary", "run", "matching", "omitted"])
def test_post_analysis_uses_requested_sources_before_following_the_pane(
    fx, monkeypatch, source
):
    tab = fx.ctrl.new_tab("fake")
    with open_client(fx.service.port) as sock:
        run = _completed_run(fx, sock, tab)
        primary = call(sock, "tab.analyze", {"tab_id": tab})["result"]["operation_id"]
        assert (
            call(sock, "operation.await", {"operation_id": primary, "timeout": 2})[
                "result"
            ]["status"]
            == "finished"
        )
        monkeypatch.setattr(
            SampleAdapter,
            "capabilities",
            replace(SampleAdapter.capabilities, post_analysis=True),
        )
        monkeypatch.setattr(
            SampleAdapter,
            "post_analyze",
            lambda self, req, *, plots: req.analyze_result,
        )
        fx.state.update_tab_post_analyze_param_instance(tab, SampleParams())
        before = call(sock, "tab.snapshot", {"tab_id": tab})["result"]
        fx.view.select_tab_pane.reset_mock()
        params = {"tab_id": tab}
        if source != "omitted":
            params.update(
                operation_id=primary + 1000 if source == "primary" else primary,
                run_operation_id=run + 1000 if source == "run" else run,
            )
        reply = call(sock, "tab.post_analyze", params)
        terminal = {}
        if reply["ok"]:
            terminal = call(
                sock,
                "operation.await",
                {"operation_id": reply["result"]["operation_id"], "timeout": 2},
            )
        if source in ("primary", "run"):
            assert reply["ok"] is False, reply
            assert reply["error"]["reason"] == "result_superseded"
            assert call(sock, "tab.snapshot", {"tab_id": tab})["result"] == before
            fx.view.select_tab_pane.assert_not_called()
        else:
            assert reply["ok"] is True, reply
            assert terminal["result"]["status"] == "finished", terminal
            result = call(sock, "tab.get_post_analyze_result", {"tab_id": tab})[
                "result"
            ]
            assert result["summary"]["peak"] > 0
            fx.view.select_tab_pane.assert_called_with(tab, "post_analysis")


def _add_fake_tab(fx, tab_id: str) -> None:
    """Register a minimal Session so has_tab(tab_id) is True."""
    from tests.gui.app.measure._synthetic import SampleAdapter

    adapter = SampleAdapter()
    from zcu_tools.cfg_model import Range
    from zcu_tools.gui.form_adapter import Form

    from tests.gui.app.measure._synthetic import SampleCfg

    fx.prepare_tab(
        tab_id, adapter, Form(SampleCfg, {"sweep": Range.between(5.0, 6.0, expts=11)})
    )


def test_tab_cfg_edit_blocked_while_owning_tab_runs(fx):
    """The resource enforces the same busy gate for every frontend."""
    tab_id = "tab-run"
    _add_fake_tab(fx, tab_id)

    fx.ctrl.cfg_resources.lookup(tab_id)
    sock = open_client(fx.service.port)
    try:
        fx.state.set_tab_running(tab_id, running=True)
        resp = call(
            sock,
            "tab.edit_cfg",
            {
                "tab_id": tab_id,
                "edits": [{"path": ["reps"], "value": 10}],
            },
        )
        assert resp["ok"] is False
        assert resp["error"]["code"] == "precondition_failed"
    finally:
        sock.close()


# ---------------------------------------------------------------------------
# Split startup + connect tools (generated from ParamSpec)
# ---------------------------------------------------------------------------


def test_soc_connect_remote_missing_ip_rejected(fx):
    sock = open_client(fx.service.port)
    try:
        resp = call(sock, "soc.connect", {"kind": "remote"})
        assert resp["ok"] is False
        assert resp["error"]["code"] == "invalid_params"
    finally:
        sock.close()


def test_soc_connect_remote_returns_summary_directly(fx):
    # soc.connect is synchronous: the handler calls Controller.connect_sync then
    # reads back the SoC summary via get_soc_info — no operation_id / handle.
    fx.ctrl.connect_sync = MagicMock()  # type: ignore[method-assign]
    fx.ctrl.get_soc_info = MagicMock(  # type: ignore[method-assign]
        return_value={
            "description": "QICK remote board",
            "cfg": {},
            "is_mock": False,
        }
    )
    sock = open_client(fx.service.port)
    try:
        resp = call(
            sock, "soc.connect", {"kind": "remote", "ip": "192.0.2.1", "port": 8888}
        )
        assert resp["ok"] is True
        fx.ctrl.connect_sync.assert_called_once()
        # The reply carries the soc summary directly (description + is_mock), with
        # no operation_id (connect is no longer an async handle).
        assert resp["result"]["soc"] == {
            "description": "QICK remote board",
            "is_mock": False,
        }
        assert "operation_id" not in resp["result"]
    finally:
        sock.close()


# ---------------------------------------------------------------------------
# Adapter spec queries (no tab needed)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("declared", [True, False])
def test_adapter_guide_returns_markdown_and_absolute_resource_directory(
    fx, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, declared: bool
) -> None:
    markdown = "# 自由 guide\n\n![pulse](not-read.png)\n"
    (tmp_path / "guide.md").write_text(markdown, encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    class GuideAdapter(SampleAdapter):
        guide_path: ClassVar[Path | None] = Path("guide.md") if declared else None

    fx.registry.register("guide-test", GuideAdapter)
    sock = open_client(fx.service.port)
    try:
        resp = call(sock, "adapter.guide", {"adapter_name": "guide-test"})
        assert resp["ok"] is True
        assert resp["result"]["guide"] == {
            "markdown": markdown if declared else "(no guide written yet)",
            "resource_dir": str(tmp_path) if declared else None,
        }
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
