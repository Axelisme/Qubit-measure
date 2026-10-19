"""Startup binds once through owners, preserves live devices, and cleans failures."""

import inspect
import sys
from contextlib import contextmanager
from typing import Literal

import numpy as np
import pytest
from zcu_tools.device.fake import FakeDevice, FakeDeviceInfo
from zcu_tools.experiment.ftf import ftf
from zcu_tools.experiment.simulation import SimulationEnvironment, white_noise
from zcu_tools.gui.app.measure.app import MeasureGuiBehavior
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.services.persistence_types import (
    AppPersistedState,
    PersistedDeviceEntry,
    PersistedStartup,
)
from zcu_tools.gui.app.measure.ui.main_window import MainWindow
from zcu_tools.gui.session.services.device import (
    ConnectDeviceRequest,
    DisconnectDeviceRequest,
    SetupDeviceRequest,
)
from zcu_tools.gui.session.state import DeviceStatus
from zcu_tools.program.v2.base import ProgramV2Cfg
from zcu_tools.program.v2.modular import ModularProgramV2
from zcu_tools.program.v2.modules.readout import PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg

from tests.experiment._ftf_templates import settings
from tests.gui._qt_waits import qt_until
from tests.gui.app.measure._reload_fakes import Loader
from tests.gui.app.measure._simulation_devices import bias_environment


def behavior_with_environment(tmp_path, monkeypatch, env):
    assert "simulate" in inspect.signature(MeasureGuiBehavior).parameters, (
        "startup environment injection required"
    )
    # Trusted Python builds the existing bundle once. No external process or Qt
    # side effects in the environment; the test module is its explicit owner.
    monkeypatch.setattr(sys, "_measure_test_environment", env, raising=False)
    path = tmp_path / "environment.py"
    path.write_text(
        "import sys\n"
        "builds = 0\n"
        "def build_environment():\n"
        "    global builds\n"
        "    builds += 1\n"
        "    if builds != 1: raise RuntimeError('builder called twice')\n"
        "    return sys._measure_test_environment\n"
    )
    return MeasureGuiBehavior(
        lambda: (Registry(), Loader()),
        clean=True,
        project_root=str(tmp_path),
        simulate=path,
    )


def test_startup_publishes_same_live_manager_before_remembered_devices(
    qapp, tmp_path, monkeypatch
):
    env = white_noise(std_adc=0, seed=1)
    driver = FakeDevice(fast_mode=True)
    env.devices.register_device("bias", driver)
    old_hook = sys.excepthook
    behavior = behavior_with_environment(tmp_path, monkeypatch, env)
    assembly = None
    try:
        assembly = behavior.assemble(None)
        ctrl = assembly.controller
        if not isinstance(ctrl, Controller) or not isinstance(
            assembly.window, MainWindow
        ):
            raise AssertionError(
                "measure assembly must expose its controller and window"
            )
        assert ctrl.has_soc()
        assert ctrl.get_soccfg() is env.soccfg
        info = ctrl.device_control.get_device_info("bias")
        assert isinstance(info, FakeDeviceInfo)
        assert info.value == driver.get_value()
        # The owner API preserves existing live rows during preference restore.
        ctrl.restore_persisted_state(
            AppPersistedState(
                startup=PersistedStartup(
                    devices=(
                        PersistedDeviceEntry(
                            name="bias", type_name="YOKOGS200", address="GPIB::1"
                        ),
                        PersistedDeviceEntry(
                            name="old_bias", type_name="YOKOGS200", address="GPIB::2"
                        ),
                    )
                )
            )
        )
        snapshots = {
            entry.name: ctrl.device_control.get_device_snapshot(entry.name)
            for entry in ctrl.device_control.list_devices()
        }
        assert (
            snapshots["bias"] is not None
            and snapshots["bias"].status is DeviceStatus.CONNECTED
        )
        assert (
            snapshots["old_bias"] is not None
            and snapshots["old_bias"].status is DeviceStatus.MEMORY_ONLY
        )
        assert env.devices.get_all_devices() == {"bias": driver}
        ctrl.close()
        assert (
            env.devices.get_device("bias") is driver
        )  # drain does not own driver close
    finally:
        if assembly is not None:
            behavior.close(assembly)
            if isinstance(assembly.window, MainWindow):
                assembly.window.deleteLater()
        env.devices.close_all_devices()
        sys.excepthook = old_hook
        qapp.processEvents()
    assert env.devices.get_all_devices() == {}


def test_environment_install_failure_closes_returned_manager(
    qapp, tmp_path, monkeypatch
):
    class FailingInfoDevice(FakeDevice):
        info_model = FakeDeviceInfo

        def get_info(self):
            raise RuntimeError("snapshot unavailable")

    env = white_noise()
    env.devices.register_device("bad", FailingInfoDevice(fast_mode=True))
    old_hook = sys.excepthook
    behavior = behavior_with_environment(tmp_path, monkeypatch, env)
    try:
        with pytest.raises(RuntimeError, match="snapshot unavailable"):
            behavior.assemble(None)
        assert env.devices.get_all_devices() == {}
    finally:
        env.devices.close_all_devices()
        sys.excepthook = old_hook
        qapp.processEvents()


@contextmanager
def assembled_environment(qapp, tmp_path, monkeypatch, env):
    """Install a trusted synthetic environment and drain the real composition."""
    old_hook = sys.excepthook
    behavior = behavior_with_environment(tmp_path, monkeypatch, env)
    assembly = None
    try:
        assembly = behavior.assemble(None)
        ctrl = assembly.controller
        if not isinstance(ctrl, Controller):
            raise AssertionError("measure composition must expose its controller")
        yield ctrl
    finally:
        behavior.close(assembly)
        if assembly is not None and isinstance(assembly.window, MainWindow):
            assembly.window.dispose_control_watches()
            assembly.window.deleteLater()
        env.devices.close_all_devices()
        sys.excepthook = old_hook
        qapp.processEvents()


def completed_operation(ctrl: Controller, token: int):
    """Dispatch the submitted worker's guaranteed terminal delivery."""
    qt_until(
        lambda: (
            (result := ctrl.await_operation(token, timeout=0)) is not None
            and result.reason == "completed"
        ),
        label="simulation device operation",
    )
    result = ctrl.await_operation(token, timeout=0)
    if result is None or result.outcome is None:
        raise AssertionError("completed operation must carry an outcome")
    return result.outcome


def setup_bias(
    ctrl: Controller,
    name: str,
    value: float,
    *,
    unit: Literal["native", "V", "A"] = "V",
):
    """Apply a fixed-unit setting through the public device facet."""
    info = ctrl.device_control.get_device_info(name)
    if info is None:
        raise AssertionError("connected device must expose its settings")
    outcome = completed_operation(
        ctrl,
        ctrl.device_control.start_setup_device(
            SetupDeviceRequest(
                name,
                FakeDeviceInfo(
                    address=info.address, value=value, value_unit=unit, rampstep=10
                ),
            )
        ),
    )
    assert outcome.status == "finished", outcome.error


def bias_trace(env: SimulationEnvironment):
    """Acquire a short real simulation trace, not an internal driver observation."""
    readout = PulseReadoutCfg(
        gen_ch=0,
        ro_ch=0,
        nqz=1,
        freq=100.0,
        gain=0.1,
        waveform=ConstWaveformCfg(length=0.02),
        ro_length=0.02,
    )
    program = ModularProgramV2(
        env.soccfg, ProgramV2Cfg(reps=1), modules=[readout.build("readout")]
    )
    return program.acquire_decimated(env.soc, progress=False)[0]


def test_reconnected_device_retains_state_and_drives_simulation(
    qapp, tmp_path, monkeypatch
):
    env = bias_environment()
    original = env.devices.get_device("bias")
    with assembled_environment(qapp, tmp_path, monkeypatch, env) as ctrl:
        setup_bias(ctrl, "bias", 2.0)
        np.testing.assert_allclose(bias_trace(env), 2.0)
        outcome = completed_operation(
            ctrl,
            ctrl.device_control.start_disconnect_device(
                DisconnectDeviceRequest("bias")
            ),
        )
        assert outcome.status == "finished", outcome.error
        outcome = completed_operation(
            ctrl, ctrl.device_control.start_reconnect_device("bias")
        )
        assert outcome.status == "finished", outcome.error
        assert ctrl.device_control.get_cached_device_value("bias") == 2.0
        assert ctrl.device_control.get_device_unit("bias") == "V"
        assert env.devices.get_device("bias") is original
        setup_bias(ctrl, "bias", 5.0)
        np.testing.assert_allclose(bias_trace(env), 5.0)


def test_new_name_at_same_address_drives_original_simulation(
    qapp, tmp_path, monkeypatch
):
    env = bias_environment()
    original = env.devices.get_device("bias")
    with assembled_environment(qapp, tmp_path, monkeypatch, env) as ctrl:
        setup_bias(ctrl, "bias", 2.0)
        np.testing.assert_allclose(bias_trace(env), 2.0)
        outcome = completed_operation(
            ctrl,
            ctrl.device_control.start_disconnect_device(
                DisconnectDeviceRequest("bias")
            ),
        )
        assert outcome.status == "finished", outcome.error
        ctrl.device_control.forget_device("bias")
        outcome = completed_operation(
            ctrl,
            ctrl.device_control.start_connect_device(
                ConnectDeviceRequest("FakeDevice", "renamed_bias", "bias")
            ),
        )
        assert outcome.status == "finished", outcome.error
        assert ctrl.device_control.get_cached_device_value("renamed_bias") == 2.0
        assert env.devices.get_device("renamed_bias") is original
        setup_bias(ctrl, "renamed_bias", 3.0)
        np.testing.assert_allclose(bias_trace(env), 3.0)


@pytest.mark.parametrize("reconnect", [True, False], ids=["reconnect", "new-name"])
def test_ftf_address_binding_preserves_source_state_and_units(
    qapp, tmp_path, monkeypatch, reconnect
):
    env = ftf(settings())
    with assembled_environment(qapp, tmp_path, monkeypatch, env) as ctrl:
        setup_bias(ctrl, "bias", 0.3)
        assert dict(env.soc.binding.capture_snapshot().values)["bias"] == 0.3
        outcome = completed_operation(
            ctrl,
            ctrl.device_control.start_disconnect_device(
                DisconnectDeviceRequest("bias")
            ),
        )
        assert outcome.status == "finished", outcome.error
        name = "bias" if reconnect else "renamed_bias"
        initial = ctrl.device_control.get_device_snapshot("bias")
        if initial is None:
            raise AssertionError("disconnected device must retain its address")
        token = (
            ctrl.device_control.start_reconnect_device("bias")
            if reconnect
            else ctrl.device_control.start_connect_device(
                ConnectDeviceRequest("FakeDevice", name, initial.address)
            )
        )
        outcome = completed_operation(ctrl, token)
        assert outcome.status == "finished", outcome.error
        assert ctrl.device_control.get_cached_device_value(name) == 0.3
        assert ctrl.device_control.get_device_unit(name) == "V"
        setup_bias(ctrl, name, 0.4)
        assert dict(env.soc.binding.capture_snapshot().values)["bias"] == 0.4


def test_unmodeled_simulation_device_reports_failure(qapp, tmp_path, monkeypatch):
    env = white_noise(std_adc=0)
    with assembled_environment(qapp, tmp_path, monkeypatch, env) as ctrl:
        outcome = completed_operation(
            ctrl,
            ctrl.device_control.start_connect_device(
                ConnectDeviceRequest("FakeDevice", "unmodeled", "none")
            ),
        )
        assert outcome.status == "failed", "unmodeled device must not silently connect"
        assert outcome.error is not None and "cannot bind" in outcome.error.lower()
        assert "unmodeled" not in env.devices.get_all_devices()


def test_second_connected_name_at_same_address_is_rejected(qapp, tmp_path, monkeypatch):
    env = bias_environment()
    with assembled_environment(qapp, tmp_path, monkeypatch, env) as ctrl:
        setup_bias(ctrl, "bias", 2.0)
        outcome = completed_operation(
            ctrl,
            ctrl.device_control.start_connect_device(
                ConnectDeviceRequest("FakeDevice", "duplicate_bias", "bias")
            ),
        )
        assert outcome.status == "failed", "one driver cannot have two connected names"
        assert outcome.error is not None and "already registered" in outcome.error
        setup_bias(ctrl, "bias", 3.0)
        np.testing.assert_allclose(bias_trace(env), 3.0)


def test_different_addresses_are_independent(qapp, tmp_path, monkeypatch):
    env = bias_environment()
    with assembled_environment(qapp, tmp_path, monkeypatch, env) as ctrl:
        for name in ("bias", "spare"):
            outcome = completed_operation(
                ctrl,
                ctrl.device_control.start_disconnect_device(
                    DisconnectDeviceRequest(name)
                ),
            )
            assert outcome.status == "finished", outcome.error
            outcome = completed_operation(
                ctrl, ctrl.device_control.start_reconnect_device(name)
            )
            assert outcome.status == "finished", outcome.error
        setup_bias(ctrl, "bias", 2.0)
        setup_bias(ctrl, "spare", 3.0)
        assert env.devices.get_device("bias") is not env.devices.get_device("spare")
        assert ctrl.device_control.get_cached_device_value("bias") == 2.0
        assert ctrl.device_control.get_cached_device_value("spare") == 3.0
        np.testing.assert_allclose(bias_trace(env), 5.0)
