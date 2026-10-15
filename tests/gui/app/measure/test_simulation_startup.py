"""Startup binds once through owners, preserves live devices, and cleans failures."""

import inspect
import sys

import pytest
from zcu_tools.device.fake import FakeDevice, FakeDeviceInfo
from zcu_tools.experiment.simulation import white_noise
from zcu_tools.gui.app.measure.app import MeasureGuiBehavior
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.services.persistence_types import AppPersistedState, PersistedStartup, PersistedDeviceEntry
from zcu_tools.gui.session.state import DeviceStatus

from tests.gui.app.measure._reload_fakes import Loader


def behavior_with_environment(tmp_path, monkeypatch, env):
    assert "simulate" in inspect.signature(MeasureGuiBehavior).parameters, "startup environment injection required"
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
    return MeasureGuiBehavior(lambda: (Registry(), Loader()), clean=True, project_root=str(tmp_path), simulate=path)


def test_startup_publishes_same_live_manager_before_remembered_devices(qapp, tmp_path, monkeypatch):
    env = white_noise(std_adc=0, seed=1)
    driver = FakeDevice(fast_mode=True)
    env.devices.register_device("bias", driver)
    old_hook = sys.excepthook
    behavior = behavior_with_environment(tmp_path, monkeypatch, env)
    assembly = None
    try:
        assembly = behavior.assemble(None)
        ctrl = assembly.controller
        assert ctrl.has_soc()
        assert ctrl.get_soccfg() is env.soccfg
        assert ctrl.device_control.get_device_info("bias").value == driver.get_value()
        # The owner API preserves existing live rows during preference restore.
        ctrl.restore_persisted_state(AppPersistedState(startup=PersistedStartup(devices=(
            PersistedDeviceEntry(name="bias", type_name="YOKOGS200", address="GPIB::1"),
            PersistedDeviceEntry(name="old_bias", type_name="YOKOGS200", address="GPIB::2"),
        ))))
        snapshots = {entry.name: ctrl.device_control.get_device_snapshot(entry.name) for entry in ctrl.device_control.list_devices()}
        assert snapshots["bias"].status is DeviceStatus.CONNECTED
        assert snapshots["old_bias"].status is DeviceStatus.MEMORY_ONLY
        assert env.devices.get_all_devices() == {"bias": driver}
        ctrl.close()
        assert env.devices.get_device("bias") is driver  # drain does not own driver close
    finally:
        if assembly is not None:
            behavior.close(assembly)
            assembly.window.deleteLater()
        env.devices.close_all_devices()
        sys.excepthook = old_hook
        qapp.processEvents()
    assert env.devices.get_all_devices() == {}


def test_environment_install_failure_closes_returned_manager(qapp, tmp_path, monkeypatch):
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
