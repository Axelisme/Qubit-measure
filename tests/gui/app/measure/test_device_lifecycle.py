"""Measure composition owns dynamically registered drivers after work drains."""

import sys

import pytest
from zcu_tools.device import FakeDevice, FakeDeviceInfo
from zcu_tools.gui.app.measure.app import MeasureGuiBehavior
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.ui.main_window import MainWindow
from zcu_tools.gui.session.state import DeviceStatus
from zcu_tools.program.v2.mocksoc import make_mock_soccfg

from tests.gui.app.measure._reload_fakes import Loader


class ClosingDriver(FakeDevice):
    info_model = FakeDeviceInfo
    attempts = 0
    fail_close = False

    def close(self) -> None:
        self.attempts += 1
        if self.fail_close:
            raise RuntimeError("close failed for synthetic driver")
        super().close()


@pytest.fixture
def assembled_app(qapp, tmp_path):
    old_hook = sys.excepthook
    behavior = MeasureGuiBehavior(
        lambda: (Registry(), Loader()), clean=True, project_root=str(tmp_path)
    )
    assembly = behavior.assemble(None)
    try:
        yield behavior, assembly
    finally:
        behavior.close(assembly)
        if isinstance(assembly.window, MainWindow):
            assembly.window.dispose_control_watches()
            assembly.window.deleteLater()
        sys.excepthook = old_hook
        qapp.processEvents()


def controller_of(value: object) -> Controller:
    if not isinstance(value, Controller):
        raise AssertionError("measure composition must expose its controller")
    return value


def test_behavior_closes_runtime_registered_drivers_after_drain(assembled_app):
    behavior, assembly = assembled_app
    ctrl = controller_of(assembly.controller)
    driver = ClosingDriver(fast_mode=True)
    soccfg = make_mock_soccfg()
    ctrl.install_soc(soccfg, soccfg)
    ctrl.register_device("runtime", driver)
    assert ctrl.device_control.get_device_info("runtime") == driver.get_info()
    ctrl.close()
    assert (
        driver.attempts == 0
    )  # drain alone must leave resources for on_close callbacks

    behavior.close(assembly)

    row = ctrl.device_control.get_device_snapshot("runtime")
    assert row is not None and row.status is DeviceStatus.MEMORY_ONLY
    assert driver.attempts == 1
    assert not ctrl.has_soc()
    behavior.close(assembly)
    assert driver.attempts == 1


def test_close_hardware_isolates_driver_failure_and_still_uninstalls_soc(
    assembled_app, caplog
):
    _behavior, assembly = assembled_app
    ctrl = controller_of(assembly.controller)
    bad = ClosingDriver(fast_mode=True)
    good = ClosingDriver(fast_mode=True)
    bad.fail_close = True
    soccfg = make_mock_soccfg()
    ctrl.install_soc(soccfg, soccfg)
    ctrl.register_device("bad", bad)
    ctrl.register_device("good", good)
    try:
        ctrl.close()
        ctrl.close_hardware()
        assert bad.attempts == good.attempts == 1
        assert not ctrl.has_soc()
        rows = {
            name: ctrl.device_control.get_device_snapshot(name)
            for name in ("bad", "good")
        }
        assert (
            rows["good"] is not None and rows["good"].status is DeviceStatus.MEMORY_ONLY
        )
        assert rows["bad"] is not None and rows["bad"].status is DeviceStatus.CONNECTED
        assert "close failed for synthetic driver" in caplog.text
        ctrl.close_hardware()
        assert bad.attempts == 1  # a failed close must not cause a hidden retry
    finally:
        bad.fail_close = False
        bad.close()
