"""The Window's Devices… button opens the shared Device dialog on this session.

The shared DeviceDialog owns device listing, setup and its inspection mode
(tests/gui/session/ui). Here the host contract is: the button opens one dialog
listing the session's devices; it accepts device changes while no run holds
the hardware, and becomes read-only from Start until terminal delivery,
whether opened before or during the run.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import get_ident

import pytest
from qtpy.QtWidgets import QApplication, QListWidget, QPushButton, QWidget
from zcu_tools.device import FakeDevice
from zcu_tools.gui.app.workflows.window import WorkflowWindow
from zcu_tools.gui.session.services.device import ConnectDeviceRequest
from zcu_tools.gui.session.ui.device_dialog import DeviceDialog

from tests.gui._user_input import ComponentDriver
from tests.gui.app._owner_waits import owner_result

from ._window_support import WindowSession, bind_point, install_soc

RUN_ID = "devices"


def connect_flux(session: WindowSession) -> None:
    """Connect one FakeDevice named flux through the session's device service."""
    harness = session.harness
    harness.services.device.start_connect_device(
        ConnectDeviceRequest("FakeDevice", "flux", "none")
    )
    harness.background.execute()
    harness.background.deliver()


def devices_button(window: QWidget) -> QPushButton:
    found = window.findChild(QPushButton, "workflow.source.devices")
    assert found is not None
    return found


def open_devices(window: QWidget, qapp: QApplication) -> DeviceDialog:
    """Click Devices… and return the single dialog a user now sees."""
    ComponentDriver(window).click(object_name="workflow.source.devices")
    qapp.processEvents()
    shown = [d for d in window.findChildren(DeviceDialog) if d.isVisible()]
    assert len(shown) == 1
    return shown[0]


def listed(dialog: DeviceDialog) -> list[str]:
    devices = dialog.findChild(QListWidget)
    assert devices is not None
    items = [devices.item(row) for row in range(devices.count())]
    return [item.text() for item in items if item is not None]


def accepts_new_device(dialog: DeviceDialog) -> bool:
    """Whether a user can register a device: Add Device is clickable."""
    add = next(b for b in dialog.findChildren(QPushButton) if b.text() == "Add Device")
    return add.isEnabled()


def shown_window(session: WindowSession, qapp: QApplication) -> WorkflowWindow[None]:
    bind_point(session.harness, session.root)
    install_soc(session.harness)
    window = session.window
    window.resize(1600, 1000)
    window.show()
    qapp.processEvents()
    return window


def test_devices_opens_one_shared_dialog_listing_session_devices(
    window_session: WindowSession, qapp: QApplication
) -> None:
    connect_flux(window_session)
    window = shown_window(window_session, qapp)

    dialog = open_devices(window, qapp)
    assert dialog.parent() is window
    assert listed(dialog) == ["flux (FakeDevice)"]
    assert accepts_new_device(dialog)

    assert open_devices(window, qapp) is dialog, "a second click raises the same one"
    dialog.reject()
    qapp.processEvents()
    assert not any(d.isVisible() for d in window.findChildren(DeviceDialog))


def test_open_dialog_is_read_only_during_a_run_and_editable_after(
    window_session: WindowSession, qapp: QApplication
) -> None:
    window = shown_window(window_session, qapp)
    dialog = open_devices(window, qapp)
    assert accepts_new_device(dialog)

    window_session.pause_at = 0
    assert isinstance(window.start(RUN_ID), int)
    window_session.harness.complete_segment()
    qapp.processEvents()
    status = window.controller.status()
    assert status is not None and status.lifecycle == "paused"
    assert not accepts_new_device(dialog)

    window.stop(RUN_ID)
    qapp.processEvents()
    status = window.controller.status()
    assert status is not None and status.lifecycle == "stopped"
    assert accepts_new_device(dialog)


def test_dialog_opened_during_a_run_is_read_only(
    window_session: WindowSession, qapp: QApplication
) -> None:
    window = shown_window(window_session, qapp)
    seen: list[tuple[bool, bool]] = []

    def hook(point: int) -> None:
        if point != 0:
            return
        button = devices_button(window)
        dialog = open_devices(window, qapp)
        seen.append((button.isEnabled(), accepts_new_device(dialog)))
        dialog.reject()
        qapp.processEvents()

    window_session.during_point = hook
    assert isinstance(window.start(RUN_ID), int)
    window_session.harness.complete_segment()
    qapp.processEvents()
    assert seen == [(True, False)]

    status = window.controller.status()
    assert status is not None and status.lifecycle == "done"
    assert accepts_new_device(open_devices(window, qapp))


@pytest.mark.parametrize("stop_paused", [False, True], ids=["done", "stopped"])
def test_workflow_terminal_refreshes_cached_values_but_pause_does_not(
    window_session: WindowSession, qapp: QApplication, tmp_path: Path, stop_paused: bool
) -> None:
    session = window_session
    connect_flux(session)
    harness = session.harness
    window = shown_window(session, qapp)
    driver = harness.services.device.get_connected_devices()["flux"]
    assert isinstance(driver, FakeDevice)
    commits: list[int] = []
    harness.services.device_control.on_device_changed(
        lambda _: commits.append(get_ident())
    )

    def change_device(_point: int) -> None:
        driver.set_value(0.45)

    session.during_point = change_device
    if stop_paused:
        session.pause_at = 0
    ui = ComponentDriver(window)
    devices = ComponentDriver(open_devices(window, qapp))
    devices.select_list_row(
        next(row.text for row in devices.list_rows() if "flux (" in row.text)
    )
    devices.screenshot(tmp_path / "workflow-before.png")
    devices.click("Close")
    ui.click("Start")
    ui.screenshot(tmp_path / "workflow-running.png")
    with ThreadPoolExecutor(max_workers=1) as pool:
        owner_result(
            harness.owner,
            pool.submit(harness.background.execute),
            label="workflow producer",
        )
    harness.owner.pump_all()
    harness.background.deliver()
    ui.screenshot(tmp_path / "workflow-delivered.png")
    if stop_paused:
        status = window.controller.status()
        assert status is not None and status.lifecycle == "paused"
        assert harness.background.work is None, (
            "pause retains the run; it is not terminal"
        )
        ui.click("Stop")
        ui.screenshot(tmp_path / "workflow-stopped.png")
    assert harness.services.device_control.get_cached_device_value("flux") == 0.0
    assert harness.background.work is not None, (
        "terminal must submit a background readback"
    )
    with ThreadPoolExecutor(max_workers=1) as pool:
        pool.submit(harness.background.execute).result(timeout=30)
    assert harness.services.device_control.get_cached_device_value("flux") == 0.0
    harness.background.deliver()
    assert harness.services.device_control.get_cached_device_value("flux") == 0.45
    assert commits == [get_ident()]
    status = window.controller.status()
    assert status is not None and status.lifecycle == (
        "stopped" if stop_paused else "done"
    )
    assert not harness.gate.has_active("run")
    devices = ComponentDriver(open_devices(window, qapp))
    devices.select_list_row(
        next(row.text for row in devices.list_rows() if "flux (" in row.text)
    )
    assert any(w.text == "0.45" for w in devices.visible().widgets)
    devices.screenshot(tmp_path / "workflow-after.png")
    devices.click("Close")
