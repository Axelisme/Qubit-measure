"""Device rename acceptance with real keyboard/mouse input and synthetic drivers."""

from __future__ import annotations

import os
from collections.abc import Iterator
from pathlib import Path

import pytest
from zcu_tools.device import DeviceManager, FakeDevice
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.session.adapters.qt_background import BackgroundRunner
from zcu_tools.gui.session.adapters.qt_progress_transport import QtProgressTransport
from zcu_tools.gui.session.device_control import DeviceControlFacet
from zcu_tools.gui.session.hardware_gate import RunBlocksHardwareGate
from zcu_tools.gui.session.operation_handles import OperationHandles
from zcu_tools.gui.session.operation_runner import OperationRunner
from zcu_tools.gui.session.ports import DeviceMemoryInfo
from zcu_tools.gui.session.services.device import DeviceService
from zcu_tools.gui.session.services.progress import ProgressService
from zcu_tools.gui.session.state import SessionState
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.gui.session.ui.device_dialog import DeviceDialog

from tests.gui._qt_waits import qt_until
from tests.gui._user_input import ComponentDriver


@pytest.fixture
def rename_ui(qapp, request: pytest.FixtureRequest) -> Iterator[ComponentDriver]:
    state = SessionState(SessionEnv(parameters=None, soc=None, soccfg=None))
    bus = BaseEventBus()
    gate = RunBlocksHardwareGate(run_kind="run", bus=bus)
    handles = OperationHandles()
    background = BackgroundRunner()
    progress = ProgressService(QtProgressTransport())
    runner = OperationRunner(gate, handles, progress, background, bus)
    manager = DeviceManager()
    manager.register_device("live", FakeDevice(fast_mode=True))
    device = DeviceService(
        bus,
        state,
        gate,
        background,
        runner,
        handles,
        driver_factory=lambda type_name, address: FakeDevice(fast_mode=True),
        device_registry=manager,
    )
    device.publish_initial_devices()
    device.register_remembered_devices(
        [
            DeviceMemoryInfo(type_name="FakeDevice", name=name, address="synthetic")
            for name in ("old", "occupied")
        ]
    )
    facet = DeviceControlFacet(bus=bus, device=device, progress=progress)
    dialog = DeviceDialog(facet, parameter_provider=lambda: None)
    if getattr(request, "param", False):
        dialog.set_read_only(read_only=True)
    try:
        dialog.show()
        qapp.processEvents()
        yield ComponentDriver(dialog)
    finally:
        dialog.close()
        assert background.quiesce(timeout_ms=30_000)
        dialog.deleteLater()
        manager.close_all_devices()


def _visible_text(ui: ComponentDriver) -> str:
    return "\n".join(widget.text for widget in ui.visible().widgets)


@pytest.mark.parametrize("rename_ui", [True], indirect=True)
def test_readonly_device_dialog_disables_rename(
    rename_ui: ComponentDriver,
    tmp_path: Path,
) -> None:
    ui = rename_ui
    evidence = Path(os.environ.get("DEVICE_RENAME_EVIDENCE", str(tmp_path)))
    ui.select_list_row("old (FakeDevice) [not connected]")
    assert any(widget.text == "Rename" for widget in ui.visible().widgets)
    assert not ui.observe("Rename").enabled
    assert not ui.observe(object_name="device_rename_name").enabled
    ui.screenshot(evidence / "12-readonly.png")
    ui.click("Close")


def test_disconnected_device_can_be_renamed_and_reconnected(
    rename_ui: ComponentDriver,
    tmp_path: Path,
) -> None:
    ui = rename_ui
    evidence = Path(os.environ.get("DEVICE_RENAME_EVIDENCE", str(tmp_path)))
    ui.screenshot(evidence / "01-open.png")
    ui.select_list_row("live (FakeDevice)")
    assert any(widget.text == "Rename" for widget in ui.visible().widgets)
    assert not ui.observe("Rename").enabled
    assert "Disconnect" in ui.observe("Rename").tooltip
    ui.screenshot(evidence / "02-connected-disabled.png")
    ui.select_list_row("old (FakeDevice) [not connected]")
    assert ui.observe("Rename").enabled
    ui.screenshot(evidence / "03-memory-enabled.png")
    ui.click("Rename")
    assert "must not be empty" in _visible_text(ui)
    ui.screenshot(evidence / "04-blank-rejected.png")
    ui.enter_text(object_name="device_rename_name", value="occupied", commit=False)
    ui.screenshot(evidence / "05-conflicting-draft.png")
    ui.click("Rename")
    assert "already in use" in _visible_text(ui)
    assert any(row.text == "old (FakeDevice) [not connected]" for row in ui.list_rows())
    ui.screenshot(evidence / "06-conflict-rejected.png")
    ui.enter_text(object_name="device_rename_name", value="renamed", commit=False)
    ui.screenshot(evidence / "07-new-name-draft.png")
    ui.click("Rename")
    assert any(
        row.text == "renamed (FakeDevice) [not connected]" and row.selected
        for row in ui.list_rows()
    )
    assert "Aliases are unchanged" in _visible_text(ui)
    ui.screenshot(evidence / "08-renamed.png")
    ui.click("Reconnect")
    ui.screenshot(evidence / "09-reconnecting.png")
    qt_until(
        lambda: (
            any(row.text == "renamed (FakeDevice)" for row in ui.list_rows())
            or "Failed" in _visible_text(ui)
        ),
        label="renamed synthetic device reconnect",
    )
    assert any(row.text == "renamed (FakeDevice)" for row in ui.list_rows())
    assert not ui.observe("Rename").enabled
    ui.screenshot(evidence / "10-reconnected.png")
    ui.click("Drop")
    qt_until(
        lambda: any(
            row.text == "renamed (FakeDevice) [not connected]" for row in ui.list_rows()
        ),
        label="renamed synthetic device disconnect",
    )
    assert ui.observe("Rename").enabled
    ui.screenshot(evidence / "11-disconnected-again.png")
    ui.click("Close")
