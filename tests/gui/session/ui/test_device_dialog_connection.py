"""DeviceDialog connection admission, retry, and lifecycle tests."""

from __future__ import annotations

import os
from pathlib import Path

import pytest
from zcu_tools.device import DeviceManager, FakeDevice
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
from zcu_tools.gui.session.adapters.qt_progress_transport import QtProgressTransport
from zcu_tools.gui.session.device_control import DeviceControlFacet
from zcu_tools.gui.session.hardware_gate import RunBlocksHardwareGate
from zcu_tools.gui.session.operation_handles import OperationHandles
from zcu_tools.gui.session.operation_runner import (
    OperationInfrastructure,
    OperationRunner,
)
from zcu_tools.gui.session.services.device import DeviceService
from zcu_tools.gui.session.services.progress import ProgressService
from zcu_tools.gui.session.state import SessionState
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.gui.session.ui.device_dialog import DeviceDialog

from tests.gui._qt_waits import qt_until
from tests.gui._user_input import ComponentDriver
from tests.gui.session.services._device_waits import DeviceWorkers
from tests.gui.session.ui._device_dialog_support import make_device_control


def test_add_type_menu_offers_only_hardware_drivers(qapp, tmp_path: Path):
    dialog = DeviceDialog(make_device_control(), parameter_provider=lambda: None)
    evidence = Path(os.environ.get("DEVICE_DIALOG_EVIDENCE", str(tmp_path)))
    try:
        dialog.show()
        qapp.processEvents()
        ui = ComponentDriver(dialog)
        with pytest.raises(LookupError, match="FakeDevice.*0 candidates"):
            ui.choose_option("Type:", "FakeDevice")
        for caption in ("YOKOGS200", "RohdeSchwarzSGS100A"):
            ui.choose_option("Type:", caption)
            ui.screenshot(evidence / f"menu-{caption}.png")
        ui.click("Close")
    finally:
        dialog.close()
        dialog.deleteLater()
        qapp.processEvents()


def test_failed_address_shows_reason_and_allows_corrected_retry(qapp, tmp_path: Path):
    workers = DeviceWorkers()
    bus = EventBus()
    gate = RunBlocksHardwareGate(run_kind="run", bus=bus)
    bg = workers.background()
    handles = OperationHandles()
    progress = ProgressService(QtProgressTransport())
    runner = OperationRunner(gate, handles, progress, bg, bus)
    driver = FakeDevice(fast_mode=True)

    def create_driver(type_name: str, address: str) -> FakeDevice:
        if address == "invalid-address":
            raise ValueError("Invalid VISA resource: invalid-address")
        return driver

    service = DeviceService(
        bus,
        SessionState(SessionEnv(parameters=None, soc=None, soccfg=None)),
        OperationInfrastructure(
            gate=gate, background=bg, runner=runner, handles=handles
        ),
        driver_factory=create_driver,
    )
    facet = DeviceControlFacet(bus=bus, device=service, progress=progress)
    dialog = DeviceDialog(facet, parameter_provider=lambda: None)
    try:
        dialog.show()
        qapp.processEvents()
        ui = ComponentDriver(dialog)
        ui.choose_option("Type:", "YOKOGS200")
        ui.enter_text("Name:", "retry_device", commit=False)
        ui.enter_text("Address:", "invalid-address", commit=False)
        ui.screenshot(tmp_path / "01-invalid-address.png")
        ui.click("Add Device")
        # Driver creation must raise; owner delivery either retains the failed row
        # or (the regression) removes it. No timing-based success condition.
        qt_until(
            lambda: (
                not ui.list_rows()
                or any(
                    "Invalid VISA resource" in widget.text
                    for widget in ui.visible().widgets
                )
            ),
            label="failed address delivery",
        )
        # Failure is already delivered. Flush its word-wrap layout before capture;
        # no form edit should be needed to make the second error line readable.
        qapp.processEvents()
        ui.screenshot(tmp_path / "02-failed-address.png")
        assert any(
            "Invalid VISA resource" in widget.text for widget in ui.visible().widgets
        )
        assert ui.observe("Add Device").enabled
        ui.enter_text("Address:", "corrected-address", commit=False)
        ui.screenshot(tmp_path / "03-corrected-address.png")
        ui.click("Add Device")
        qt_until(
            lambda: any(
                "Connected retry_device." in widget.text
                for widget in ui.visible().widgets
            ),
            label="corrected address connected",
        )
        ui.screenshot(tmp_path / "04-retry-connected.png")
        assert [row.text for row in ui.list_rows()] == ["retry_device (YOKOGS200)"]
        ui.click("Close")
    finally:
        workers.close()
        dialog.close()
        dialog.deleteLater()
        driver.close()
        qapp.processEvents()


def test_disconnect_shows_terminal_status_and_allows_reconnect(qapp, tmp_path: Path):
    evidence = Path(os.environ.get("DEVICE_DIALOG_EVIDENCE", str(tmp_path)))
    workers = DeviceWorkers()
    bus = EventBus()
    gate = RunBlocksHardwareGate(run_kind="run", bus=bus)
    bg = workers.background()
    handles = OperationHandles()
    progress = ProgressService(QtProgressTransport())
    runner = OperationRunner(gate, handles, progress, bg, bus)
    driver = FakeDevice(address="synthetic", fast_mode=True)
    manager = DeviceManager()
    service = DeviceService(
        bus,
        SessionState(SessionEnv(parameters=None, soc=None, soccfg=None)),
        OperationInfrastructure(
            gate=gate, background=bg, runner=runner, handles=handles
        ),
        driver_factory=lambda type_name, address: driver,
        device_registry=manager,
    )
    service.register_device("label_device", driver)
    facet = DeviceControlFacet(bus=bus, device=service, progress=progress)
    dialog = DeviceDialog(facet, parameter_provider=lambda: None)
    try:
        dialog.show()
        qapp.processEvents()
        ui = ComponentDriver(dialog)
        ui.screenshot(evidence / "01-published-devices.png")
        ui.select_list_row("label_device (FakeDevice)")
        assert ui.observe("Apply Changes").enabled
        ui.screenshot(evidence / "02-published-connected.png")
        ui.click("Drop")
        # Close returns immediately. Reconnect becoming enabled proves the
        # disconnect terminal has reached the view, independently of its label.
        qt_until(
            lambda: any(
                widget.text == "Reconnect" and widget.enabled
                for widget in ui.visible().widgets
            ),
            label="disconnect delivered",
        )
        ui.screenshot(evidence / "03-disconnected.png")
        visible_text = [widget.text for widget in ui.visible().widgets]
        assert "Disconnected label_device." in visible_text, visible_text
        assert [row.text for row in ui.list_rows()] == [
            "label_device (FakeDevice) [not connected]"
        ]
        ui.enter_text(
            object_name="device_rename_name", value="renamed_device", commit=False
        )
        ui.screenshot(evidence / "04-rename-draft.png")
        ui.click("Rename")
        ui.screenshot(evidence / "05-renamed.png")
        assert [row.text for row in ui.list_rows()] == [
            "renamed_device (FakeDevice) [not connected]"
        ]
        ui.click("Reconnect")
        qt_until(
            lambda: any(
                "Connected renamed_device." in widget.text
                for widget in ui.visible().widgets
            ),
            label="device reconnected",
        )
        ui.screenshot(evidence / "06-reconnected.png")
        assert ui.observe("Apply Changes").enabled
        ui.click("Close")
    finally:
        workers.close()
        dialog.close()
        dialog.deleteLater()
        driver.close()
        qapp.processEvents()
