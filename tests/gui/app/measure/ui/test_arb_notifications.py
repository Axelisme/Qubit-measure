"""ARB notifications retain host policy while Local modal choices stay guarded."""

from __future__ import annotations

import os
import socket
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Never

import pytest
from qtpy.QtCore import QCoreApplication, QEvent, QTimer
from qtpy.QtWidgets import QApplication, QInputDialog, QMessageBox
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.remote import ControlOptions, RemoteControlAdapter
from zcu_tools.gui.app.measure.remote.dialogs import DialogName
from zcu_tools.gui.app.measure.state import State
from zcu_tools.gui.app.measure.ui.main_window import MainWindow
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.session.arb_waveform import ArbWaveformPort, ArbWaveformWriteStatus
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.gui.session.ui.arb_waveform_dialog import ArbWaveformDialog
from zcu_tools.resources.waveform_assets import (
    ArbWaveformData,
    ArbWaveformInfo,
    FormulaRecipe,
    render_formula_recipe,
)

from tests.gui._user_input import ComponentDriver
from tests.gui.app.measure.remote._helpers import recv_response, send

pytestmark = [pytest.mark.requires_loopback, pytest.mark.uses_wall_clock]


class FailingAssets:
    """In-memory public asset port with a one-shot persistence failure."""

    failure = ""

    def __init__(self) -> None:
        self.data = render_formula_recipe(
            {"segments": [{"duration": 1.0, "formula": "1"}], "normalize": "peak"}
        )

    def list_data_keys(self) -> list[str]:
        return ["existing", "duplicate"]

    def list_infos(self) -> list[ArbWaveformInfo]:
        return [
            ArbWaveformInfo(key, 1.0, 10, False, 1.0, True, 0.0, 0, self.data.recipe)
            for key in self.list_data_keys()
        ]

    def fail(self, operation: str) -> None:
        if self.failure == operation:
            self.failure = ""
            raise OSError(f"{operation} disk failure")

    def load_data(self, data_key: str) -> ArbWaveformData:
        self.fail("reload")
        return self.data

    def set_formula(
        self,
        data_key: str,
        recipe: FormulaRecipe | dict[str, object],
        *,
        overwrite: bool,
    ) -> ArbWaveformWriteStatus:
        self.fail("save")
        return "created"

    def rename(self, old_data_key: str, new_data_key: str) -> None:
        self.fail("rename")

    def delete(self, data_key: str) -> None:
        self.fail("delete")

    def get_preview(self, data_key: str) -> Never:
        raise AssertionError("ARB preview must render locally")


class AssetController(Controller):
    """Compose the named window with a fake at its public asset port."""

    assets: ArbWaveformPort

    @property
    def arb_waveforms(self) -> ArbWaveformPort:
        return self.assets


@dataclass
class NotificationUi:
    window: MainWindow
    client: socket.socket
    assets: FailingAssets

    def connect(self) -> dict[str, object]:
        send(
            self.client,
            {
                "id": "notification",
                "method": "control.connect",
                "params": {"label": "arb-agent"},
            },
        )
        return recv_response(self.client, "notification")


@pytest.fixture
def notification_ui(qapp: QApplication, tmp_path: Path) -> Iterator[NotificationUi]:
    controller = AssetController(
        State(SessionEnv(parameters=None, soc=None, soccfg=None)),
        Registry(),
        None,
        BaseEventBus(),
        project_root=str(tmp_path),
    )
    assets = FailingAssets()
    controller.assets = assets
    window = MainWindow(controller, templates=lambda: {})
    controller.add_view(window)
    remote = RemoteControlAdapter(
        controller,
        ControlOptions(port=0),
        controller.owner_scheduler,
        render_view=window,
    )
    client = socket.create_connection(("127.0.0.1", remote.start()), timeout=30)
    window.show()
    window.open_dialog(DialogName.ARB_WAVEFORM)
    qapp.processEvents()
    try:
        yield NotificationUi(window, client, assets)
    finally:
        client.close()
        remote.stop()
        controller.close()
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        qapp.processEvents()


def evidence(driver: ComponentDriver, name: str) -> None:
    """Save opt-in UI evidence without depending on a task-local path."""
    if directory := os.environ.get("ARB_NOTIFICATION_EVIDENCE"):
        driver.screenshot(Path(directory) / f"{name}.png")


@pytest.mark.parametrize(
    ("operation", "title", "message"),
    [
        ("save", "Save failed", "save disk failure"),
        ("reload", "Reload failed", "reload disk failure"),
        ("invalid", "Invalid data_key", "data_key must match ^[A-Za-z][A-Za-z0-9_]*$"),
        (
            "duplicate",
            "Rename failed",
            "Arbitrary waveform 'duplicate' already exists.",
        ),
        ("rename", "Rename failed", "rename disk failure"),
        ("delete", "Delete failed", "delete disk failure"),
    ],
)
def test_named_arb_failure_stays_visible_during_owner_connect(
    notification_ui: NotificationUi,
    qapp: QApplication,
    operation: str,
    title: str,
    message: str,
) -> None:
    ui = notification_ui
    dialog = ui.window.findChild(ArbWaveformDialog)
    assert dialog is not None
    driver = ComponentDriver(dialog)
    evidence(driver, f"{operation}-before")
    # Selection through a visible row is not part of the notification contract.
    from qtpy.QtWidgets import QTableWidget

    table = next(
        table for table in dialog.findChildren(QTableWidget) if table.columnCount() == 3
    )
    if operation not in {"save", "reload"}:
        table.selectRow(
            next(
                row
                for row in range(table.rowCount())
                if (item := table.item(row, 0)) is not None
                and item.text() == "existing"
            )
        )
    ui.assets.failure = operation
    replies: list[dict[str, object]] = []
    visible: list[tuple[str, str]] = []

    def observe() -> None:
        boxes = [
            box
            for box in dialog.findChildren(QMessageBox)
            if box.isVisible() and box.windowTitle() == title
        ]
        if boxes:
            box = boxes[0]
            visible.append((box.windowTitle(), box.text()))
            evidence(driver, f"{operation}-notification")
            replies.append(ui.connect())
            evidence(driver, f"{operation}-owner")
            # Only unwind the old synchronous modal call, after recording refusal.
            # The green non-modal notification must stay visible for the assertions.
            if box.isModal():
                box.close()

    def answer() -> None:
        prompt = qapp.activeModalWidget()
        if isinstance(prompt, QInputDialog):
            prompt.setTextValue(
                {"invalid": "bad key", "duplicate": "duplicate"}.get(
                    operation, "renamed"
                )
            )
            QTimer.singleShot(0, observe)
            prompt.accept()
        elif isinstance(prompt, QMessageBox):
            QTimer.singleShot(0, observe)
            button = prompt.button(QMessageBox.StandardButton.Yes)
            if button is not None:
                button.click()

    if operation in {"save", "reload"}:
        QTimer.singleShot(0, observe)
        driver.click("Save")
    else:
        QTimer.singleShot(0, answer)
        driver.click("Delete" if operation == "delete" else "Rename")
    if not replies:
        observe()
    try:
        assert visible == [(title, message)]
        assert replies and replies[0]["ok"], (
            "visible ARB notification blocked owner connect"
        )
        assert replies[0]["result"] == {
            "holder": "rpc:arb-agent",
            "locked": False,
            "online": True,
            "me": True,
        }
        assert any(item.title == title for item in driver.dialogs()), (
            "notification lifetime ended before user dismissal"
        )
        assert ui.connect()["ok"], (
            "retained notification blocked subsequent control query"
        )
    finally:
        for box in dialog.findChildren(QMessageBox):
            box.close()


def test_shared_arb_notification_default_remains_modal(qapp: QApplication) -> None:
    assets = FailingAssets()
    dialog = ArbWaveformDialog(assets)
    dialog.show()
    assets.failure = "save"
    observed: list[tuple[str, bool, str]] = []

    def observe_and_dismiss() -> None:
        box = qapp.activeModalWidget()
        if isinstance(box, QMessageBox):
            observed.append((box.windowTitle(), box.isModal(), box.text()))
            box.close()

    try:
        QTimer.singleShot(0, observe_and_dismiss)
        ComponentDriver(dialog).click("Save")
        assert observed == [("Save failed", True, "save disk failure")]
    finally:
        dialog.close()
        dialog.deleteLater()
        qapp.processEvents()
