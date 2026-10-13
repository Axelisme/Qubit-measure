"""Tests for the main-window toolbar coordinator."""

from __future__ import annotations

import json
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

import pytest
from qtpy.QtCore import QCoreApplication, QEvent, QPointF, Qt, QTimer
from qtpy.QtGui import QKeyEvent, QMouseEvent
from qtpy.QtWidgets import QApplication, QMenu, QPushButton, QWidget
from zcu_tools.gui.app.measure.adapter import SessionEnv
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.remote import ControlOptions, RemoteControlAdapter
from zcu_tools.gui.app.measure.remote.dialogs import DialogName
from zcu_tools.gui.app.measure.state import State
from zcu_tools.gui.app.measure.ui.main_window import MainWindow
from zcu_tools.gui.app.measure.ui.main_window_toolbar import (
    AdapterMenuItem,
    MainWindowToolbar,
    adapter_menu_items,
)
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.remote.rpc_endpoint import ClientLink
from zcu_tools.gui.remote.wire import Request
from zcu_tools.gui.session.context_types import (
    CommandOrigin,
    EntryRequest,
    PointCreateRequest,
)

from tests.gui._dialog_fakes import RecordingDialogPresenter
from tests.gui.app.fluxdep.remote._route_harness import (
    ImmediateOwnerScheduler,
    RouteReply,
)
from tests.gui.app.measure._reload_fakes import OldAdapter


def test_adapter_menu_items_preserve_flat_adapter_names() -> None:
    assert adapter_menu_items(["lookback", "onetone"]) == (
        AdapterMenuItem("lookback", "lookback", ()),
        AdapterMenuItem("onetone", "onetone", ()),
    )


def test_adapter_menu_items_split_nested_adapter_names() -> None:
    assert adapter_menu_items(
        [
            "twotone/rabi/amp_rabi",
            "twotone/rabi/len_rabi",
            "time/t1",
        ]
    ) == (
        AdapterMenuItem("twotone/rabi/amp_rabi", "amp_rabi", ("twotone", "rabi")),
        AdapterMenuItem("twotone/rabi/len_rabi", "len_rabi", ("twotone", "rabi")),
        AdapterMenuItem("time/t1", "t1", ("time",)),
    )


def test_adapter_menu_items_preserve_input_order() -> None:
    items = adapter_menu_items(["b/second", "a/first", "root"])

    assert [item.adapter_name for item in items] == ["b/second", "a/first", "root"]


class _RecordingToolbarHost:
    def __init__(self) -> None:
        self.opened: list[DialogName] = []

    def list_adapter_names(self) -> list[str]:
        return []

    def create_tab(self, adapter_name: str) -> None:
        raise AssertionError(f"unexpected tab creation: {adapter_name!r}")

    def open_dialog(self, name: DialogName) -> None:
        self.opened.append(name)

    def reload_experiments(self) -> None:
        pass

    def retry_skipped_experiment_tabs(self) -> None:
        pass


def test_toolbar_dialog_buttons_open_expected_dialogs(qapp) -> None:
    parent = QWidget()
    host = _RecordingToolbarHost()
    toolbar = MainWindowToolbar(host, parent=parent)

    buttons: dict[str, QPushButton] = {}
    for index in range(toolbar.layout.count()):
        item = toolbar.layout.itemAt(index)
        if item is None:
            continue
        widget = item.widget()
        if isinstance(widget, QPushButton):
            buttons[widget.text()] = widget

    buttons["Setup…"].click()
    buttons["Devices…"].click()
    buttons["Predictor…"].click()
    buttons["Inspect…"].click()

    assert host.opened == [
        DialogName.SETUP,
        DialogName.DEVICE,
        DialogName.PREDICTOR,
        DialogName.INSPECT,
    ]


class ToolbarController(Controller):
    """Expose the existing background drain for this real UI composition."""

    def quiesce(self) -> None:
        self._background_svc.quiesce()


@dataclass
class ToolbarApplication:
    controller: ToolbarController
    window: MainWindow
    remote: RemoteControlAdapter
    link: ClientLink
    tab_id: str

    def rpc(self, method: str, **params: object) -> RouteReply:
        """Route and encode one same-process RPC through the real dispatch."""
        self.remote.route(self.link, Request("toolbar", method, params))
        reply: RouteReply = json.loads(self.link.outbound.get_nowait())
        return reply

    def connect(self) -> None:
        assert self.rpc("control.connect", label="measure-agent")["ok"]


@pytest.fixture
def toolbar_application(
    qapp: QApplication, tmp_path: Path
) -> Iterator[ToolbarApplication]:
    registry = Registry()
    registry.register("stoppable", OldAdapter)
    registry.register("group/nested", OldAdapter)
    scheduler = ImmediateOwnerScheduler()
    ctrl = ToolbarController(
        State(SessionEnv(parameters=None, soc=None, soccfg=None)),
        registry,
        None,
        BaseEventBus(),
        owner_scheduler=scheduler,
        project_root=str(tmp_path),
    )
    ctrl.setup_control.new_entry(EntryRequest("toolbar"))
    ctrl.setup_control.new_point(PointCreateRequest("point", "", CommandOrigin("gui")))
    window = MainWindow(
        ctrl, templates=lambda: {}, dialog_presenter=RecordingDialogPresenter()
    )
    ctrl.add_view(window)
    tab_id = ctrl.new_tab("stoppable")
    remote = RemoteControlAdapter(
        ctrl, ControlOptions(port=0), scheduler, render_view=window
    )
    link = ClientLink("toolbar-client", token_required=False)
    remote.on_client_open(link)
    window.show()
    try:
        yield ToolbarApplication(ctrl, window, remote, link, tab_id)
    finally:
        remote.on_client_close(link, on_owner_thread=True)
        ctrl.quiesce()
        for tid in tuple(ctrl.list_tab_ids()):
            ctrl.tab_control.close_tab(tid)
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        qapp.processEvents()


def button(widget: QWidget, text: str) -> QPushButton:
    return next(b for b in widget.findChildren(QPushButton) if b.text() == text)


@pytest.mark.uses_wall_clock
@pytest.mark.parametrize("adapter_name", ["stoppable", "group/nested"])
@pytest.mark.parametrize("decision", ["local", "cancel", "remote"])
def test_live_new_tab_popup_preserves_local_selection_cancel_and_remote_gate(
    toolbar_application: ToolbarApplication,
    qapp: QApplication,
    adapter_name: str,
    decision: str,
) -> None:
    ui = toolbar_application
    baseline = tuple(ui.controller.list_tab_ids())
    failures: list[Exception] = []
    visited: list[bool] = []

    def choose() -> None:
        popup = qapp.activePopupWidget()
        try:
            assert isinstance(popup, QMenu), "New Tab must open a real popup"
            assert qapp.activeModalWidget() is None
            visited.append(True)
            target_menu = popup
            if "/" in adapter_name:
                target_menu = next(
                    menu
                    for menu in popup.findChildren(QMenu)
                    if menu.title() == "group"
                )
                popup.setActiveAction(target_menu.menuAction())
                QApplication.sendEvent(
                    popup,
                    QKeyEvent(
                        QEvent.Type.KeyPress,
                        Qt.Key.Key_Right,
                        Qt.KeyboardModifier.NoModifier,
                    ),
                )
                qapp.processEvents()
            selected = next(
                action
                for action in target_menu.actions()
                if action.data() == adapter_name
            )
            if decision == "remote":
                ui.connect()
                assert (
                    ui.controller.control_identity.status().holder
                    == "rpc:measure-agent"
                )
                assert not selected.isEnabled(), (
                    "Live adapter actions must receive Remote projection"
                )
            if decision != "cancel":
                position = target_menu.actionGeometry(selected).center()
                for event_type, buttons in (
                    (QEvent.Type.MouseButtonPress, Qt.MouseButton.LeftButton),
                    (QEvent.Type.MouseButtonRelease, Qt.MouseButton.NoButton),
                ):
                    QApplication.sendEvent(
                        target_menu,
                        QMouseEvent(
                            event_type,
                            QPointF(position),
                            QPointF(target_menu.mapToGlobal(position)),
                            Qt.MouseButton.LeftButton,
                            buttons,
                            Qt.KeyboardModifier.NoModifier,
                        ),
                    )
        except (AssertionError, StopIteration) as exc:
            failures.append(exc)
        finally:
            if isinstance(popup, QMenu):
                for submenu in popup.findChildren(QMenu):
                    submenu.close()
                popup.close()

    QTimer.singleShot(0, choose)
    button(ui.window, "New Tab ▾").click()
    assert visited == [True]
    if failures:
        raise failures[0]
    after = tuple(ui.controller.list_tab_ids())
    if decision == "local":
        assert len(after) == len(baseline) + 1
        assert ui.controller.get_tab_snapshot(after[-1]).adapter_name == adapter_name
    else:
        assert after == baseline


@pytest.mark.uses_wall_clock
def test_run_with_alias_overrides_is_rejected_without_effects(
    toolbar_application: ToolbarApplication,
) -> None:
    """Run takes no alias overrides; Reset with aliases is the way to use them."""
    ui = toolbar_application
    cfg = ui.controller.cfg_resources.lookup(ui.tab_id)
    cfg.write({("knob",): {"expr": "$qubit.gain"}})
    before = cfg.input.data
    publications: list[object] = []
    remove = cfg.subscribe(lambda: publications.append(cfg.input.data))
    rows = ui.controller.context_control.get_alias_rows()
    ui.connect()
    try:
        rejected = ui.rpc(
            "tab.run_start", tab_id=ui.tab_id, aliases={"qubit": "missing"}
        )
        assert rejected["ok"] is False
        error = rejected["error"]
        assert error["code"] == "precondition_failed"
        assert error.get("reason") == "run_aliases_not_supported"
        assert ui.controller.operation_control.active_operations() == ()
        assert cfg.input.data == before
        assert publications == []
        assert ui.controller.context_control.get_alias_rows() == rows
    finally:
        remove()
