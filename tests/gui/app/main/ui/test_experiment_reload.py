"""Local UI and remote-facing controller facets share the reload lifecycle."""

from __future__ import annotations

from collections.abc import Iterator
from dataclasses import dataclass

import pytest
from qtpy.QtWidgets import QPushButton
from zcu_tools.gui.app.main.app import _make_empty_ctx
from zcu_tools.gui.app.main.catalog import CatalogReloadError
from zcu_tools.gui.app.main.controller import Controller
from zcu_tools.gui.app.main.registry import Registry
from zcu_tools.gui.app.main.state import State
from zcu_tools.gui.app.main.ui.main_window import MainWindow
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.session.services.io_manager import IOManager

from tests.gui._dialog_fakes import RecordingDialogPresenter
from tests.gui.services.test_experiment_reload import Loader, OldAdapter


@dataclass
class WindowApp:
    ctrl: Controller
    window: MainWindow
    state: State
    loader: Loader
    dialogs: RecordingDialogPresenter


@pytest.fixture
def window_app(qapp) -> Iterator[WindowApp]:
    state = State(_make_empty_ctx())
    registry = Registry()
    registry.register("demo", OldAdapter)
    loader = Loader()
    ctrl = Controller(
        state, registry, IOManager(), None, BaseEventBus(), catalog_loader=loader
    )
    dialogs = RecordingDialogPresenter()
    window = MainWindow(ctrl, dialog_presenter=dialogs)
    ctrl.add_view(window)
    window.show()
    try:
        yield WindowApp(ctrl, window, state, loader, dialogs)
    finally:
        ctrl._background_svc.quiesce()
        window.deleteLater()
        qapp.processEvents()


def test_reload_button_cancellation_and_successful_restore(
    window_app: WindowApp,
) -> None:
    app = window_app
    first = app.ctrl.new_tab("demo")
    app.ctrl.new_tab("demo")
    app.ctrl.set_active_tab(first)
    button = next(
        button
        for button in app.window.findChildren(QPushButton)
        if button.text() == "Reload experiments"
    )
    app.dialogs.queue_destructive_confirm(False)
    button.click()
    assert app.ctrl.list_tab_ids()[0] == first
    assert app.loader.loads == 0
    app.dialogs.queue_destructive_confirm(True)
    button.click()
    assert app.loader.loads == 1
    assert len(app.ctrl.list_tab_ids()) == 2
    assert first not in app.ctrl.list_tab_ids()
    assert app.ctrl.get_active_tab_id() == app.ctrl.list_tab_ids()[0]
    assert app.window.view_tab_ids() == app.ctrl.list_tab_ids()
    app.dialogs.assert_no_unexpected_messages()


def test_catalog_failure_retry_button_reuses_original_snapshot(
    window_app: WindowApp,
) -> None:
    app = window_app
    app.ctrl.new_tab("demo")
    app.loader.failure = CatalogReloadError("broken import")
    app.dialogs.queue_destructive_confirm(True)
    app.window.reload_experiments()
    assert app.ctrl.list_tab_ids() == []
    app.dialogs.consume_message_containing("critical", "broken import")
    retry = next(
        button
        for button in app.window.findChildren(QPushButton)
        if button.text() == "Retry reload"
    )
    app.loader.failure = None
    retry.click()
    assert len(app.ctrl.list_tab_ids()) == 1
    app.dialogs.assert_no_unexpected_messages()


def test_remote_facing_facets_cannot_reenter_during_import(
    window_app: WindowApp,
) -> None:
    app = window_app
    previous = app.ctrl.new_tab("demo")

    def during_load() -> None:
        with pytest.raises(FailedPreconditionError, match="reload"):
            app.ctrl.tab_control.new_tab("demo")
        with pytest.raises(FailedPreconditionError, match="reload"):
            app.ctrl.run_analyze_control.start_run(previous)
        with pytest.raises(FailedPreconditionError, match="reload"):
            app.ctrl.run_analyze_control.load_tab_result(previous, "unused.h5")

    app.loader.during_load = during_load
    app.dialogs.queue_destructive_confirm(True)
    app.window.reload_experiments()
    assert len(app.ctrl.list_tab_ids()) == 1
    app.dialogs.assert_no_unexpected_messages()


def test_busy_save_is_rejected_before_confirmation(window_app: WindowApp) -> None:
    app = window_app
    previous = app.ctrl.new_tab("demo")
    app.state.get_tab(previous).is_saving_data = True
    app.window.reload_experiments()
    app.dialogs.consume_message_containing("critical", "Finish all operations")
    assert app.ctrl.list_tab_ids() == [previous]
    assert app.loader.loads == 0
    app.dialogs.assert_no_unexpected_messages()
