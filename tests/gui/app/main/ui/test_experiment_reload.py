"""Local UI and remote-facing controller facets share the reload lifecycle."""

from __future__ import annotations

from collections.abc import Callable, Iterator
from dataclasses import dataclass
from queue import Queue
from unittest.mock import MagicMock

import pytest
from qtpy.QtWidgets import QPushButton
from zcu_tools.gui.app.main.app import _make_empty_ctx
from zcu_tools.gui.app.main.catalog import CatalogReloadError
from zcu_tools.gui.app.main.controller import Controller
from zcu_tools.gui.app.main.registry import Registry
from zcu_tools.gui.app.main.services.remote import ControlOptions, RemoteControlAdapter
from zcu_tools.gui.app.main.state import State
from zcu_tools.gui.app.main.ui.exp_tab_widget import ExpTabWidget
from zcu_tools.gui.app.main.ui.main_window import MainWindow
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.session.adapters.qt_owner_scheduler import QtOwnerScheduler
from zcu_tools.gui.session.services.io_manager import IOManager

from tests.gui._dialog_fakes import RecordingDialogPresenter
from tests.gui.app.main.services.remote._helpers import open_client, recv_response, send
from tests.gui.app.main.services.test_experiment_reload import (
    Loader,
    NewAdapter,
    OldAdapter,
)


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


def test_cancel_after_shutdown_settles_restores_experiment_access(
    window_app: WindowApp, qapp, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = window_app
    previous = app.ctrl.new_tab("demo")
    unsaved = False
    operations = 1
    monkeypatch.setattr(ExpTabWidget, "has_unsaved_data", lambda self: unsaved)
    monkeypatch.setattr(app.ctrl, "active_operation_count", lambda: operations)

    def settle(on_closed: Callable[[], None]) -> None:
        nonlocal unsaved, operations
        with pytest.raises(FailedPreconditionError):
            app.ctrl.new_tab("demo")
        unsaved = True
        operations = 0
        on_closed()

    driver = MagicMock()
    driver.begin.side_effect = settle
    monkeypatch.setattr(app.ctrl, "_shutdown_driver", driver)
    app.dialogs.queue_confirm(True)
    app.dialogs.queue_destructive_confirm(False)
    app.window.close()
    qapp.processEvents()

    driver.begin.assert_called_once()
    assert [call.kind for call in app.dialogs.calls] == [
        "confirm",
        "destructive_confirm",
    ]
    assert app.window.isVisible()
    assert app.ctrl.list_tab_ids() == [previous]
    assert app.ctrl.new_tab("demo") != previous
    app.ctrl.prepare_experiment_reload()
    app.dialogs.assert_no_unexpected_messages()


def test_shutdown_begin_failure_restores_access(
    window_app: WindowApp, monkeypatch: pytest.MonkeyPatch
) -> None:
    app = window_app
    driver = MagicMock()
    driver.begin.side_effect = RuntimeError("shutdown failed")
    monkeypatch.setattr(app.ctrl, "_shutdown_driver", driver)
    with pytest.raises(RuntimeError, match="shutdown failed"):
        app.ctrl.begin_shutdown(lambda: None)
    app.ctrl.new_tab("demo")
    app.ctrl.prepare_experiment_reload()


class ObservedOwnerScheduler(QtOwnerScheduler):
    """Observe queue admission without dispatching inside the reload callback."""

    def __init__(self) -> None:
        super().__init__()
        self.record = False
        self.posted: Queue[None] = Queue()

    def post(self, callback: Callable[[], None]) -> None:
        super().post(callback)
        if self.record:
            self.posted.put(None)


@pytest.mark.parametrize("request_kind", ["stale", "gone", "new"])
def test_remote_requests_queued_during_reload_use_fresh_state(
    window_app: WindowApp, request_kind: str
) -> None:
    app = window_app
    previous = app.ctrl.new_tab("demo")
    app.ctrl.update_tab_cfg(previous, app.state.get_tab(previous).cfg_schema)
    versions = {
        f"tab:{previous}:cfg": app.ctrl.resources_versions()[f"tab:{previous}:cfg"]
    }
    assert next(iter(versions.values())) > 0
    scheduler = ObservedOwnerScheduler()
    remote = RemoteControlAdapter(
        controller=app.ctrl,
        opts=ControlOptions(port=0),
        owner_scheduler=scheduler,
        render_view=app.window,
    )
    port = remote.start()
    try:
        with open_client(port) as client:
            send(client, {"id": "ready", "method": "tab.list_all", "params": {}})
            assert recv_response(client, "ready")["ok"]

            def during_load() -> None:
                scheduler.record = True
                params: dict[str, object] = {"tab_id": previous}
                method = "tab.close"
                if request_kind == "stale":
                    method = "tab.run_start"
                    params["expected_versions"] = versions
                elif request_kind == "new":
                    method = "tab.new"
                    params = {"adapter_name": "demo"}
                send(client, {"id": "queued", "method": method, "params": params})
                scheduler.posted.get(timeout=3)
                scheduler.record = False
                # No event pumping: the command is admitted but not executed.
                assert app.ctrl.list_tab_ids() == []

            app.loader.during_load = during_load
            app.dialogs.queue_destructive_confirm(True)
            app.window.reload_experiments()
            restored = app.ctrl.list_tab_ids()
            assert len(restored) == 1
            assert previous not in restored
            reply = recv_response(client, "queued")
            if request_kind == "new":
                assert reply["ok"]
                created = reply["result"]["tab_id"]
                assert app.ctrl.list_tab_ids() == [*restored, created]
                assert isinstance(app.state.get_tab(created).adapter, NewAdapter)
            else:
                assert not reply["ok"]
                assert app.ctrl.list_tab_ids() == restored
                if request_kind == "stale":
                    assert reply["error"]["reason"] == "stale_version"
                else:
                    assert "unknown tab_id" in reply["error"]["message"]
            app.dialogs.assert_no_unexpected_messages()
    finally:
        remote.stop()


def test_busy_save_is_rejected_before_confirmation(window_app: WindowApp) -> None:
    app = window_app
    previous = app.ctrl.new_tab("demo")
    app.state.get_tab(previous).is_saving_data = True
    app.window.reload_experiments()
    app.dialogs.consume_message_containing("critical", "Finish all operations")
    assert app.ctrl.list_tab_ids() == [previous]
    assert app.loader.loads == 0
    app.dialogs.assert_no_unexpected_messages()
