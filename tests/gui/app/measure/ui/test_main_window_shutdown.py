"""MainWindow shutdown decisions, draining and persistence contracts."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from unittest.mock import MagicMock

import pytest
from qtpy.QtCore import QCoreApplication
from qtpy.QtWidgets import (
    QLabel,
    QStackedWidget,
)
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
from zcu_tools.gui.plotting import FigureContainer
from zcu_tools.gui.plotting.explicit import QtPlotHost
from zcu_tools.gui.session.adapters.qt_owner_scheduler import QtOwnerScheduler
from zcu_tools.gui.widgets.dialog_presenter import DestructiveConfirmation
from zcu_tools.plotting.plots import Plots

from tests.gui._dialog_fakes import RecordingDialogPresenter
from tests.gui.app.measure._form_fakes import configure_form_lookup
from tests.gui.app.measure.ui._main_window_fakes import (
    add_tab,
    experiment_tabs,
    make_tab_snapshot,
    mock_ctrl,
    setup_window_with_tabs,
)


def _shutdown_callback(ctrl: MagicMock) -> Callable[[], None]:
    ctrl.begin_shutdown.assert_called_once()
    callback: Callable[[], None] = ctrl.begin_shutdown.call_args.args[0]
    assert callable(callback)
    return callback


class _DeferredClosePresenter(RecordingDialogPresenter):
    def __init__(self) -> None:
        super().__init__(confirm_answers=[True] * 4, destructive_answers=[True] * 4)
        self.confirm_decisions: list[Callable[[bool], None]] = []
        self.destructive_decisions: list[Callable[[bool], None]] = []

    def confirm_async(
        self,
        parent: Any,
        title: str,
        message: str,
        *,
        on_decision: Callable[[bool], None],
        default: bool = False,
    ) -> None:
        super().confirm_async(
            parent,
            title,
            message,
            on_decision=lambda _confirmed: None,
            default=default,
        )
        self.confirm_decisions.append(on_decision)

    def destructive_confirm(
        self,
        parent: Any,
        confirmation: DestructiveConfirmation,
        *,
        on_decision: Callable[[bool], None],
    ) -> None:
        """Retain the decision callback while recording both button labels."""
        super().destructive_confirm(
            parent,
            confirmation,
            on_decision=lambda _confirmed: None,
        )
        self.destructive_decisions.append(on_decision)


def test_main_window_confirms_and_begins_shutdown_when_operations_active(
    main_window_control_factory,
    qapp,
):
    """User close with work in progress: confirm, then begin_shutdown (which
    cancels-all and waits). The event is ignored now; begin_shutdown is deferred
    to the next event-loop turn and the coordinator drives the real close
    later."""
    from qtpy.QtCore import QCoreApplication
    from qtpy.QtGui import QCloseEvent
    from zcu_tools.gui.app.measure.ui.main_window import MainWindow

    dialogs = RecordingDialogPresenter(confirm_answers=[True])
    ctrl = mock_ctrl(main_window_control_factory)
    configure_form_lookup(ctrl)
    ctrl.get_bus.return_value = EventBus()
    ctrl.active_operation_count.return_value = 2
    window = MainWindow(ctrl, templates=lambda: {}, dialog_presenter=dialogs)
    event = QCloseEvent()

    window.closeEvent(event)

    assert dialogs.calls[-1].title == "Operations in progress"
    assert event.isAccepted() is False  # async wait — not closed yet
    QCoreApplication.processEvents()  # drain the deferred singleShot(0)
    _shutdown_callback(ctrl)


def test_main_window_declining_confirmation_keeps_window_open(
    main_window_control_factory, qapp
):
    from qtpy.QtGui import QCloseEvent
    from zcu_tools.gui.app.measure.ui.main_window import MainWindow

    dialogs = RecordingDialogPresenter(confirm_answers=[False])
    ctrl = mock_ctrl(main_window_control_factory)
    configure_form_lookup(ctrl)
    ctrl.get_bus.return_value = EventBus()
    ctrl.active_operation_count.return_value = 1
    window = MainWindow(ctrl, templates=lambda: {}, dialog_presenter=dialogs)
    event = QCloseEvent()

    window.closeEvent(event)

    assert dialogs.calls[-1].title == "Operations in progress"
    assert event.isAccepted() is False
    ctrl.begin_shutdown.assert_not_called()


def test_main_window_persists_session_on_close_when_idle(
    main_window_control_factory, qapp
):
    """Idle close: no confirmation; closeEvent ignores the event and *defers*
    begin_shutdown to the next event-loop turn (so it never re-enters
    self.close() within the closeEvent stack — the single-click bug). After the
    deferred turn fires, begin_shutdown runs _perform_close → persist."""
    from qtpy.QtCore import QCoreApplication
    from qtpy.QtGui import QCloseEvent
    from zcu_tools.gui.app.measure.ui.main_window import MainWindow

    ctrl = mock_ctrl(main_window_control_factory)
    configure_form_lookup(ctrl)
    ctrl.get_bus.return_value = EventBus()
    ctrl.active_operation_count.return_value = 0
    # The real coordinator runs on_closed once nothing is pending; here drive it
    # synchronously so the deferred turn exercises _perform_close.
    ctrl.begin_shutdown.side_effect = lambda on_closed: on_closed()
    window = MainWindow(ctrl, templates=lambda: {})
    event = QCloseEvent()

    window.closeEvent(event)

    # Deferred: closeEvent must NOT have begun shutdown synchronously.
    assert event.isAccepted() is False
    ctrl.begin_shutdown.assert_not_called()

    # Drain the singleShot(0) — the deferred turn runs begin_shutdown.
    QCoreApplication.processEvents()

    _shutdown_callback(ctrl)
    ctrl.persist_all.assert_called_once_with()


def test_main_window_close_removes_event_bus_subscriptions(
    main_window_control_factory, qapp, monkeypatch
):
    from qtpy.QtCore import QCoreApplication
    from qtpy.QtGui import QCloseEvent
    from zcu_tools.gui.app.measure.events.run import (
        RunFinishedPayload,
        RunStartedPayload,
    )
    from zcu_tools.gui.app.measure.events.tab import (
        TabAddedPayload,
        TabClosedPayload,
        TabContentChangedPayload,
        TabInteractionChangedPayload,
    )
    from zcu_tools.gui.app.measure.ui.main_window import MainWindow
    from zcu_tools.gui.session.events import (
        ContextSwitchedPayload,
        SocChangedPayload,
    )

    ctrl = mock_ctrl(main_window_control_factory)
    configure_form_lookup(ctrl)
    bus = EventBus()
    subscribe_spy = MagicMock(wraps=bus.subscribe)
    unsubscribe_spy = MagicMock(wraps=bus.unsubscribe)
    monkeypatch.setattr(bus, "subscribe", subscribe_spy)
    monkeypatch.setattr(bus, "unsubscribe", unsubscribe_spy)
    ctrl.get_bus.return_value = bus
    ctrl.active_operation_count.return_value = 0
    ctrl.begin_shutdown.side_effect = lambda on_closed: on_closed()
    window = MainWindow(ctrl, templates=lambda: {})

    payload_types = (
        TabInteractionChangedPayload,
        RunStartedPayload,
        RunFinishedPayload,
        ContextSwitchedPayload,
        TabAddedPayload,
        TabClosedPayload,
        TabContentChangedPayload,
        SocChangedPayload,
    )
    for payload_type in payload_types:
        subscription_calls = [
            call
            for call in subscribe_spy.call_args_list
            if call.args[0] is payload_type
        ]
        assert subscription_calls
    unsubscribe_spy.assert_not_called()

    window.closeEvent(QCloseEvent())
    QCoreApplication.processEvents()

    for payload_type in payload_types:
        subscription_calls = [
            call
            for call in subscribe_spy.call_args_list
            if call.args[0] is payload_type
        ]
        for subscription_call in subscription_calls:
            assert unsubscribe_spy.call_args_list.count(
                subscription_call
            ) == subscription_calls.count(subscription_call)


def test_window_drains_before_persisting_and_stopping_remote(
    main_window_control_factory, qapp
):
    dialogs = RecordingDialogPresenter()
    window, ctrl = setup_window_with_tabs(main_window_control_factory, dialogs)
    effects: list[str] = []
    ctrl.close.side_effect = lambda: effects.append("drained")
    ctrl.persist_all.side_effect = lambda: effects.append("persisted")
    remote = MagicMock()
    remote.stop.side_effect = lambda: effects.append("remote stopped")
    window.remote_control_service = remote
    ctrl.begin_shutdown.side_effect = lambda on_closed: on_closed()
    window.show()

    window.close()
    qapp.processEvents()

    assert effects == ["drained", "persisted", "remote stopped"]
    assert not window.isVisible()
    assert dialogs.calls == []


@pytest.mark.parametrize("programmatic", [False, True])
def test_window_timeout_waits_for_decision_and_retries(
    main_window_control_factory, qapp, programmatic
):
    dialogs = _DeferredClosePresenter()
    window, ctrl = setup_window_with_tabs(main_window_control_factory, dialogs)
    ctrl.begin_shutdown.side_effect = lambda on_closed: on_closed()
    ctrl.close.side_effect = [
        TimeoutError("still running"),
        TimeoutError("still running"),
        None,
    ]
    remote = MagicMock()
    window.remote_control_service = remote
    window.show()

    if programmatic:
        window.request_shutdown()
    else:
        window.close()
    qapp.processEvents()

    assert window.isVisible()
    ctrl.persist_all.assert_not_called()
    remote.stop.assert_not_called()
    assert len(dialogs.destructive_decisions) == 1
    assert "背景工作仍在執行" in dialogs.calls[-1].message
    assert dialogs.calls[-1].action_text == "強制關閉"
    assert dialogs.calls[-1].cancel_text == "再等一下"

    # Reentrant GUI/RPC requests cannot turn a pending decision into force.
    window.close()
    window.request_shutdown()
    window.request_shutdown()
    qapp.processEvents()
    assert window.isVisible()
    assert len(dialogs.destructive_decisions) == 1
    assert ctrl.close.call_count == 1

    dialogs.destructive_decisions.pop(0)(False)
    qapp.processEvents()
    assert window.isVisible()
    assert len(dialogs.destructive_decisions) == 1
    assert ctrl.close.call_count == 2
    ctrl.persist_all.assert_not_called()
    remote.stop.assert_not_called()

    dialogs.destructive_decisions.pop(0)(False)
    qapp.processEvents()
    assert not window.isVisible()
    ctrl.persist_all.assert_called_once()
    remote.stop.assert_called_once()
    assert ctrl.close.call_count == 3


@pytest.mark.parametrize("programmatic", [False, True])
def test_window_timeout_requires_explicit_force(
    main_window_control_factory, qapp, programmatic
):
    dialogs = _DeferredClosePresenter()
    window, ctrl = setup_window_with_tabs(main_window_control_factory, dialogs)
    ctrl.begin_shutdown.side_effect = lambda on_closed: on_closed()
    ctrl.close.side_effect = TimeoutError("still running")
    remote = MagicMock()
    window.remote_control_service = remote
    window.show()

    if programmatic:
        window.request_shutdown()
    else:
        window.close()
    qapp.processEvents()

    assert window.isVisible()
    ctrl.persist_all.assert_not_called()
    remote.stop.assert_not_called()
    assert len(dialogs.destructive_decisions) == 1
    dialogs.destructive_decisions.pop(0)(True)
    qapp.processEvents()

    assert not window.isVisible()
    ctrl.persist_all.assert_called_once()
    remote.stop.assert_called_once()
    assert ctrl.close.call_count == 1


def test_app_close_idle_with_no_unsaved_tabs_shuts_down_without_prompt(
    main_window_control_factory, qapp
):
    """Scenario 6: App close with neither unsaved data nor active operations closes cleanly."""
    dialogs = RecordingDialogPresenter()
    window, ctrl = setup_window_with_tabs(
        main_window_control_factory, dialogs, active_operations=0
    )
    add_tab(window, "tab-1", has_run=True, saved=True, ctrl=ctrl)

    window.close()

    assert len(dialogs.calls) == 0
    QCoreApplication.processEvents()
    _shutdown_callback(ctrl)


def test_closing_window_leaves_plot_host_available_until_app_quits(
    main_window_control_factory, qapp
):
    window, _ctrl = setup_window_with_tabs(
        main_window_control_factory, RecordingDialogPresenter(), active_operations=0
    )
    window.close()
    QCoreApplication.processEvents()

    stack = QStackedWidget()
    placeholder = QLabel("empty")
    stack.addWidget(placeholder)
    container = FigureContainer(stack, placeholder)
    owner = QtOwnerScheduler()
    try:
        plots = Plots(QtPlotHost(container, owner))
        plots.subplots("fresh")
        plots.finish()
        assert stack.count() == 2
        plots.release()
        assert stack.count() == 1
    finally:
        container.clear_dynamic_canvases()
        owner.deleteLater()
        stack.deleteLater()
        window.deleteLater()
        qapp.processEvents()


def test_app_close_with_unsaved_tabs_prompts_and_declining_cancels(
    main_window_control_factory, qapp
):
    """Scenario 4: App close with unsaved data warns; cancelling leaves app open."""
    dialogs = RecordingDialogPresenter(destructive_answers=[False])
    window, ctrl = setup_window_with_tabs(
        main_window_control_factory, dialogs, active_operations=0
    )
    add_tab(window, "tab-1", has_run=True, saved=False, ctrl=ctrl)

    window.close()

    assert len(dialogs.calls) == 1
    call = dialogs.calls[0]
    assert call.kind == "destructive_confirm"
    assert call.title == "Unsaved measurement data"
    assert call.action_text == "Discard and Close"
    QCoreApplication.processEvents()
    ctrl.begin_shutdown.assert_not_called()


def test_app_close_with_unsaved_tabs_prompts_and_confirming_shuts_down(
    main_window_control_factory, qapp
):
    """Scenario 4: App close with unsaved data warns; confirming begins shutdown."""
    dialogs = RecordingDialogPresenter(destructive_answers=[True])
    window, ctrl = setup_window_with_tabs(
        main_window_control_factory, dialogs, active_operations=0
    )
    add_tab(window, "tab-1", has_run=True, saved=False, ctrl=ctrl)
    add_tab(window, "tab-2", has_run=False, ctrl=ctrl)

    window.close()

    assert len(dialogs.calls) == 1
    call = dialogs.calls[0]
    assert call.kind == "destructive_confirm"
    assert call.title == "Unsaved measurement data"
    assert call.action_text == "Discard and Close"
    QCoreApplication.processEvents()
    _shutdown_callback(ctrl)


def test_app_close_with_both_unsaved_tabs_and_active_operations(
    main_window_control_factory, qapp
):
    """Scenario 5: Both risks present; prompt communicates both, confirm shuts down."""
    dialogs = RecordingDialogPresenter(destructive_answers=[True])
    window, ctrl = setup_window_with_tabs(
        main_window_control_factory, dialogs, active_operations=3
    )
    add_tab(window, "tab-1", has_run=True, saved=False, ctrl=ctrl)

    window.close()

    assert len(dialogs.calls) == 1
    call = dialogs.calls[0]
    assert call.kind == "destructive_confirm"
    assert call.title == "Unsaved data and operations in progress"
    assert "unsaved measurement data" in call.message.lower()
    assert "3 operation(s)" in call.message
    assert call.action_text == "Discard and Close"
    QCoreApplication.processEvents()
    _shutdown_callback(ctrl)


def test_app_close_with_both_unsaved_tabs_and_active_operations_declined(
    main_window_control_factory, qapp
):
    """Scenario 5: Both risks present; decline preserves app state."""
    dialogs = RecordingDialogPresenter(destructive_answers=[False])
    window, ctrl = setup_window_with_tabs(
        main_window_control_factory, dialogs, active_operations=2
    )
    add_tab(window, "tab-1", has_run=True, saved=False, ctrl=ctrl)

    window.close()

    assert len(dialogs.calls) == 1
    call = dialogs.calls[0]
    assert call.kind == "destructive_confirm"
    assert call.title == "Unsaved data and operations in progress"
    assert call.action_text == "Discard and Close"
    QCoreApplication.processEvents()
    ctrl.begin_shutdown.assert_not_called()


def test_app_close_with_active_operations_only_uses_confirm_async(
    main_window_control_factory, qapp
):
    """Active-operation-only prompt uses non-blocking confirm_async."""
    dialogs = RecordingDialogPresenter(confirm_answers=[True])
    window, ctrl = setup_window_with_tabs(
        main_window_control_factory, dialogs, active_operations=2
    )
    add_tab(window, "tab-1", has_run=True, saved=True, ctrl=ctrl)

    window.close()

    assert len(dialogs.calls) == 1
    call = dialogs.calls[0]
    assert call.kind == "confirm"
    assert call.title == "Operations in progress"
    assert "2 operation(s)" in call.message
    QCoreApplication.processEvents()
    _shutdown_callback(ctrl)


def test_app_close_suppresses_duplicate_prompt_while_decision_pending(
    main_window_control_factory, qapp
):
    """Scenario 4: Duplicate app close requests while prompt is open are suppressed."""
    decision_cb: list[Callable[[bool], None]] = []

    class _DeferredPresenter(RecordingDialogPresenter):
        def destructive_confirm(
            self,
            parent: Any,
            confirmation: DestructiveConfirmation,
            *,
            on_decision: Callable[[bool], None],
        ) -> None:
            """Retain the decision and preserve the alternative button label."""
            super().destructive_confirm(
                parent,
                confirmation,
                on_decision=lambda _: None,
            )
            decision_cb.append(on_decision)

    deferred_presenter = _DeferredPresenter(destructive_answers=[True])
    window, ctrl = setup_window_with_tabs(
        main_window_control_factory, deferred_presenter, active_operations=0
    )
    add_tab(window, "tab-1", has_run=True, saved=False, ctrl=ctrl)

    # First close
    window.close()
    assert len(deferred_presenter.calls) == 1

    # Second close while dialog is pending
    window.close()
    assert len(deferred_presenter.calls) == 1

    decision_cb[0](True)
    QCoreApplication.processEvents()
    _shutdown_callback(ctrl)


def test_app_close_reprompts_if_unsaved_data_appears_before_decision(
    main_window_control_factory, qapp
):
    """A newly completed result upgrades an active-only prompt before shutdown."""
    dialogs = _DeferredClosePresenter()
    window, ctrl = setup_window_with_tabs(
        main_window_control_factory, dialogs, active_operations=1
    )
    tab = add_tab(window, "tab-1", has_run=True, saved=True, ctrl=ctrl)

    window.close()

    assert [call.kind for call in dialogs.calls] == ["confirm"]
    tab.update_interaction_state(make_tab_snapshot("tab-1", has_run_result=True))
    ctrl.active_operation_count.return_value = 0
    dialogs.confirm_decisions.pop()(True)

    assert [call.kind for call in dialogs.calls] == [
        "confirm",
        "destructive_confirm",
    ]
    assert dialogs.calls[-1].title == "Unsaved measurement data"
    ctrl.begin_shutdown.assert_not_called()

    dialogs.destructive_decisions.pop()(False)
    ctrl.begin_shutdown.assert_not_called()


def test_app_close_suppresses_reentry_while_shutdown_is_coordinating(
    main_window_control_factory, qapp
):
    """Accepted close owns the lifecycle until the coordinator callback runs."""
    dialogs = RecordingDialogPresenter(confirm_answers=[True])
    window, ctrl = setup_window_with_tabs(
        main_window_control_factory, dialogs, active_operations=1
    )
    add_tab(window, "tab-1", has_run=True, saved=True, ctrl=ctrl)

    window.close()
    window.close()
    QCoreApplication.processEvents()
    _shutdown_callback(ctrl)

    window.close()

    assert len(dialogs.calls) == 1
    ctrl.begin_shutdown.assert_called_once()


def test_app_close_suppresses_duplicate_idle_shutdown_requests(
    main_window_control_factory, qapp
):
    """Repeated idle close events schedule only one shutdown coordinator run."""
    dialogs = RecordingDialogPresenter()
    window, ctrl = setup_window_with_tabs(
        main_window_control_factory, dialogs, active_operations=0
    )
    add_tab(window, "tab-1", has_run=True, saved=True, ctrl=ctrl)

    window.close()
    window.close()
    QCoreApplication.processEvents()
    _shutdown_callback(ctrl)

    window.close()

    assert dialogs.calls == []
    ctrl.begin_shutdown.assert_called_once()


def test_app_close_rechecks_unsaved_data_after_operations_settle(
    main_window_control_factory, qapp
):
    """A result completed during shutdown gets a final data-loss decision."""
    dialogs = RecordingDialogPresenter(
        confirm_answers=[True], destructive_answers=[False]
    )
    window, ctrl = setup_window_with_tabs(
        main_window_control_factory, dialogs, active_operations=1
    )
    tab = add_tab(window, "tab-1", has_run=True, saved=True, ctrl=ctrl)

    window.close()
    QCoreApplication.processEvents()
    on_closed = _shutdown_callback(ctrl)

    tab.update_interaction_state(make_tab_snapshot("tab-1", has_run_result=True))
    ctrl.active_operation_count.return_value = 0
    on_closed()

    assert [call.kind for call in dialogs.calls] == [
        "confirm",
        "destructive_confirm",
    ]
    assert "completed with unsaved measurement data" in dialogs.calls[-1].message
    ctrl.persist_all.assert_not_called()

    dialogs.queue_destructive_confirm(False)
    window.close()
    assert [call.kind for call in dialogs.calls] == [
        "confirm",
        "destructive_confirm",
        "destructive_confirm",
    ]


def test_app_close_waits_for_pending_tab_close_decision(
    main_window_control_factory, qapp
):
    """Tab and app close confirmations never overlap."""
    dialogs = _DeferredClosePresenter()
    window, ctrl = setup_window_with_tabs(
        main_window_control_factory, dialogs, active_operations=0
    )
    add_tab(window, "tab-1", has_run=True, saved=False, ctrl=ctrl)

    experiment_tabs(window).tabCloseRequested.emit(0)
    window.close()

    assert [call.kind for call in dialogs.calls] == ["destructive_confirm"]
    ctrl.begin_shutdown.assert_not_called()

    dialogs.destructive_decisions.pop()(False)


def test_repeated_programmatic_shutdown_waits_for_pre_prompt_coordination(
    main_window_control_factory, qapp
):
    """Repeated RPC takeover cannot bypass the captured shutdown coordinator."""
    dialogs = _DeferredClosePresenter()
    window, ctrl = setup_window_with_tabs(
        main_window_control_factory, dialogs, active_operations=1
    )
    add_tab(window, "tab-1", has_run=True, saved=True, ctrl=ctrl)

    window.close()
    assert [call.kind for call in dialogs.calls] == ["confirm"]

    window.request_shutdown()
    QCoreApplication.processEvents()
    on_closed = _shutdown_callback(ctrl)

    window.request_shutdown()
    QCoreApplication.processEvents()
    ctrl.persist_all.assert_not_called()

    dialogs.confirm_decisions.pop()(True)
    ctrl.persist_all.assert_not_called()

    on_closed()
    ctrl.persist_all.assert_called_once_with()


def test_programmatic_shutdown_supersedes_user_shutdown_coordination(
    main_window_control_factory, qapp
):
    """RPC intent forces close when user-started coordination later settles."""
    dialogs = RecordingDialogPresenter(confirm_answers=[True])
    window, ctrl = setup_window_with_tabs(
        main_window_control_factory, dialogs, active_operations=1
    )
    tab = add_tab(window, "tab-1", has_run=True, saved=True, ctrl=ctrl)

    window.close()
    QCoreApplication.processEvents()
    on_closed = _shutdown_callback(ctrl)

    window.request_shutdown()
    tab.update_interaction_state(make_tab_snapshot("tab-1", has_run_result=True))
    ctrl.active_operation_count.return_value = 0
    on_closed()

    assert [call.kind for call in dialogs.calls] == ["confirm"]
    ctrl.begin_shutdown.assert_called_once()
    ctrl.persist_all.assert_called_once_with()


def test_programmatic_shutdown_supersedes_final_unsaved_prompt(
    main_window_control_factory, qapp
):
    """RPC intent tears down without waiting for a post-settlement dialog."""
    dialogs = _DeferredClosePresenter()
    window, ctrl = setup_window_with_tabs(
        main_window_control_factory, dialogs, active_operations=1
    )
    tab = add_tab(window, "tab-1", has_run=True, saved=True, ctrl=ctrl)

    window.close()
    dialogs.confirm_decisions.pop()(True)
    QCoreApplication.processEvents()
    on_closed = _shutdown_callback(ctrl)

    tab.update_interaction_state(make_tab_snapshot("tab-1", has_run_result=True))
    ctrl.active_operation_count.return_value = 0
    on_closed()
    assert [call.kind for call in dialogs.calls] == [
        "confirm",
        "destructive_confirm",
    ]

    window.request_shutdown()
    ctrl.persist_all.assert_not_called()
    QCoreApplication.processEvents()
    ctrl.persist_all.assert_called_once_with()

    dialogs.destructive_decisions.pop()(False)
    ctrl.persist_all.assert_called_once_with()


def test_programmatic_request_shutdown_bypasses_unsaved_guard(
    main_window_control_factory, qapp
):
    """Programmatic request_shutdown() remains non-interactive even with unsaved data."""
    dialogs = RecordingDialogPresenter()
    window, ctrl = setup_window_with_tabs(
        main_window_control_factory, dialogs, active_operations=1
    )
    add_tab(window, "tab-1", has_run=True, saved=False, ctrl=ctrl)

    window.request_shutdown()

    assert len(dialogs.calls) == 0
    ctrl.begin_shutdown.assert_not_called()
    QCoreApplication.processEvents()
    ctrl.close.assert_not_called()
    ctrl.persist_all.assert_not_called()
    _shutdown_callback(ctrl)()
    ctrl.close.assert_called_once_with()
    ctrl.persist_all.assert_called_once_with()
    assert len(dialogs.calls) == 0
