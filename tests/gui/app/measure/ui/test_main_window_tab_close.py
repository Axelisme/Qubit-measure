"""MainWindow unsaved tab close decisions and captured identity contracts."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

import pytest
from qtpy.QtWidgets import (
    QTabWidget,
)
from zcu_tools.gui.app.measure.artifacts import (
    SaveStatus,
)
from zcu_tools.gui.widgets.dialog_presenter import DestructiveConfirmation

from tests.gui._dialog_fakes import RecordingDialogPresenter
from tests.gui.app.measure.ui._main_window_fakes import (
    add_tab,
    experiment_tabs,
    make_tab_snapshot,
    setup_window_with_tabs,
)


def test_tab_close_without_result_closes_immediately(main_window_control_factory, qapp):
    """Scenario 1: Tab with NO_RESULT closes without prompting."""
    dialogs = RecordingDialogPresenter()
    window, ctrl = setup_window_with_tabs(main_window_control_factory, dialogs)
    add_tab(window, "tab-1", has_run=False, ctrl=ctrl)

    experiment_tabs(window).tabCloseRequested.emit(0)

    assert len(dialogs.calls) == 0
    ctrl.close_tab.assert_called_once_with("tab-1")


def test_tab_close_with_saved_data_closes_immediately(
    main_window_control_factory, qapp
):
    """Scenario 1: Tab with SAVED data closes without prompting."""
    dialogs = RecordingDialogPresenter()
    window, ctrl = setup_window_with_tabs(main_window_control_factory, dialogs)
    tab = add_tab(window, "tab-1", has_run=True, saved=True, ctrl=ctrl)
    assert tab.has_unsaved_data() is False

    experiment_tabs(window).tabCloseRequested.emit(0)

    assert len(dialogs.calls) == 0
    ctrl.close_tab.assert_called_once_with("tab-1")


@pytest.mark.parametrize(
    "data_status", [SaveStatus.NOT_SAVED, SaveStatus.UNSAVED_CHANGES]
)
def test_tab_close_with_unsaved_data_declined_keeps_tab(
    main_window_control_factory, qapp, data_status
):
    """Cancelling tab-close leaves either unsaved data state untouched."""
    dialogs = RecordingDialogPresenter(destructive_answers=[False])
    window, ctrl = setup_window_with_tabs(main_window_control_factory, dialogs)
    tab = add_tab(window, "tab-1", has_run=True, saved=False, ctrl=ctrl)
    tab.update_interaction_state(make_tab_snapshot("tab-1", data_status=data_status))
    assert tab.has_unsaved_data() is True

    experiment_tabs(window).tabCloseRequested.emit(0)

    assert len(dialogs.calls) == 1
    call = dialogs.calls[0]
    assert call.kind == "destructive_confirm"
    assert call.title == "Unsaved measurement data"
    assert call.action_text == "Discard and Close"
    ctrl.close_tab.assert_not_called()
    assert window.has_tab_widget("tab-1") is True


def test_tab_close_with_unsaved_data_confirmed_closes_tab(
    main_window_control_factory, qapp
):
    """Scenario 2: Confirming tab-close closes the unsaved tab."""
    dialogs = RecordingDialogPresenter(destructive_answers=[True])
    window, ctrl = setup_window_with_tabs(main_window_control_factory, dialogs)
    tab = add_tab(window, "tab-1", has_run=True, saved=False, ctrl=ctrl)
    assert tab.has_unsaved_data() is True

    experiment_tabs(window).tabCloseRequested.emit(0)

    assert len(dialogs.calls) == 1
    call = dialogs.calls[0]
    assert call.kind == "destructive_confirm"
    assert call.title == "Unsaved measurement data"
    assert call.action_text == "Discard and Close"
    ctrl.close_tab.assert_called_once_with("tab-1")


def test_moved_tab_with_unsaved_data_closes_by_captured_identity(
    main_window_control_factory, qapp
):
    """Scenario 3: A tab moved before close request closes by captured widget identity."""
    dialogs = RecordingDialogPresenter(destructive_answers=[True])
    window, ctrl = setup_window_with_tabs(main_window_control_factory, dialogs)
    tab_a = add_tab(window, "tab-a", has_run=True, saved=False, ctrl=ctrl)
    add_tab(window, "tab-b", has_run=False, ctrl=ctrl)
    assert tab_a.has_unsaved_data() is True

    tab_bar = experiment_tabs(window).tabBar()
    assert tab_bar is not None
    # Move tab-a from index 0 to index 1
    tab_bar.moveTab(0, 1)
    assert experiment_tabs(window).widget(1) is tab_a

    # Request close on moved tab (now at index 1)
    experiment_tabs(window).tabCloseRequested.emit(1)

    assert len(dialogs.calls) == 1
    assert dialogs.calls[0].kind == "destructive_confirm"
    assert dialogs.calls[0].title == "Unsaved measurement data"
    ctrl.close_tab.assert_called_once_with("tab-a")


def test_tab_moved_while_confirmation_pending_closes_captured_widget(
    main_window_control_factory, qapp
):
    """Scenario 3: Tab moved while confirmation prompt is open closes captured identity."""
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
        main_window_control_factory, deferred_presenter
    )
    tab_a = add_tab(window, "tab-a", has_run=True, saved=False, ctrl=ctrl)
    add_tab(window, "tab-b", has_run=False, ctrl=ctrl)

    tabs = window.findChild(QTabWidget, "experiment.tabs")
    assert tabs is not None
    # Request close on tab-a at index 0 via signal
    tabs.tabCloseRequested.emit(0)
    assert len(decision_cb) == 1

    # While dialog is pending, move tab-a to index 1
    tab_bar = tabs.tabBar()
    assert tab_bar is not None
    tab_bar.moveTab(0, 1)
    assert tabs.widget(1) is tab_a

    # User confirms the dialog
    decision_cb[0](True)

    ctrl.close_tab.assert_called_once_with("tab-a")


def test_tab_close_suppresses_duplicate_prompt_while_decision_pending(
    main_window_control_factory, qapp
):
    """Scenario 2/3: Duplicate tab close requests for same tab are suppressed while prompt is open."""
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
        main_window_control_factory, deferred_presenter
    )
    add_tab(window, "tab-1", has_run=True, saved=False, ctrl=ctrl)

    tabs = window.findChild(QTabWidget, "experiment.tabs")
    assert tabs is not None
    # First request
    tabs.tabCloseRequested.emit(0)
    assert len(deferred_presenter.calls) == 1

    # Second request while prompt is pending
    tabs.tabCloseRequested.emit(0)
    assert len(deferred_presenter.calls) == 1

    # Confirm
    decision_cb[0](True)
    ctrl.close_tab.assert_called_once_with("tab-1")
