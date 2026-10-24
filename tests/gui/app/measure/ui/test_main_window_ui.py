"""MainWindow loading, dialogs, status and tab routing contracts."""

from __future__ import annotations

from collections.abc import Iterator
from pathlib import Path
from typing import ClassVar
from unittest.mock import MagicMock

import pytest
from pydantic import field_validator
from qtpy.QtCore import QCoreApplication, QEvent, Qt
from qtpy.QtWidgets import (
    QApplication,
    QLabel,
    QPushButton,
    QTabWidget,
)
from zcu_tools.cfg_model import ConfigBase
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    AnalysisMode,
    ParameterWriteback,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.adapter.base import BaseAdapter
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.state import State
from zcu_tools.gui.app.measure.ui.exp_tab_widget import ExpTabWidget
from zcu_tools.gui.app.measure.ui.main_window import MainWindow
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.gui.session.context_types import (
    CommandOrigin,
    EntryRequest,
    PointCreateRequest,
)
from zcu_tools.gui.session.events import SocChangedPayload
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.resources.entry import (
    RoleResolutionError,
)

from tests.gui._dialog_fakes import RecordingDialogPresenter
from tests.gui._qt_waits import qt_until
from tests.gui._user_input import ComponentDriver
from tests.gui.app.measure._form_fakes import configure_form_lookup
from tests.gui.app.measure._synthetic import SampleAdapter, SampleAnalysis, SampleRun
from tests.gui.app.measure.ui._main_window_fakes import (
    apply_window_defaults,
    editor_wiring_ctrl,
    experiment_tabs,
    make_tab_snapshot,
    mock_ctrl,
)


def test_main_window_role_failure_is_presented_without_registering_tab(
    main_window_control_factory, qapp
):
    ctrl = apply_window_defaults(mock_ctrl(main_window_control_factory))
    ctrl.get_bus.return_value = EventBus()
    ctrl.new_tab.side_effect = RoleResolutionError(
        "qubit", "R1", "qubit/*", "wrong kind"
    )
    dialogs = RecordingDialogPresenter()
    window = MainWindow(ctrl, templates=lambda: {}, dialog_presenter=dialogs)
    window.create_tab("fake")
    assert len(dialogs.calls) == 1
    assert "qubit" in dialogs.calls[0].message
    tabs = window.findChild(QTabWidget, "experiment.tabs")
    assert tabs is not None and tabs.count() == 0


def _new_tab_button(window: MainWindow) -> QPushButton:
    """Return the New Tab button by its label, failing unless it is unique."""
    buttons = [
        button
        for button in window.findChildren(QPushButton)
        if button.text() == "New Tab ▾"
    ]
    assert len(buttons) == 1
    return buttons[0]


@pytest.mark.parametrize("analysis_error", [None, "analysis defaults unavailable"])
def test_main_window_load_data_dialog_calls_controller(
    main_window_control_factory, qapp, monkeypatch, tmp_path, analysis_error
):
    from qtpy.QtWidgets import QFileDialog
    from zcu_tools.gui.app.measure.services.load import LoadTabResultOutcome
    from zcu_tools.gui.app.measure.ui.main_window import MainWindow
    from zcu_tools.gui.app.measure.ui.run_load_dialog import RunLoadDialog

    ctrl = apply_window_defaults(mock_ctrl(main_window_control_factory))
    ctrl.get_bus.return_value = EventBus()
    ctrl.has_tab.return_value = True
    ctrl.get_tab_snapshot.return_value = make_tab_snapshot(
        "tab-1", supports_load_data=True, has_figure=False
    )
    database_root = tmp_path / "Database" / "Q3_2D" / "Q1"
    database_root.mkdir(parents=True)
    ctrl.save_control.list_saved_runs.return_value = ()
    ctrl.get_session_env.return_value = SessionEnv(
        parameters=None,
        soc=None,
        soccfg=None,
        database_path=str(database_root / "2026" / "06" / "Data_0625"),
    )
    ctrl.load_tab_result.return_value = LoadTabResultOutcome(
        tab_id="tab-1",
        data_path="/tmp/result.hdf5",
        result_type="Result",
        cfg_issue="cfg unavailable",
        has_analyze_params=analysis_error is None,
        analysis_error=analysis_error,
    )
    dialogs = RecordingDialogPresenter()
    window = MainWindow(ctrl, templates=lambda: {}, dialog_presenter=dialogs)
    window.add_tab_widget("tab-1", "fake")
    try:
        assert window.has_tab_widget("tab-1") is True
        window.show_status_message = MagicMock()

        captured_dir: dict[str, str] = {}

        def fake_get_open_file_name(*args, **kwargs):
            captured_dir["directory"] = args[2]
            return ("/tmp/result.hdf5", "")

        monkeypatch.setattr(
            QFileDialog,
            "getOpenFileName",
            fake_get_open_file_name,
        )

        def choose_file(dialog: RunLoadDialog) -> int:
            button = dialog.findChild(QPushButton, "load_file")
            assert button is not None
            button.click()
            return dialog.result()

        monkeypatch.setattr(RunLoadDialog, "exec", choose_file)
        window.load_tab_data_dialog("tab-1")

        ctrl.load_tab_result.assert_called_once_with("tab-1", "/tmp/result.hdf5")
        window.show_status_message.assert_called_once()
        assert captured_dir["directory"] == str(database_root)
        if analysis_error is not None:
            assert [(call.kind, call.title) for call in dialogs.calls] == [
                ("warning", "Analysis preparation failed"),
            ]
            assert "Data loaded successfully" in dialogs.consume_message_containing(
                "warning", analysis_error
            )
        dialogs.assert_no_unexpected_messages()
    finally:
        window.remove_tab_widget("tab-1")


def test_main_window_successful_writeback_relies_on_event_owned_projection(
    main_window_control_factory, qapp
):
    from zcu_tools.gui.app.measure.ui.main_window import MainWindow

    ctrl = apply_window_defaults(mock_ctrl(main_window_control_factory))
    ctrl.get_bus.return_value = EventBus()
    ctrl.get_tab_snapshot.return_value = make_tab_snapshot("tab-1", has_figure=False)
    ctrl.has_tab.return_value = True
    ctrl.apply_writeback_for_pane.return_value = {
        "applied_ids": ["param-1"],
        "written": {"parameter": ["r_f"], "ml_modules": [], "ml_waveforms": []},
    }
    window = MainWindow(ctrl, templates=lambda: {})
    window.add_tab_widget("tab-1", "fake")
    try:
        assert window.has_tab_widget("tab-1") is True
        window.refresh_tab_writeback = MagicMock()
        window.show_status_message = MagicMock()

        window.apply_tab_writeback("tab-1", pane="analysis")

        ctrl.apply_writeback_for_pane.assert_called_once_with("tab-1", "analysis")
        window.refresh_tab_writeback.assert_not_called()
        window.show_status_message.assert_called_once_with("Writeback applied: param-1")
    finally:
        window.remove_tab_widget("tab-1")


@pytest.fixture
def blocked_writeback_window(
    tmp_path: Path, qapp: QApplication
) -> Iterator[tuple[MainWindow, str]]:
    class BlockedWritebackAdapter(SampleAdapter):
        def get_writeback_items(
            self, req: WritebackRequest[SampleRun, SampleAnalysis]
        ) -> list[ParameterWriteback]:
            return [
                ParameterWriteback(
                    "general.ext.blocked.sigma", "Nested calibration", 1.0
                )
            ]

    registry = Registry()
    registry.register("blocked-writeback", BlockedWritebackAdapter)
    ctrl = Controller(
        State(SessionEnv(parameters=None, soc=None, soccfg=None)),
        registry,
        None,
        EventBus(),
        project_root=str(tmp_path),
    )
    ctrl.setup_control.new_entry(EntryRequest("synthetic"))
    target = ctrl.setup_control.new_point(
        PointCreateRequest("working", "", CommandOrigin("gui"))
    )
    ctrl.context_control.set_parameter(target, "general.ext.blocked", 3)
    tab_id = ctrl.new_tab("blocked-writeback")
    window = MainWindow(ctrl, templates=lambda: {})
    ctrl.add_view(window)
    window.add_tab_widget(tab_id, "blocked-writeback")
    window.refresh_context_panel()
    window.show()
    try:
        yield window, tab_id
    finally:
        ctrl.close()
        ctrl.close_tab(tab_id)
        window.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        qapp.processEvents()


def test_apply_failure_preserves_edited_draft_and_selection_without_closing_window(
    blocked_writeback_window: tuple[MainWindow, str], tmp_path: Path
) -> None:
    window, tab_id = blocked_writeback_window
    driver = ComponentDriver(window)
    images = tmp_path / "apply-failure"
    driver.screenshot(images / "01-ready.png")
    driver.click("Run")
    driver.select_tab("Analysis")
    qt_until(
        lambda: driver.observe("Analyze").enabled,
        label="run result ready for analysis",
    )
    driver.click("Analyze")
    qt_until(
        lambda: driver.observe(object_name="writebackApply").enabled,
        label="analysis writeback ready",
    )
    driver.screenshot(images / "02-proposal.png")
    driver.click("Edit")
    driver.enter_text(object_name="writebackValue", value="7.25", commit=False)
    driver.screenshot(images / "03-edited.png")
    driver.click("Save")
    proposed = driver.observe(object_name="writebackProposed")
    selected = driver.observe("general.ext.blocked.sigma*")
    assert proposed.text == "7.25"
    assert selected.checked is True
    driver.screenshot(images / "04-draft.png")

    # Exercise the public Qt boundary directly first: a missing exception
    # barrier fails here without allowing the later real signal to abort pytest.
    window.apply_tab_writeback(tab_id)
    qt_until(lambda: bool(driver.dialogs()), label="direct Apply error visible")
    error = driver.dialogs()[0]
    assert error.title == "Writeback failed"
    assert any("not a field container" in text for text in error.texts)
    driver.screenshot(images / "05-boundary-failure.png")
    driver.click("OK")

    # This second attempt uses the shipped Apply signal and real mouse input.
    driver.click("Apply Selected")
    qt_until(lambda: bool(driver.dialogs()), label="clicked Apply error visible")
    error = driver.dialogs()[0]
    assert error.title == "Writeback failed"
    assert any("not a field container" in text for text in error.texts)
    driver.screenshot(images / "06-click-failure.png")
    driver.click("OK")
    qt_until(
        lambda: (
            tuple(
                widget
                for widget in driver.visible().widgets
                if widget.object_name == "writebackProposed"
            )
            == (proposed,)
        ),
        label="retained proposal row after owner-loop refresh",
    )
    assert driver.observe("general.ext.blocked.sigma*") == selected
    assert driver.observe("Apply Selected").enabled
    assert "Writeback applied" not in driver.status().text
    assert driver.tabs("Analysis").current == "Analysis"
    driver.screenshot(images / "07-retained.png")


def _status_bar_labels(window: MainWindow, text: str) -> list[QLabel]:
    status_bar = window.statusBar()
    assert status_bar is not None
    return [label for label in status_bar.findChildren(QLabel) if label.text() == text]


def test_status_bar_message_is_mouse_selectable(main_window_control_factory, qapp):
    ctrl = apply_window_defaults(mock_ctrl(main_window_control_factory))
    ctrl.get_bus.return_value = EventBus()
    ctrl.list_tab_ids.return_value = []
    window = MainWindow(ctrl, templates=lambda: {})

    assert len(_status_bar_labels(window, "Ready")) == 1

    window.show_status_message("Writeback applied: param-1")

    labels = _status_bar_labels(window, "Writeback applied: param-1")
    assert len(labels) == 1
    flags = labels[0].textInteractionFlags()
    assert flags & Qt.TextInteractionFlag.TextSelectableByMouse
    assert window.get_view_snapshot()["status"] == "Writeback applied: param-1"


def test_long_status_message_stays_within_window_with_full_tooltip(
    main_window_control_factory, qapp
):
    ctrl = apply_window_defaults(mock_ctrl(main_window_control_factory))
    ctrl.get_bus.return_value = EventBus()
    ctrl.list_tab_ids.return_value = []
    window = MainWindow(ctrl, templates=lambda: {})
    message = "Config invalid: " + (
        "Fill in F1 pulse frequency (MHz), normalized gain and duration (us). " * 8
    )
    window.show_status_message(message)
    window.resize(1700, 900)
    window.show()
    qapp.processEvents()

    assert window.width() == 1700
    labels = _status_bar_labels(window, message)
    assert len(labels) == 1
    assert labels[0].width() <= window.width()
    assert labels[0].toolTip() == message
    assert window.get_view_snapshot()["status"] == message


@pytest.fixture
def footer_form_window(tmp_path: Path, qapp: QApplication) -> Iterator[MainWindow]:
    """Show a synthetic form whose input selects a short or long diagnostic."""

    class DiagnosticCfg(ConfigBase):
        message: str = ""

        @field_validator("message")
        @classmethod
        def validate_message(cls, value: str) -> str:
            if value == "wrapped":
                raise ValueError("Fill in frequency, gain and duration. " * 40)
            if value == "multiline":
                raise ValueError("\n".join(["Fill in frequency."] * 6))
            if value:
                raise ValueError("Fill in frequency.")
            return value

    class DiagnosticAdapter(BaseAdapter[DiagnosticCfg, object]):
        capabilities: ClassVar[AdapterCapabilities] = AdapterCapabilities(
            requires_soc=False, analysis=AnalysisMode.NONE, load_data=False
        )

        @staticmethod
        def make_form(ctx: FormContext) -> Form[DiagnosticCfg]:
            del ctx
            return Form(DiagnosticCfg, {})

    registry = Registry()
    registry.register("diagnostic", DiagnosticAdapter)
    ctrl = Controller(
        State(SessionEnv(parameters=None, soc=None, soccfg=None)),
        registry,
        None,
        EventBus(),
        project_root=str(tmp_path),
    )
    window = MainWindow(
        ctrl,
        templates=lambda: {},
        dialog_presenter=RecordingDialogPresenter(confirm_answers=[True]),
    )
    ctrl.add_view(window)
    tab_id = ctrl.new_tab("diagnostic")
    window.resize(1700, 900)
    window.show()
    qapp.processEvents()
    try:
        yield window
    finally:
        ctrl.close()
        ctrl.close_tab(tab_id)
        window.dispose()


@pytest.mark.parametrize("message", ["wrapped", "multiline"])
def test_footer_tracks_current_error_lines_after_edits_and_resize(
    footer_form_window: MainWindow, tmp_path: Path, message: str
) -> None:
    driver = ComponentDriver(footer_form_window)

    def footer_height() -> int:
        bounds = driver.observe(driver.status().text).bounds
        assert bounds is not None
        return bounds[3]

    driver.screenshot(tmp_path / "01-ready.png")
    single_line_height = footer_height()
    driver.enter_text(object_name="form.input.message", value=message)
    driver.screenshot(tmp_path / "02-long.png")
    assert "Config invalid:" in driver.status().text
    assert footer_height() > single_line_height

    driver.enter_text(object_name="form.input.message", value="short")
    driver.screenshot(tmp_path / "03-short.png")
    assert "Fill in frequency." in driver.status().text
    assert footer_height() <= single_line_height

    for width in (1100, 2000, 1700):
        driver.resize_window(width, 900)
        driver.screenshot(tmp_path / f"04-short-{width}.png")
        assert footer_height() <= single_line_height

    driver.enter_text(object_name="form.input.message", value=message)
    driver.resize_window(1100, 900)
    driver.screenshot(tmp_path / "05-long-narrow.png")
    narrow_height = footer_height()
    driver.resize_window(2000, 900)
    driver.screenshot(tmp_path / "06-long-wide.png")
    if message == "wrapped":
        assert single_line_height < footer_height() < narrow_height
    else:
        assert footer_height() == narrow_height

    driver.enter_text(object_name="form.input.message", value="short")
    driver.click("Refresh")
    driver.screenshot(tmp_path / "07-refreshed.png")
    assert footer_height() <= single_line_height
    driver.select_tab("Guide")
    driver.select_tab("Run")
    driver.screenshot(tmp_path / "08-tab-return.png")
    assert footer_height() <= single_line_height
    driver.click("Reset")
    driver.screenshot(tmp_path / "09-reset.png")
    assert "Config invalid:" not in driver.status().text


def test_main_window_toolbar_does_not_show_arb_waveforms(
    main_window_control_factory, qapp
):
    from qtpy.QtWidgets import QPushButton
    from zcu_tools.gui.app.measure.ui.main_window import MainWindow

    ctrl = apply_window_defaults(mock_ctrl(main_window_control_factory))
    ctrl.get_bus.return_value = EventBus()

    window = MainWindow(ctrl, templates=lambda: {})
    texts = {button.text() for button in window.findChildren(QPushButton)}

    assert "Inspect…" in texts
    assert "Agent…" not in texts
    assert "Arb Waveforms…" not in texts


def test_main_window_named_dialog_facade_delegates_to_registry(
    main_window_control_factory, qapp, monkeypatch
):
    from zcu_tools.gui.app.measure.remote.dialogs import DialogName
    from zcu_tools.gui.app.measure.ui.main_dialog_registry import MainDialogRegistry
    from zcu_tools.gui.app.measure.ui.main_window import MainWindow

    ctrl = apply_window_defaults(mock_ctrl(main_window_control_factory))
    ctrl.get_bus.return_value = EventBus()
    ctrl.get_running_tab_id.return_value = None
    registry = MagicMock()
    registry.visible_names.return_value = [DialogName.PREDICTOR]
    registry.take_screenshot.return_value = b"png"
    monkeypatch.setattr(MainDialogRegistry, "close", registry.close)
    monkeypatch.setattr(MainDialogRegistry, "open", registry.open)
    monkeypatch.setattr(MainDialogRegistry, "visible_names", registry.visible_names)
    monkeypatch.setattr(MainDialogRegistry, "take_screenshot", registry.take_screenshot)
    window = MainWindow(ctrl, templates=lambda: {})

    window.close_dialog(DialogName.PREDICTOR)
    window.open_dialog(DialogName.PREDICTOR)
    assert window.list_open_dialogs() == [DialogName.PREDICTOR]

    assert window.take_dialog_screenshot(DialogName.SETUP) == b"png"

    registry.close.assert_called_once_with(DialogName.PREDICTOR)
    registry.open.assert_called_once_with(DialogName.PREDICTOR, writable=True)
    registry.visible_names.assert_called_once_with()
    registry.take_screenshot.assert_called_once_with(DialogName.SETUP)


def test_show_error_dialog_retains_until_close(main_window_control_factory, qapp):
    import gc
    import weakref

    from qtpy.QtWidgets import QMessageBox

    ctrl = apply_window_defaults(mock_ctrl(main_window_control_factory))
    ctrl.get_bus.return_value = EventBus()
    window = MainWindow(ctrl, templates=lambda: {})

    window.show_error_dialog("Broken", "Something failed")
    qapp.processEvents()

    dialogs = [
        dialog
        for dialog in window.findChildren(QMessageBox)
        if dialog.windowTitle() == "Broken"
    ]
    assert len(dialogs) == 1
    dialog_ref = weakref.ref(dialogs[0])
    del dialogs
    gc.collect()

    dialog = dialog_ref()
    assert dialog is not None
    assert dialog.text() == "Something failed"
    assert dialog.icon() == QMessageBox.Icon.Critical
    assert dialog.isVisible()
    assert not dialog.isModal()
    assert dialog.windowModality() == Qt.WindowModality.NonModal
    assert window.list_open_dialogs() == []

    destroyed: list[bool] = []
    dialog.destroyed.connect(lambda: destroyed.append(True))
    dialog.close()
    del dialog
    QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
    qapp.processEvents()

    assert destroyed
    assert window.list_open_dialogs() == []


def test_main_window_load_data_dialog_cancel_is_noop(
    main_window_control_factory, qapp, monkeypatch
):
    from zcu_tools.gui.app.measure.ui.main_window import MainWindow
    from zcu_tools.gui.app.measure.ui.run_load_dialog import RunLoadDialog

    ctrl = apply_window_defaults(mock_ctrl(main_window_control_factory))
    ctrl.get_bus.return_value = EventBus()
    ctrl.has_tab.return_value = True
    ctrl.get_tab_snapshot.return_value = make_tab_snapshot(
        "tab-1", supports_load_data=True, has_figure=False
    )
    ctrl.save_control.list_saved_runs.return_value = ()
    window = MainWindow(ctrl, templates=lambda: {})
    window.add_tab_widget("tab-1", "fake")
    try:
        assert window.has_tab_widget("tab-1") is True

        def cancel(dialog: RunLoadDialog) -> int:
            button = dialog.findChild(QPushButton, "load_cancel")
            assert button is not None
            button.click()
            return dialog.result()

        monkeypatch.setattr(RunLoadDialog, "exec", cancel)
        window.load_tab_data_dialog("tab-1")

        ctrl.load_tab_result.assert_not_called()
    finally:
        window.remove_tab_widget("tab-1")


def test_main_window_load_data_dialog_shows_user_facing_error(
    main_window_control_factory, qapp, monkeypatch
):
    from qtpy.QtWidgets import QFileDialog
    from zcu_tools.gui.app.measure.services.load import LoadDataError
    from zcu_tools.gui.app.measure.ui.main_window import MainWindow
    from zcu_tools.gui.app.measure.ui.run_load_dialog import RunLoadDialog

    ctrl = apply_window_defaults(mock_ctrl(main_window_control_factory))
    ctrl.get_bus.return_value = EventBus()
    ctrl.has_tab.return_value = True
    ctrl.get_tab_snapshot.return_value = make_tab_snapshot(
        "tab-1", supports_load_data=True, has_figure=False
    )
    ctrl.save_control.list_saved_runs.return_value = ()
    ctrl.load_tab_result.side_effect = LoadDataError(
        "Cannot load this data file into the current tab.\n\nDetails: bad axes",
        reason_code="invalid_data_file",
    )
    window = MainWindow(ctrl, templates=lambda: {})
    window.add_tab_widget("tab-1", "fake")
    try:
        assert window.has_tab_widget("tab-1") is True
        window.show_error_dialog = MagicMock()
        monkeypatch.setattr(
            QFileDialog,
            "getOpenFileName",
            lambda *args, **kwargs: ("/tmp/bad.hdf5", ""),
        )

        def choose_file(dialog: RunLoadDialog) -> int:
            button = dialog.findChild(QPushButton, "load_file")
            assert button is not None
            button.click()
            return dialog.result()

        monkeypatch.setattr(RunLoadDialog, "exec", choose_file)
        window.load_tab_data_dialog("tab-1")

        window.show_error_dialog.assert_called_once()
        title, message = window.show_error_dialog.call_args.args
        assert title == "Load data failed"
        assert "Cannot load this data file" in message
        assert "bad axes" in message
    finally:
        window.remove_tab_widget("tab-1")


def test_main_window_run_lock_keeps_new_tab_available(
    main_window_control_factory, qapp, monkeypatch
):
    from zcu_tools.gui.app.measure.ui.exp_tab_widget import ExpTabWidget
    from zcu_tools.gui.app.measure.ui.main_window import MainWindow

    ctrl = mock_ctrl(main_window_control_factory)
    configure_form_lookup(ctrl)
    ctrl.get_bus.return_value = EventBus()
    ctrl.has_tab.return_value = True
    ctrl.get_tab_snapshot.side_effect = lambda tab_id: make_tab_snapshot(
        tab_id,
        global_run_active=tab_id != "tab-1",
        is_running=tab_id == "tab-1",
    )

    window = MainWindow(ctrl, templates=lambda: {})
    try:
        window.add_tab_widget("tab-1", "fake")
        window.add_tab_widget("tab-2", "fake")
        tabs = {tab.tab_id: tab for tab in window.findChildren(ExpTabWidget)}
        tab_one = tabs["tab-1"]
        tab_two = tabs["tab-2"]
        tab_one_update = MagicMock(wraps=tab_one.update_interaction_state)
        tab_two_update = MagicMock(wraps=tab_two.update_interaction_state)
        monkeypatch.setattr(tab_one, "update_interaction_state", tab_one_update)
        monkeypatch.setattr(tab_two, "update_interaction_state", tab_two_update)

        window.refresh_run_lock("tab-1")

        assert _new_tab_button(window).isEnabled() is True
        tab_one_update.assert_called_once()
        tab_two_update.assert_called_once()
    finally:
        window.remove_tab_widget("tab-1")
        window.remove_tab_widget("tab-2")


def test_main_window_tabs_are_movable_and_close_uses_moved_widget(
    main_window_control_factory, qapp
):
    from zcu_tools.gui.app.measure.ui.main_window import MainWindow

    ctrl = apply_window_defaults(editor_wiring_ctrl(main_window_control_factory))
    ctrl.get_bus.return_value = EventBus()
    ctrl.has_tab.side_effect = lambda tab_id: tab_id in {"tab-a", "tab-b"}
    ctrl.get_tab_snapshot.side_effect = lambda tab_id: make_tab_snapshot(
        tab_id, has_run_result=False
    )
    window = MainWindow(ctrl, templates=lambda: {})
    try:
        window.add_tab_widget("tab-a", "A")
        window.add_tab_widget("tab-b", "B")
        tab_a = experiment_tabs(window).widget(0)
        assert tab_a is not None

        assert experiment_tabs(window).isMovable() is True

        tab_bar = experiment_tabs(window).tabBar()
        assert tab_bar is not None
        tab_bar.moveTab(0, 1)
        assert experiment_tabs(window).widget(1) is tab_a
        ctrl.reorder_tabs.assert_called_once_with(["tab-b", "tab-a"])

        experiment_tabs(window).tabCloseRequested.emit(1)

        ctrl.close_tab.assert_called_once_with("tab-a")
    finally:
        window.remove_tab_widget("tab-a")
        window.remove_tab_widget("tab-b")


def test_main_window_soc_changed_refreshes_run_lock(
    main_window_control_factory, qapp, monkeypatch
):
    from zcu_tools.gui.app.measure.ui.exp_tab_widget import ExpTabWidget
    from zcu_tools.gui.app.measure.ui.main_window import MainWindow

    ctrl = apply_window_defaults(mock_ctrl(main_window_control_factory))
    bus = EventBus()
    ctrl.get_bus.return_value = bus
    ctrl.get_running_tab_id.return_value = None
    ctrl.has_tab.return_value = True
    ctrl.get_tab_snapshot.return_value = make_tab_snapshot("tab-1", has_soc=False)

    window = MainWindow(ctrl, templates=lambda: {})
    try:
        window.add_tab_widget("tab-1", "fake")
        tab = next(
            tab for tab in window.findChildren(ExpTabWidget) if tab.tab_id == "tab-1"
        )
        update_interaction_state = MagicMock(wraps=tab.update_interaction_state)
        monkeypatch.setattr(tab, "update_interaction_state", update_interaction_state)

        bus.emit(SocChangedPayload(soc=None, soccfg=None))

        update_interaction_state.assert_called()
    finally:
        window.remove_tab_widget("tab-1")


def test_main_window_content_event_queries_single_tab_snapshot(
    main_window_control_factory, qapp, monkeypatch
):
    from zcu_tools.gui.app.measure.events.tab import (
        TabContentChangedPayload,
        TabContentFact,
    )
    from zcu_tools.gui.app.measure.ui.main_window import MainWindow

    ctrl = apply_window_defaults(mock_ctrl(main_window_control_factory))
    bus = EventBus()
    ctrl.get_bus.return_value = bus
    snapshot = make_tab_snapshot("tab-1")
    assert snapshot.analysis is not None
    ctrl.get_tab_snapshot.return_value = snapshot
    window = MainWindow(ctrl, templates=lambda: {})
    tab = MagicMock()
    try:
        window.add_tab_widget("tab-1", "fake")
        view = next(
            view for view in window.findChildren(ExpTabWidget) if view.tab_id == "tab-1"
        )
        monkeypatch.setattr(view.analyze_form, "sync", tab.analyze_form.sync)
        monkeypatch.setattr(view, "update_writeback_items", tab.update_writeback_items)
        ctrl.get_tab_snapshot.reset_mock()

        bus.emit(TabContentChangedPayload("tab-1", TabContentFact.RUN_RESULT_COMMITTED))

        ctrl.get_tab_snapshot.assert_called_once_with("tab-1")
        tab.analyze_form.sync.assert_called_once_with(snapshot.analysis.params)
        tab.update_writeback_items.assert_called_once_with(
            list(snapshot.analysis.writeback_items)
        )
    finally:
        window.remove_tab_widget("tab-1")


def test_new_tab_menu_supports_nested_paths(
    main_window_control_factory, qapp, monkeypatch
):
    from qtpy.QtWidgets import QMenu
    from zcu_tools.gui.app.measure.ui.main_window import MainWindow

    del qapp
    ctrl = mock_ctrl(main_window_control_factory)
    configure_form_lookup(ctrl)
    ctrl.get_bus.return_value = EventBus()
    ctrl.get_adapter_names.return_value = [
        "fake",
        "twotone/rabi/length",
        "twotone/rabi/amp",
    ]
    window = MainWindow(ctrl, templates=lambda: {})

    def _find_action_by_data(menu: QMenu, target: str):
        for action in menu.actions():
            if action.data() == target:
                return action
            child = action.menu()
            if isinstance(child, QMenu):
                found = _find_action_by_data(child, target)
                if found is not None:
                    return found
        return None

    def _fake_exec(self, *_args, **_kwargs):
        action = _find_action_by_data(self, "twotone/rabi/length")
        assert action is not None
        return action

    monkeypatch.setattr(QMenu, "exec", _fake_exec)

    _new_tab_button(window).click()

    ctrl.new_tab.assert_called_once_with("twotone/rabi/length")
