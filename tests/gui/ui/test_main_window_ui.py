"""UI structure tests for ExpTabWidget layout decisions."""

from __future__ import annotations

from typing import cast
from unittest.mock import MagicMock

import pytest
from qtpy.QtCore import Qt
from zcu_tools.gui.app.main.adapter import AdapterCapabilities, AnalysisMode
from zcu_tools.gui.app.main.services import PersistedStartup, TabSnapshot
from zcu_tools.gui.app.main.state import TabInteractionState
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
from zcu_tools.gui.session.events import SocChangedPayload
from zcu_tools.gui.session.types import ExpContext

from tests.gui._dialog_fakes import RecordingDialogPresenter


def _mock_ctrl() -> MagicMock:
    ctrl = MagicMock()
    ctrl.get_persisted_startup.return_value = PersistedStartup(left_panel_width=500)
    return ctrl


def _apply_window_defaults(ctrl: MagicMock) -> MagicMock:
    """Set the minimal return values required by MainWindow.__init__ on a mock ctrl.

    MainWindow calls active_operation_count() and has_agent_connected() during
    bus-event handlers (FeedbackPanel docking, ADR-0025 C3); tests
    that emit bus events must stub both to deterministic values.
    """
    ctrl.active_operation_count.return_value = 0
    ctrl.has_agent_connected.return_value = False
    ctrl.get_exp_context.return_value = ExpContext(
        md=MagicMock(), ml=MagicMock(), soc=None, soccfg=None
    )
    return ctrl


# Distinguishes "caller did not specify analyze_params" (default to a MagicMock)
# from "caller explicitly wants None" (a non-analysis adapter's snapshot).
_DEFAULT_PARAMS = object()


class _RecordingTabActions:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []

    def refresh_interaction(self, tab_id: str) -> None:
        self.calls.append(("refresh_interaction", tab_id))

    def run_or_stop(self, tab_id: str) -> None:
        self.calls.append(("run_or_stop", tab_id))

    def load_data(self, tab_id: str) -> None:
        self.calls.append(("load_data", tab_id))

    def analyze(self, tab_id: str) -> None:
        self.calls.append(("analyze", tab_id))

    def post_analyze(self, tab_id: str) -> None:
        self.calls.append(("post_analyze", tab_id))

    def apply_writeback(self, tab_id: str) -> None:
        self.calls.append(("apply_writeback", tab_id))

    def save_data(self, tab_id: str) -> None:
        self.calls.append(("save_data", tab_id))

    def save_image(self, tab_id: str) -> None:
        self.calls.append(("save_image", tab_id))

    def save_result(self, tab_id: str) -> None:
        self.calls.append(("save_result", tab_id))

    def save_post_image(self, tab_id: str) -> None:
        self.calls.append(("save_post_image", tab_id))


def _snapshot(
    tab_id: str,
    *,
    global_run_active: bool = False,
    is_running: bool = False,
    is_analyzing: bool = False,
    is_saving_data: bool = False,
    has_context: bool = True,
    has_active_context: bool = True,
    has_soc: bool = True,
    has_run_result: bool = True,
    has_analyze_result: bool = True,
    has_figure: bool = True,
    has_post_analyze_result: bool = False,
    supports_analysis: bool = True,
    supports_post_analysis: bool = False,
    analyze_params: object = _DEFAULT_PARAMS,
    post_analyze_params: object | None = None,
) -> TabSnapshot:
    return TabSnapshot(
        adapter_name="fake",
        tab_id=tab_id,
        interaction=TabInteractionState(
            global_run_active=global_run_active,
            is_running=is_running,
            is_analyzing=is_analyzing,
            is_saving_data=is_saving_data,
            has_context=has_context,
            has_active_context=has_active_context,
            has_soc=has_soc,
            has_run_result=has_run_result,
            has_analyze_result=has_analyze_result,
            has_figure=has_figure,
            has_post_analyze_result=has_post_analyze_result,
        ),
        cfg_schema=MagicMock(),
        save_paths_override=None,
        capabilities=AdapterCapabilities(
            analysis=AnalysisMode.FIT if supports_analysis else AnalysisMode.NONE,
            post_analysis=supports_post_analysis,
        ),
        analyze_params=MagicMock()
        if analyze_params is _DEFAULT_PARAMS
        else analyze_params,
        post_analyze_params=post_analyze_params,
        writeback_items=(),
        save_paths=None,
        figure=None,
    )


def test_left_panel_toggle_is_attached_to_tab_bar(qapp):
    from qtpy.QtWidgets import QApplication
    from zcu_tools.gui.app.main.ui.main_window import ExpTabWidget

    tab = ExpTabWidget("tab-1", _mock_ctrl())
    tab.show()
    QApplication.processEvents()

    corner = tab._left_tabs.cornerWidget(Qt.TopLeftCorner)  # type: ignore[attr-defined]
    assert corner is None
    assert tab._left_edge_handle.isVisible() is True


def test_left_panel_toggle_uses_collapsed_boundary_handle(qapp):
    from qtpy.QtWidgets import QApplication
    from zcu_tools.gui.app.main.ui.main_window import ExpTabWidget

    tab = ExpTabWidget("tab-1", _mock_ctrl())
    tab.resize(1000, 700)
    tab.show()
    QApplication.processEvents()

    expanded_left = tab._splitter.sizes()[0]
    expanded_handle_x = tab._left_edge_handle.x()
    assert expanded_handle_x > 0
    assert tab._left_panel_collapsed is False

    tab._left_edge_handle.click()
    QApplication.processEvents()

    assert tab._splitter.sizes()[0] == 0
    assert tab._left_panel_collapsed is True
    assert tab._left_edge_handle.x() == 0

    tab._left_edge_handle.click()
    QApplication.processEvents()

    assert tab._left_panel_collapsed is False
    assert tab._left_edge_handle.x() > 0
    assert tab._splitter.sizes()[0] >= expanded_left


def test_left_panel_handle_tracks_splitter_boundary(qapp):
    from qtpy.QtWidgets import QApplication
    from zcu_tools.gui.app.main.ui.main_window import ExpTabWidget

    tab = ExpTabWidget("tab-1", _mock_ctrl())
    tab.resize(1000, 700)
    tab.show()
    QApplication.processEvents()

    initial_x = tab._left_edge_handle.x()
    splitter_x = tab._splitter.geometry().x()
    left_boundary = splitter_x + tab._left_tabs.geometry().right() + 1
    assert abs(initial_x - (left_boundary - tab._left_edge_handle.width() // 2)) <= 2

    tab._splitter.setSizes([260, 740])
    tab._schedule_handle_layout()
    QApplication.processEvents()

    moved_x = tab._left_edge_handle.x()
    splitter_x = tab._splitter.geometry().x()
    left_boundary = splitter_x + tab._left_tabs.geometry().right() + 1
    assert abs(moved_x - (left_boundary - tab._left_edge_handle.width() // 2)) <= 2


def test_exp_tab_disables_local_buttons_while_analyzing(qapp):
    from zcu_tools.gui.app.main.ui.main_window import ExpTabWidget

    tab = ExpTabWidget("tab-1", _mock_ctrl())
    tab.update_writeback_items([MagicMock(selected=True)])
    tab.update_interaction_state(
        _snapshot(
            "tab-1",
            global_run_active=False,
            is_running=False,
            is_analyzing=True,
            is_saving_data=False,
            has_context=True,
            has_active_context=True,
            has_soc=True,
            has_run_result=True,
            has_analyze_result=True,
            has_figure=True,
        )
    )

    assert tab.analyze_btn.isEnabled() is False
    assert tab.load_data_btn.isEnabled() is False
    assert tab.writeback_widget.isEnabled() is False
    assert tab.save_image_btn.isEnabled() is False  # disabled because is_analyzing


def test_exp_tab_keeps_analyze_enabled_while_other_tab_running(qapp):
    from zcu_tools.gui.app.main.ui.main_window import ExpTabWidget

    tab = ExpTabWidget("tab-1", _mock_ctrl())
    tab.update_interaction_state(
        _snapshot(
            "tab-1",
            global_run_active=True,
            is_running=False,
            is_analyzing=False,
            is_saving_data=False,
            has_context=True,
            has_active_context=True,
            has_soc=True,
            has_run_result=True,
            has_analyze_result=True,
            has_figure=True,
        )
    )

    assert tab.run_btn.isEnabled() is False
    assert tab.analyze_btn.isEnabled() is True
    assert tab.save_data_btn.isEnabled() is True


def test_exp_tab_disables_save_buttons_while_saving_data(qapp):
    from zcu_tools.gui.app.main.ui.main_window import ExpTabWidget

    tab = ExpTabWidget("tab-1", _mock_ctrl())
    tab.update_interaction_state(
        _snapshot(
            "tab-1",
            global_run_active=False,
            is_running=False,
            is_analyzing=False,
            is_saving_data=True,
            has_context=True,
            has_active_context=True,
            has_soc=True,
            has_run_result=True,
            has_analyze_result=True,
            has_figure=True,
        )
    )

    assert tab.save_data_btn.isEnabled() is False
    assert tab.save_result_btn.isEnabled() is False
    assert tab.run_btn.text() == "Run"
    assert tab.run_btn.toolTip() == "Tab is busy"


def test_exp_tab_run_tooltip_shows_no_soc_reason(qapp):
    from zcu_tools.gui.app.main.ui.main_window import ExpTabWidget

    tab = ExpTabWidget("tab-1", _mock_ctrl())
    tab.update_interaction_state(
        _snapshot(
            "tab-1",
            global_run_active=False,
            is_running=False,
            is_analyzing=False,
            is_saving_data=False,
            has_context=True,
            has_active_context=True,
            has_soc=False,
            has_run_result=False,
            has_analyze_result=False,
            has_figure=False,
        )
    )

    assert tab.run_btn.isEnabled() is False
    assert tab.run_btn.toolTip() == "No SoC connection"


def test_exp_tab_run_tooltip_shows_cfg_invalid_reason(qapp):
    from zcu_tools.gui.app.main.ui.main_window import ExpTabWidget

    tab = ExpTabWidget("tab-1", _mock_ctrl())
    tab.cfg_form.first_invalid_reason = MagicMock(
        return_value="modules.readout: invalid"
    )
    tab.cfg_form.is_valid = MagicMock(return_value=False)
    tab.update_interaction_state(
        _snapshot(
            "tab-1",
            global_run_active=False,
            is_running=False,
            is_analyzing=False,
            is_saving_data=False,
            has_context=True,
            has_active_context=True,
            has_soc=True,
            has_run_result=False,
            has_analyze_result=False,
            has_figure=False,
        )
    )

    assert tab.run_btn.isEnabled() is False
    assert tab.run_btn.toolTip() == "Config invalid: modules.readout: invalid"


def test_exp_tab_draft_context_allows_analysis_but_disables_run_and_save(qapp):
    from zcu_tools.gui.app.main.ui.main_window import ExpTabWidget

    tab = ExpTabWidget("tab-1", _mock_ctrl())
    tab.update_writeback_items([MagicMock(selected=True)])
    tab.update_interaction_state(
        _snapshot(
            "tab-1",
            global_run_active=False,
            is_running=False,
            is_analyzing=False,
            is_saving_data=False,
            has_context=True,
            has_active_context=False,
            has_soc=True,
            has_run_result=True,
            has_analyze_result=True,
            has_figure=True,
        )
    )

    assert tab.run_btn.isEnabled() is False
    assert tab.run_btn.toolTip() == "Select or create a file-backed context"
    assert tab.load_data_btn.isEnabled() is True
    assert tab.analyze_btn.isEnabled() is True
    assert tab.writeback_widget.isEnabled() is True
    assert tab.save_data_btn.isEnabled() is False
    assert tab.save_image_btn.isEnabled() is False
    assert tab.save_result_btn.isEnabled() is False


def test_non_analysis_adapter_hides_analysis_widgets_but_keeps_save(qapp):
    """flux_dep / power_dep adapters (analysis=NONE) hide only the
    analysis widgets, never the Save section. Regression: the whole second tab
    used to be hidden, so the user could not save a 2D-sweep run at all."""
    from zcu_tools.gui.app.main.ui.main_window import ExpTabWidget

    tab = ExpTabWidget("tab-1", _mock_ctrl())
    tab.update_interaction_state(
        _snapshot(
            "tab-1",
            has_run_result=True,
            has_active_context=True,
            supports_analysis=False,
            analyze_params=None,
        )
    )

    # The second tab stays present and is labelled for what it now holds.
    # (isHidden reflects the widget's own setVisible state independent of whether
    # an ancestor is shown — the tab widget is never .show()n in this test.)
    assert tab._left_tabs.isTabVisible(1) is True
    assert tab._left_tabs.tabText(1) == "Save"
    # Analysis widgets are hidden ...
    assert tab._analyze_section.isHidden() is True
    assert tab.analyze_btn.isHidden() is True
    # Load is an Analysis-tab action, not a Save Browse button.
    assert tab.load_data_btn.text() == "Load Data..."
    assert tab.load_data_btn.isHidden() is True
    assert tab.load_data_btn.isEnabled() is False
    # ... but Save stays reachable and usable (run result + active context).
    assert tab.save_data_btn.isHidden() is False
    assert tab.save_data_btn.isEnabled() is True


def test_analysis_adapter_shows_analysis_widgets_and_labels_tab(qapp):
    """An analysis adapter keeps the analysis widgets visible and the second tab
    labelled 'Analysis' — the counterpart to the non-analysis case."""
    from zcu_tools.gui.app.main.ui.main_window import ExpTabWidget

    tab = ExpTabWidget("tab-1", _mock_ctrl())
    tab.update_interaction_state(
        _snapshot("tab-1", has_run_result=True, supports_analysis=True)
    )

    assert tab._left_tabs.tabText(1) == "Analysis"
    assert tab._analyze_section.isHidden() is False
    assert tab.analyze_btn.isHidden() is False
    assert tab.save_data_btn.isHidden() is False


def test_exp_tab_load_button_requires_context_but_not_soc(qapp):
    from zcu_tools.gui.app.main.ui.main_window import ExpTabWidget

    tab = ExpTabWidget("tab-1", _mock_ctrl())
    tab.update_interaction_state(
        _snapshot(
            "tab-1",
            has_context=True,
            has_active_context=False,
            has_soc=False,
            has_run_result=False,
            has_analyze_result=False,
            has_figure=False,
        )
    )
    assert tab.load_data_btn.isEnabled() is True

    tab.update_interaction_state(
        _snapshot(
            "tab-1",
            has_context=False,
            has_active_context=False,
            has_soc=False,
            has_run_result=False,
            has_analyze_result=False,
            has_figure=False,
        )
    )
    assert tab.load_data_btn.isEnabled() is False


def test_main_window_load_data_dialog_calls_controller(qapp, monkeypatch, tmp_path):
    from qtpy.QtWidgets import QFileDialog
    from zcu_tools.gui.app.main.ui.main_window import MainWindow

    ctrl = _apply_window_defaults(MagicMock())
    ctrl.get_bus.return_value = EventBus()
    ctrl.has_tab.return_value = True
    database_root = tmp_path / "Database" / "Q3_2D" / "Q1"
    database_root.mkdir(parents=True)
    ctrl.get_exp_context.return_value = ExpContext(
        md=MagicMock(),
        ml=MagicMock(),
        soc=None,
        soccfg=None,
        database_path=str(database_root / "2026" / "06" / "Data_0625"),
    )
    window = MainWindow(ctrl)
    window._tab_widgets["tab-1"] = MagicMock()
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

    window._on_load_data_clicked("tab-1")

    ctrl.load_tab_result.assert_called_once_with("tab-1", "/tmp/result.hdf5")
    window.show_status_message.assert_called_once()
    assert captured_dir["directory"] == str(database_root)


def test_main_window_toolbar_does_not_show_arb_waveforms(qapp):
    from qtpy.QtWidgets import QPushButton
    from zcu_tools.gui.app.main.ui.main_window import MainWindow

    ctrl = _apply_window_defaults(MagicMock())
    ctrl.get_bus.return_value = EventBus()

    window = MainWindow(ctrl)
    texts = {button.text() for button in window.findChildren(QPushButton)}

    assert "Inspect…" in texts
    assert "Agent…" not in texts
    assert "Arb Waveforms…" not in texts


def test_main_window_tab_actions_forward_to_private_handlers(qapp, monkeypatch):
    from zcu_tools.gui.app.main.ui.main_window import MainWindow

    ctrl = _apply_window_defaults(MagicMock())
    ctrl.get_bus.return_value = EventBus()
    window = MainWindow(ctrl)
    handlers = {
        "refresh_tab_interaction": MagicMock(),
        "_on_run_stop_clicked": MagicMock(),
        "_on_load_data_clicked": MagicMock(),
        "_on_analyze_clicked": MagicMock(),
        "_on_post_analyze_clicked": MagicMock(),
        "_on_writeback_inline_apply": MagicMock(),
        "_on_save_data_clicked": MagicMock(),
        "_on_save_image_clicked": MagicMock(),
        "_on_save_result_clicked": MagicMock(),
        "_on_post_save_image_clicked": MagicMock(),
    }
    for name, handler in handlers.items():
        monkeypatch.setattr(window, name, handler)

    action_to_handler = [
        ("refresh_interaction", "refresh_tab_interaction"),
        ("run_or_stop", "_on_run_stop_clicked"),
        ("load_data", "_on_load_data_clicked"),
        ("analyze", "_on_analyze_clicked"),
        ("post_analyze", "_on_post_analyze_clicked"),
        ("apply_writeback", "_on_writeback_inline_apply"),
        ("save_data", "_on_save_data_clicked"),
        ("save_image", "_on_save_image_clicked"),
        ("save_result", "_on_save_result_clicked"),
        ("save_post_image", "_on_post_save_image_clicked"),
    ]
    for action_name, handler_name in action_to_handler:
        getattr(window._tab_actions, action_name)("tab-1")
        handlers[handler_name].assert_called_once_with("tab-1")


def test_main_window_named_dialog_facade_delegates_to_registry(qapp):
    from qtpy.QtWidgets import QDialog
    from zcu_tools.gui.app.main.services.remote.dialogs import DialogName
    from zcu_tools.gui.app.main.ui.main_window import MainWindow

    ctrl = _apply_window_defaults(MagicMock())
    ctrl.get_bus.return_value = EventBus()
    window = MainWindow(ctrl)
    registry = MagicMock()
    registry.visible_names.return_value = [DialogName.PREDICTOR]
    registry.take_screenshot.return_value = b"png"
    window._dialog_registry = registry

    window.close_dialog(DialogName.PREDICTOR)
    window.open_dialog(DialogName.PREDICTOR)
    assert window.list_open_dialogs() == [DialogName.PREDICTOR]

    dialog = QDialog(window)
    window.register_dialog(DialogName.STARTUP, dialog)
    assert window.take_dialog_screenshot(DialogName.STARTUP) == b"png"

    registry.close.assert_called_once_with(DialogName.PREDICTOR)
    registry.open.assert_called_once_with(DialogName.PREDICTOR)
    registry.visible_names.assert_called_once_with()
    registry.register.assert_called_once_with(DialogName.STARTUP, dialog)
    registry.take_screenshot.assert_called_once_with(DialogName.STARTUP)


def test_show_error_dialog_retains_until_close(qapp):
    from qtpy.QtWidgets import QMessageBox
    from zcu_tools.gui.app.main.ui.main_window import MainWindow

    ctrl = _apply_window_defaults(MagicMock())
    ctrl.get_bus.return_value = EventBus()
    window = MainWindow(ctrl)

    window.show_error_dialog("Broken", "Something failed")
    qapp.processEvents()

    dialogs = [
        dialog
        for dialog in window.findChildren(QMessageBox)
        if dialog.windowTitle() == "Broken"
    ]
    assert len(dialogs) == 1
    assert len(window._dialog_refs) == 1
    assert window.list_open_dialogs() == []

    dialogs[0].reject()
    qapp.processEvents()

    assert len(window._dialog_refs) == 0


def test_open_notify_prompt_retains_until_close(qapp):
    from zcu_tools.gui.app.main.ui.main_window import MainWindow
    from zcu_tools.gui.app.main.ui.notify_dialog import NotifyUserDialog

    ctrl = _apply_window_defaults(MagicMock())
    ctrl.get_bus.return_value = EventBus()
    window = MainWindow(ctrl)

    window.open_notify_prompt(17, "Need input", timeout=60.0)
    qapp.processEvents()

    dialogs = window.findChildren(NotifyUserDialog)
    assert len(dialogs) == 1
    assert dialogs[0]._token == 17
    assert len(window._dialog_refs) == 1
    assert window.list_open_dialogs() == []

    dialogs[0].reject()
    qapp.processEvents()

    assert len(window._dialog_refs) == 0


def test_main_window_load_data_dialog_cancel_is_noop(qapp, monkeypatch):
    from qtpy.QtWidgets import QFileDialog
    from zcu_tools.gui.app.main.ui.main_window import MainWindow

    ctrl = _apply_window_defaults(MagicMock())
    ctrl.get_bus.return_value = EventBus()
    ctrl.has_tab.return_value = True
    window = MainWindow(ctrl)
    window._tab_widgets["tab-1"] = MagicMock()
    monkeypatch.setattr(
        QFileDialog, "getOpenFileName", lambda *args, **kwargs: ("", "")
    )

    window._on_load_data_clicked("tab-1")

    ctrl.load_tab_result.assert_not_called()


def test_main_window_load_data_dialog_shows_user_facing_error(qapp, monkeypatch):
    from qtpy.QtWidgets import QFileDialog
    from zcu_tools.gui.app.main.services.load import LoadDataError
    from zcu_tools.gui.app.main.ui.main_window import MainWindow

    ctrl = _apply_window_defaults(MagicMock())
    ctrl.get_bus.return_value = EventBus()
    ctrl.has_tab.return_value = True
    ctrl.load_tab_result.side_effect = LoadDataError(
        "Cannot load this data file into the current tab.\n\nDetails: bad axes",
        reason_code="invalid_data_file",
    )
    window = MainWindow(ctrl)
    window._tab_widgets["tab-1"] = MagicMock()
    window.show_error_dialog = MagicMock()
    monkeypatch.setattr(
        QFileDialog,
        "getOpenFileName",
        lambda *args, **kwargs: ("/tmp/bad.hdf5", ""),
    )

    window._on_load_data_clicked("tab-1")

    window.show_error_dialog.assert_called_once()
    title, message = window.show_error_dialog.call_args.args
    assert title == "Load data failed"
    assert "Cannot load this data file" in message
    assert "bad axes" in message


def test_main_window_run_lock_keeps_new_tab_available(qapp):
    from zcu_tools.gui.app.main.ui.main_window import MainWindow

    ctrl = MagicMock()
    ctrl.get_bus.return_value = EventBus()
    ctrl.has_tab.return_value = True
    ctrl.get_tab_snapshot.side_effect = lambda tab_id: _snapshot(
        tab_id,
        global_run_active=tab_id != "tab-1",
        is_running=tab_id == "tab-1",
    )

    window = MainWindow(ctrl)
    tab_one = MagicMock()
    tab_two = MagicMock()
    window._tab_widgets["tab-1"] = tab_one
    window._tab_widgets["tab-2"] = tab_two

    window.refresh_run_lock("tab-1")

    assert window._toolbar.new_tab_button.isEnabled() is True
    tab_one.update_interaction_state.assert_called_once()
    tab_two.update_interaction_state.assert_called_once()


def test_main_window_tabs_are_movable_and_close_uses_moved_widget(qapp):
    from zcu_tools.gui.app.main.ui.exp_tab_widget import ExpTabWidget
    from zcu_tools.gui.app.main.ui.main_window import MainWindow

    ctrl = _apply_window_defaults(_mock_ctrl())
    ctrl.get_bus.return_value = EventBus()
    ctrl.has_tab.side_effect = lambda tab_id: tab_id in {"tab-a", "tab-b"}
    window = MainWindow(ctrl)
    tab_a = ExpTabWidget("tab-a", ctrl)
    tab_b = ExpTabWidget("tab-b", ctrl)
    window._tab_widgets["tab-a"] = tab_a
    window._tab_widgets["tab-b"] = tab_b
    window._tabs.addTab(tab_a, "A")
    window._tabs.addTab(tab_b, "B")

    assert window._tabs.isMovable() is True

    tab_bar = window._tabs.tabBar()
    assert tab_bar is not None
    tab_bar.moveTab(0, 1)
    assert window._tabs.widget(1) is tab_a
    ctrl.reorder_tabs.assert_called_once_with(["tab-b", "tab-a"])

    window._on_tab_close_requested(1)

    ctrl.close_tab.assert_called_once_with("tab-a")


def test_main_window_soc_changed_refreshes_run_lock(qapp):
    from zcu_tools.gui.app.main.ui.main_window import MainWindow

    ctrl = _apply_window_defaults(MagicMock())
    bus = EventBus()
    ctrl.get_bus.return_value = bus
    ctrl.get_running_tab_id.return_value = None
    ctrl.has_tab.return_value = True
    ctrl.get_tab_snapshot.return_value = _snapshot("tab-1", has_soc=False)

    window = MainWindow(ctrl)
    tab = MagicMock()
    window._tab_widgets["tab-1"] = tab

    bus.emit(SocChangedPayload(soc=None, soccfg=None))

    tab.update_interaction_state.assert_called()


def test_main_window_content_event_queries_single_tab_snapshot(qapp):
    from zcu_tools.gui.app.main.events.tab import TabContentChangedPayload
    from zcu_tools.gui.app.main.ui.main_window import MainWindow

    ctrl = _apply_window_defaults(MagicMock())
    bus = EventBus()
    ctrl.get_bus.return_value = bus
    ctrl.get_tab_snapshot.return_value = _snapshot("tab-1")
    window = MainWindow(ctrl)
    window._tab_widgets["tab-1"] = MagicMock()

    bus.emit(TabContentChangedPayload(tab_id="tab-1"))

    ctrl.get_tab_snapshot.assert_called_once_with("tab-1")


def test_main_window_interaction_event_refreshes_finished_analysis_figure(qapp):
    from dataclasses import replace

    from matplotlib.figure import Figure
    from zcu_tools.gui.app.main.events.tab import TabInteractionChangedPayload
    from zcu_tools.gui.app.main.ui.main_window import MainWindow

    ctrl = _apply_window_defaults(MagicMock())
    bus = EventBus()
    ctrl.get_bus.return_value = bus
    ctrl.has_tab.return_value = True
    figure = Figure()
    writeback_item = MagicMock()
    ctrl.get_tab_snapshot.return_value = replace(
        _snapshot(
            "tab-1",
            is_analyzing=False,
            has_analyze_result=True,
            has_figure=True,
        ),
        figure=figure,
        writeback_items=(writeback_item,),
    )
    window = MainWindow(ctrl)
    tab = MagicMock()
    window._tab_widgets["tab-1"] = tab

    bus.emit(TabInteractionChangedPayload(tab_id="tab-1"))

    ctrl.get_tab_snapshot.assert_called_once_with("tab-1")
    tab.update_writeback_items.assert_called_once_with([writeback_item])
    tab.show_analysis_figure.assert_called_once_with(figure)


def test_main_window_interaction_event_does_not_restore_old_figure_on_analyze_start(
    qapp,
):
    from dataclasses import replace

    from matplotlib.figure import Figure
    from zcu_tools.gui.app.main.events.tab import TabInteractionChangedPayload
    from zcu_tools.gui.app.main.ui.main_window import MainWindow

    ctrl = _apply_window_defaults(MagicMock())
    bus = EventBus()
    ctrl.get_bus.return_value = bus
    ctrl.has_tab.return_value = True
    figure = Figure()
    ctrl.get_tab_snapshot.return_value = replace(
        _snapshot(
            "tab-1",
            is_analyzing=True,
            has_analyze_result=True,
            has_figure=True,
        ),
        figure=figure,
    )
    window = MainWindow(ctrl)
    tab = MagicMock()
    window._tab_widgets["tab-1"] = tab

    bus.emit(TabInteractionChangedPayload(tab_id="tab-1"))

    ctrl.get_tab_snapshot.assert_called_once_with("tab-1")
    tab.show_analysis_figure.assert_not_called()


def test_main_window_interaction_event_does_not_restore_old_figure_on_run_start(
    qapp,
):
    from dataclasses import replace

    from matplotlib.figure import Figure
    from zcu_tools.gui.app.main.events.tab import TabInteractionChangedPayload
    from zcu_tools.gui.app.main.ui.main_window import MainWindow

    ctrl = _apply_window_defaults(MagicMock())
    bus = EventBus()
    ctrl.get_bus.return_value = bus
    ctrl.has_tab.return_value = True
    figure = Figure()
    ctrl.get_tab_snapshot.return_value = replace(
        _snapshot(
            "tab-1",
            is_running=True,
            is_analyzing=False,
            has_analyze_result=True,
            has_figure=True,
        ),
        figure=figure,
    )
    window = MainWindow(ctrl)
    tab = MagicMock()
    window._tab_widgets["tab-1"] = tab

    bus.emit(TabInteractionChangedPayload(tab_id="tab-1"))

    ctrl.get_tab_snapshot.assert_called_once_with("tab-1")
    tab.show_analysis_figure.assert_not_called()


def test_main_window_interaction_event_shows_post_figure_after_primary(qapp):
    from dataclasses import replace

    from matplotlib.figure import Figure
    from zcu_tools.gui.app.main.events.tab import TabInteractionChangedPayload
    from zcu_tools.gui.app.main.ui.main_window import MainWindow

    ctrl = _apply_window_defaults(MagicMock())
    bus = EventBus()
    ctrl.get_bus.return_value = bus
    ctrl.has_tab.return_value = True
    primary = Figure()
    post = Figure()
    ctrl.get_tab_snapshot.return_value = replace(
        _snapshot(
            "tab-1",
            is_analyzing=False,
            has_analyze_result=True,
            has_figure=True,
            has_post_analyze_result=True,
        ),
        figure=primary,
        post_figure=post,
    )
    window = MainWindow(ctrl)
    tab = MagicMock()
    window._tab_widgets["tab-1"] = tab

    bus.emit(TabInteractionChangedPayload(tab_id="tab-1"))

    assert tab.show_analysis_figure.call_args_list == [
        ((primary,),),
        ((post,),),
    ]


def _emit_run_finished(bus, tab_id: str, outcome: str) -> None:
    from zcu_tools.gui.app.main.events.run import RunFinishedPayload

    bus.emit(RunFinishedPayload(tab_id=tab_id, outcome=outcome))


def test_finished_run_auto_switches_to_analysis_tab(qapp):
    """RUN_FINISHED with outcome=finished switches the tab to Analysis — the
    decision reads the outcome straight off the RUN_FINISHED payload."""
    from zcu_tools.gui.app.main.ui.main_window import MainWindow

    ctrl = _apply_window_defaults(MagicMock())
    bus = EventBus()
    ctrl.get_bus.return_value = bus
    ctrl.get_running_tab_id.return_value = None
    ctrl.get_tab_snapshot.return_value = _snapshot("tab-1", has_run_result=True)
    window = MainWindow(ctrl)
    tab = MagicMock()
    window._tab_widgets["tab-1"] = tab

    _emit_run_finished(bus, "tab-1", outcome="finished")

    tab._left_tabs.setCurrentIndex.assert_called_once_with(1)


def test_stopped_run_does_not_auto_switch_to_analysis_tab(qapp):
    """A stopped (cancelled) run may leave a partial result, but the user
    interrupted on purpose — must not yank them to the Analysis tab."""
    from zcu_tools.gui.app.main.ui.main_window import MainWindow

    ctrl = _apply_window_defaults(MagicMock())
    bus = EventBus()
    ctrl.get_bus.return_value = bus
    ctrl.get_running_tab_id.return_value = None
    ctrl.get_tab_snapshot.return_value = _snapshot("tab-1", has_run_result=True)
    window = MainWindow(ctrl)
    tab = MagicMock()
    window._tab_widgets["tab-1"] = tab

    _emit_run_finished(bus, "tab-1", outcome="cancelled")

    tab._left_tabs.setCurrentIndex.assert_not_called()


def test_non_analysis_adapter_run_auto_switches_to_second_tab(qapp):
    """flux_dep / power_dep adapters (analysis=NONE) keep the second
    tab — its analysis widgets are hidden but the Save section stays — so a
    finished run still switches there, landing the user on Save (where they save
    the 2D sweep). Regression: switching used to be skipped, and earlier the
    whole tab was hidden so the user could not save at all."""
    from zcu_tools.gui.app.main.ui.main_window import MainWindow

    ctrl = _apply_window_defaults(MagicMock())
    bus = EventBus()
    ctrl.get_bus.return_value = bus
    ctrl.get_running_tab_id.return_value = None
    ctrl.get_tab_snapshot.return_value = _snapshot(
        "tab-1", has_run_result=True, supports_analysis=False, analyze_params=None
    )
    window = MainWindow(ctrl)
    tab = MagicMock()
    window._tab_widgets["tab-1"] = tab

    _emit_run_finished(bus, "tab-1", outcome="finished")

    tab._left_tabs.setCurrentIndex.assert_called_once_with(1)


def test_refresh_analyze_form_skips_non_analysis_adapter_without_raising(qapp):
    """A finished run on a non-analysis adapter has no analyze params; the
    content refresh must skip the analyze form rather than hit the Fast-Fail
    guard that demands initialized params (regression: it used to raise)."""
    from zcu_tools.gui.app.main.ui.main_window import MainWindow

    ctrl = _apply_window_defaults(MagicMock())
    bus = EventBus()
    ctrl.get_bus.return_value = bus
    ctrl.get_running_tab_id.return_value = None
    ctrl.get_tab_snapshot.return_value = _snapshot(
        "tab-1", has_run_result=True, supports_analysis=False, analyze_params=None
    )
    window = MainWindow(ctrl)
    tab = MagicMock()
    window._tab_widgets["tab-1"] = tab

    # Must not raise "Run result has no initialized analyze parameters".
    window.refresh_tab_analyze_form("tab-1")

    tab.populate_analyze_params.assert_not_called()


def _editor_wiring_ctrl() -> MagicMock:
    """Mock ctrl that supplies measure cfg binding ports for a real attach()."""
    ctrl = MagicMock()
    ctrl.get_persisted_startup.return_value = PersistedStartup(left_panel_width=500)
    ctrl.get_bus.return_value = EventBus()
    ctrl.get_current_md.return_value = MagicMock()
    ctrl.get_current_ml.return_value = MagicMock()
    ctrl.list_device_names.return_value = []
    ctrl.list_arb_waveforms.return_value = []
    ctrl.has_soc.return_value = False
    # MainWindow reads both during bus-event handlers (ADR-0025 C3 gate).
    ctrl.active_operation_count.return_value = 0
    ctrl.has_agent_connected.return_value = False

    # populate_cfg now opens a service-owned (gc=False) seeded session and
    # attaches the widget to the service-owned model (ADR-0008). Build a real
    # CfgDraft for get_cfg_editor_draft so attach() works.
    from zcu_tools.gui.app.main.adapter import CfgSchema, make_default_value
    from zcu_tools.gui.app.main.cfg_binding import MeasureCfgBindings
    from zcu_tools.gui.app.main.cfg_schemas import _MODULE_SPEC_FACTORIES

    spec = _MODULE_SPEC_FACTORIES["pulse"]()
    draft = MeasureCfgBindings(ctrl).new_draft(
        CfgSchema(spec, make_default_value(spec))
    )
    ctrl.open_seeded_cfg_editor.return_value = ("editor-tab1", [])
    ctrl.get_cfg_editor_draft.return_value = draft
    return ctrl


def _pulse_schema():
    from zcu_tools.gui.app.main.adapter import CfgSchema, make_default_value
    from zcu_tools.gui.app.main.cfg_schemas import _MODULE_SPEC_FACTORIES

    spec = _MODULE_SPEC_FACTORIES["pulse"]()
    return CfgSchema(spec=spec, value=make_default_value(spec))


def test_exp_tab_opens_cfg_editor_on_attach(qapp):
    import dataclasses

    from zcu_tools.gui.app.main.ui.main_window import ExpTabWidget

    ctrl = _editor_wiring_ctrl()
    tab = ExpTabWidget("tab-1", ctrl)
    snapshot = dataclasses.replace(_snapshot("tab-1"), cfg_schema=_pulse_schema())
    tab.attach(snapshot, _RecordingTabActions())

    # Opened a gc=False seeded session keyed by the tab id, and attached the
    # widget to the service-owned model.
    ctrl.open_seeded_cfg_editor.assert_called_once()
    kwargs = ctrl.open_seeded_cfg_editor.call_args.kwargs
    assert kwargs["owner_key"] == "tab-1"
    assert kwargs["gc"] is False
    assert tab._cfg_editor_id == "editor-tab1"
    assert tab.cfg_form._draft is ctrl.get_cfg_editor_draft.return_value


def test_exp_tab_tears_down_cfg_editor_on_detach(qapp):
    import dataclasses

    from zcu_tools.gui.app.main.ui.main_window import ExpTabWidget

    ctrl = _editor_wiring_ctrl()
    tab = ExpTabWidget("tab-1", ctrl)
    snapshot = dataclasses.replace(_snapshot("tab-1"), cfg_schema=_pulse_schema())
    tab.attach(snapshot, _RecordingTabActions())
    tab.detach()

    ctrl.teardown_cfg_editor.assert_called_once_with("editor-tab1")
    assert tab._cfg_editor_id is None


def test_exp_tab_buttons_dispatch_public_tab_actions(qapp):
    import dataclasses

    from zcu_tools.gui.app.main.ui.main_window import ExpTabWidget

    ctrl = _editor_wiring_ctrl()
    tab = ExpTabWidget("tab-1", ctrl)
    actions = _RecordingTabActions()
    snapshot = dataclasses.replace(
        _snapshot(
            "tab-1",
            has_run_result=True,
            has_analyze_result=True,
            has_figure=True,
            has_post_analyze_result=True,
            supports_post_analysis=True,
        ),
        cfg_schema=_pulse_schema(),
    )
    tab.attach(snapshot, actions)

    assert tab.run_btn.isEnabled() is True
    assert tab.load_data_btn.isEnabled() is True
    assert tab.analyze_btn.isEnabled() is True
    assert tab.post_analyze_btn.isEnabled() is True
    assert tab.save_data_btn.isEnabled() is True
    assert tab.save_image_btn.isEnabled() is True
    assert tab.save_result_btn.isEnabled() is True
    assert tab.post_save_image_btn.isEnabled() is True

    actions.calls.clear()
    tab.run_btn.click()
    tab.load_data_btn.click()
    tab.analyze_btn.click()
    tab.post_analyze_btn.click()
    tab.writeback_widget.apply_requested.emit()
    tab.save_data_btn.click()
    tab.save_image_btn.click()
    tab.save_result_btn.click()
    tab.post_save_image_btn.click()

    assert actions.calls == [
        ("run_or_stop", "tab-1"),
        ("load_data", "tab-1"),
        ("analyze", "tab-1"),
        ("post_analyze", "tab-1"),
        ("apply_writeback", "tab-1"),
        ("save_data", "tab-1"),
        ("save_image", "tab-1"),
        ("save_result", "tab-1"),
        ("save_post_image", "tab-1"),
    ]


def _make_pulse_model(ctrl):
    from zcu_tools.gui.app.main.adapter import CfgSchema, make_default_value
    from zcu_tools.gui.app.main.cfg_binding import MeasureCfgBindings
    from zcu_tools.gui.app.main.cfg_schemas import _MODULE_SPEC_FACTORIES

    spec = _MODULE_SPEC_FACTORIES["pulse"]()
    return MeasureCfgBindings(ctrl).new_draft(CfgSchema(spec, make_default_value(spec)))


def test_exp_tab_reset_reseeds_cfg_editor_session(qapp):
    """Reset tears down the old cfg-editor session and re-seeds a fresh one over
    the controller's regenerated default schema (user confirms the dialog)."""
    import dataclasses

    from zcu_tools.gui.app.main.ui.main_window import ExpTabWidget

    dialogs = RecordingDialogPresenter(confirm_answers=[True])
    ctrl = _editor_wiring_ctrl()
    first_model = ctrl.get_cfg_editor_draft.return_value
    tab = ExpTabWidget("tab-1", ctrl, dialog_presenter=dialogs)
    actions = _RecordingTabActions()
    snapshot = dataclasses.replace(_snapshot("tab-1"), cfg_schema=_pulse_schema())
    tab.attach(snapshot, actions)

    # After attach: a second session for the reset, returning a NEW model.
    reset_schema = _pulse_schema()
    second_model = _make_pulse_model(ctrl)
    ctrl.reset_tab_cfg.return_value = reset_schema
    ctrl.open_seeded_cfg_editor.reset_mock()
    ctrl.open_seeded_cfg_editor.return_value = ("editor-tab1-v2", [])
    ctrl.get_cfg_editor_draft.return_value = second_model

    tab._on_reset_cfg_clicked()

    assert dialogs.calls[-1].title == "Reset config"
    ctrl.reset_tab_cfg.assert_called_once_with("tab-1")
    # Old session torn down, new one opened keyed by the same tab.
    ctrl.teardown_cfg_editor.assert_called_once_with("editor-tab1")
    ctrl.open_seeded_cfg_editor.assert_called_once()
    kwargs = ctrl.open_seeded_cfg_editor.call_args.kwargs
    assert kwargs["owner_key"] == "tab-1"
    assert kwargs["gc"] is False
    assert tab._cfg_editor_id == "editor-tab1-v2"
    # The form now views the new model (root widget rebuilt).
    assert tab.cfg_form._draft is second_model
    assert tab.cfg_form._draft is not first_model
    assert actions.calls[-1] == ("refresh_interaction", "tab-1")


def test_exp_tab_reset_confirm_no_does_not_reset(qapp):
    """Clicking No in the confirmation dialog must not reset the cfg."""
    import dataclasses

    from zcu_tools.gui.app.main.ui.main_window import ExpTabWidget

    dialogs = RecordingDialogPresenter(confirm_answers=[False])
    ctrl = _editor_wiring_ctrl()
    tab = ExpTabWidget("tab-1", ctrl, dialog_presenter=dialogs)
    actions = _RecordingTabActions()
    snapshot = dataclasses.replace(_snapshot("tab-1"), cfg_schema=_pulse_schema())
    tab.attach(snapshot, actions)

    ctrl.reset_tab_cfg.reset_mock()
    ctrl.open_seeded_cfg_editor.reset_mock()
    actions.calls.clear()

    tab._on_reset_cfg_clicked()

    assert dialogs.calls[-1].title == "Reset config"
    # Controller must not be touched when the user cancels.
    ctrl.reset_tab_cfg.assert_not_called()
    ctrl.open_seeded_cfg_editor.assert_not_called()
    assert actions.calls == []


def test_exp_tab_reset_btn_idle_only_enable(qapp):
    """reset_btn must be enabled when idle and disabled while the tab is busy."""
    from zcu_tools.gui.app.main.ui.main_window import ExpTabWidget

    tab = ExpTabWidget("tab-1", _mock_ctrl())

    # Idle: reset_btn should be enabled.
    tab.update_interaction_state(_snapshot("tab-1", is_running=False))
    assert tab.reset_btn.isEnabled() is True

    # Busy (running): reset_btn must be disabled.
    tab.update_interaction_state(_snapshot("tab-1", is_running=True))
    assert tab.reset_btn.isEnabled() is False

    # Back to idle: reset_btn re-enabled.
    tab.update_interaction_state(_snapshot("tab-1", is_running=False))
    assert tab.reset_btn.isEnabled() is True


def test_exp_tab_reset_does_not_double_connect_schema_changed(qapp):
    """After reset, editing a field commits exactly once — re-seeding must not
    duplicate the widget→controller schema_changed binding."""
    import dataclasses

    from zcu_tools.gui.app.main.ui.main_window import ExpTabWidget

    dialogs = RecordingDialogPresenter(confirm_answers=[True])
    ctrl = _editor_wiring_ctrl()
    tab = ExpTabWidget("tab-1", ctrl, dialog_presenter=dialogs)
    actions = _RecordingTabActions()
    snapshot = dataclasses.replace(_snapshot("tab-1"), cfg_schema=_pulse_schema())
    tab.attach(snapshot, actions)

    reset_schema = _pulse_schema()
    second_model = _make_pulse_model(ctrl)
    ctrl.reset_tab_cfg.return_value = reset_schema
    ctrl.get_cfg_editor_draft.return_value = second_model
    tab._on_reset_cfg_clicked()

    # Drive a single field edit on the re-seeded model and count commits.
    ctrl.update_tab_cfg.reset_mock()
    draft = tab.cfg_form._draft
    assert draft is not None
    scalar = draft.root.fields["gain"]
    scalar.set_value(0.42)

    assert ctrl.update_tab_cfg.call_count == 1
    committed = ctrl.update_tab_cfg.call_args.args[1]
    assert committed.value.fields["gain"].value == pytest.approx(0.42)


def test_main_window_confirms_and_begins_shutdown_when_operations_active(
    qapp,
):
    """User close with work in progress: confirm, then begin_shutdown (which
    cancels-all and waits). The event is ignored now; begin_shutdown is deferred
    to the next event-loop turn and the coordinator drives the real close
    later."""
    from qtpy.QtCore import QCoreApplication
    from qtpy.QtGui import QCloseEvent
    from zcu_tools.gui.app.main.ui.main_window import MainWindow

    dialogs = RecordingDialogPresenter(confirm_answers=[True])
    ctrl = MagicMock()
    ctrl.get_bus.return_value = EventBus()
    ctrl.active_operation_count.return_value = 2
    window = MainWindow(ctrl, dialog_presenter=dialogs)
    event = QCloseEvent()

    window.closeEvent(event)

    assert dialogs.calls[-1].title == "Operations in progress"
    assert event.isAccepted() is False  # async wait — not closed yet
    QCoreApplication.processEvents()  # drain the deferred singleShot(0)
    ctrl.begin_shutdown.assert_called_once_with(window._perform_close)


def test_main_window_declining_confirmation_keeps_window_open(qapp):
    from qtpy.QtGui import QCloseEvent
    from zcu_tools.gui.app.main.ui.main_window import MainWindow

    dialogs = RecordingDialogPresenter(confirm_answers=[False])
    ctrl = MagicMock()
    ctrl.get_bus.return_value = EventBus()
    ctrl.active_operation_count.return_value = 1
    window = MainWindow(ctrl, dialog_presenter=dialogs)
    event = QCloseEvent()

    window.closeEvent(event)

    assert dialogs.calls[-1].title == "Operations in progress"
    assert event.isAccepted() is False
    ctrl.begin_shutdown.assert_not_called()


def test_main_window_persists_session_on_close_when_idle(qapp):
    """Idle close: no confirmation; closeEvent ignores the event and *defers*
    begin_shutdown to the next event-loop turn (so it never re-enters
    self.close() within the closeEvent stack — the single-click bug). After the
    deferred turn fires, begin_shutdown runs _perform_close → persist."""
    from qtpy.QtCore import QCoreApplication
    from qtpy.QtGui import QCloseEvent
    from zcu_tools.gui.app.main.ui.main_window import MainWindow

    ctrl = MagicMock()
    ctrl.get_bus.return_value = EventBus()
    ctrl.active_operation_count.return_value = 0
    # The real coordinator runs on_closed once nothing is pending; here drive it
    # synchronously so the deferred turn exercises _perform_close.
    ctrl.begin_shutdown.side_effect = lambda on_closed: on_closed()
    window = MainWindow(ctrl)
    event = QCloseEvent()

    window.closeEvent(event)

    # Deferred: closeEvent must NOT have begun shutdown synchronously.
    assert event.isAccepted() is False
    ctrl.begin_shutdown.assert_not_called()

    # Drain the singleShot(0) — the deferred turn runs begin_shutdown.
    QCoreApplication.processEvents()

    ctrl.begin_shutdown.assert_called_once_with(window._perform_close)
    ctrl.persist_all.assert_called_once_with()


def test_main_window_close_removes_event_bus_subscriptions(qapp):
    from qtpy.QtCore import QCoreApplication
    from qtpy.QtGui import QCloseEvent
    from zcu_tools.gui.app.main.events.run import RunFinishedPayload, RunStartedPayload
    from zcu_tools.gui.app.main.events.tab import (
        TabAddedPayload,
        TabClosedPayload,
        TabContentChangedPayload,
        TabInteractionChangedPayload,
    )
    from zcu_tools.gui.app.main.ui.main_window import MainWindow
    from zcu_tools.gui.session.events import (
        ContextSwitchedPayload,
        DeviceChangedPayload,
        DeviceSetupFinishedPayload,
        DeviceSetupStartedPayload,
        MlChangedPayload,
        PredictorChangedPayload,
        SocChangedPayload,
    )

    ctrl = MagicMock()
    bus = EventBus()
    ctrl.get_bus.return_value = bus
    ctrl.active_operation_count.return_value = 0
    ctrl.begin_shutdown.side_effect = lambda on_closed: on_closed()
    window = MainWindow(ctrl)

    payload_types = (
        TabInteractionChangedPayload,
        RunStartedPayload,
        RunFinishedPayload,
        ContextSwitchedPayload,
        MlChangedPayload,
        TabAddedPayload,
        TabClosedPayload,
        TabContentChangedPayload,
        PredictorChangedPayload,
        SocChangedPayload,
        DeviceSetupStartedPayload,
        DeviceSetupFinishedPayload,
        DeviceChangedPayload,
    )
    for payload_type in payload_types:
        assert bus._subs.get(payload_type)

    window.closeEvent(QCloseEvent())
    QCoreApplication.processEvents()

    for payload_type in payload_types:
        assert bus._subs.get(payload_type, []) == []


def test_new_tab_menu_supports_nested_paths(qapp, monkeypatch):
    from qtpy.QtWidgets import QMenu
    from zcu_tools.gui.app.main.ui.main_window import MainWindow

    del qapp
    ctrl = MagicMock()
    ctrl.get_bus.return_value = EventBus()
    ctrl.get_adapter_names.return_value = [
        "fake",
        "twotone/rabi/length",
        "twotone/rabi/amp",
    ]
    window = MainWindow(ctrl)

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

    window._toolbar.show_new_tab_menu()

    ctrl.new_tab.assert_called_once_with("twotone/rabi/length")


def test_show_analysis_figure_draws_canvas(qapp, monkeypatch):
    from matplotlib.figure import Figure
    from zcu_tools.gui.app.main.ui.main_window import ExpTabWidget

    del qapp
    tab = ExpTabWidget("tab-1", _mock_ctrl())
    canvas = MagicMock()

    monkeypatch.setattr(
        "zcu_tools.gui.app.main.ui.exp_tab_widget.attach_existing_figure_to_container",
        lambda fig, container: canvas,
    )

    tab.show_analysis_figure(Figure())

    canvas.draw.assert_called_once_with()


def test_show_analysis_figure_keeps_two_figures_coexisting(qapp):
    """The analyze figure and post figure share one container's stack; showing
    one brings it to front without evicting the other (the post-analysis shared-
    container regression). The most recently shown figure is current; both
    canvases stay alive in the stack."""
    from matplotlib.figure import Figure
    from qtpy.QtWidgets import QApplication
    from zcu_tools.gui.app.main.ui.main_window import ExpTabWidget

    del qapp
    tab = ExpTabWidget("tab-1", _mock_ctrl())
    tab.show()
    QApplication.processEvents()

    fig1 = Figure()  # run/analyze figure
    fig2 = Figure()  # post-analysis figure
    stack = tab._figure_container._stack

    tab.show_analysis_figure(fig1)
    first_canvas = stack.currentWidget()
    assert first_canvas is not None

    tab.show_analysis_figure(fig2)
    second_canvas = stack.currentWidget()
    assert second_canvas is not None
    assert second_canvas is not first_canvas

    # Both canvases coexist (placeholder + 2 canvases); fig2 is current.
    assert stack.count() == 3
    assert stack.indexOf(first_canvas) >= 0

    # Re-showing fig1 brings it back to front without deleting fig2's canvas.
    tab.show_analysis_figure(fig1)
    assert stack.currentWidget() is first_canvas
    assert stack.count() == 3


# ---------------------------------------------------------------------------
# FeedbackPanel docking gate (ADR-0025 C3): op-count AND agent-connected
# ---------------------------------------------------------------------------


def _gate_window(
    qapp,
    *,
    op_count: int,
    agent_connected: bool,
    running_tab_id: str | None = None,
    active_tab_id: str | None = None,
):
    """Build a MainWindow + register a real ExpTabWidget per provided tab id.

    Returns (window, {tab_id: ExpTabWidget}). The C3 gate inputs and the
    running/active-tab resolution are stubbed on the mock controller; the tab
    widgets are real so mount_feedback_panel docks into a live plot_layout.
    """
    from zcu_tools.gui.app.main.ui.main_window import ExpTabWidget, MainWindow

    del qapp
    ctrl = MagicMock()
    ctrl.get_bus.return_value = EventBus()
    ctrl.get_persisted_startup.return_value = PersistedStartup(left_panel_width=500)
    ctrl.active_operation_count.return_value = op_count
    ctrl.has_agent_connected.return_value = agent_connected
    ctrl.can_cancel_active_operation.return_value = False
    ctrl.get_running_tab_id.return_value = running_tab_id
    ctrl.get_active_tab_id.return_value = active_tab_id
    ctrl.has_tab.side_effect = lambda tid: tid in tabs

    window = MainWindow(ctrl)

    tabs: dict[str, ExpTabWidget] = {}
    for tid in {t for t in (running_tab_id, active_tab_id) if t is not None}:
        tab_w = ExpTabWidget(tid, ctrl)
        tabs[tid] = tab_w
        window._tab_widgets[tid] = tab_w
    return window, tabs


def _panel_docked_below_stack(window, tab_w) -> bool:
    """True iff the window's feedback panel sits at plot_layout index 1, i.e.
    directly below the plot stack (index 0)."""
    layout = tab_w._plot_layout
    panel = _feedback_panel(window)
    return layout.indexOf(panel) == 1 and layout.indexOf(tab_w._plot_stack) == 0


def _feedback_panel(window):
    return window._feedback_dock.panel


def _feedback_host_tab(window):
    return window._feedback_dock.host_tab


def test_feedback_panel_unmounted_when_op_active_but_no_agent(qapp):
    """C3 gate: active op alone is not enough — agent must also be connected."""
    window, tabs = _gate_window(
        qapp, op_count=1, agent_connected=False, active_tab_id="tab-1"
    )
    window.refresh_feedback_widget()
    assert _feedback_host_tab(window) is None
    assert tabs["tab-1"]._plot_layout.indexOf(_feedback_panel(window)) == -1


def test_feedback_panel_unmounted_when_agent_connected_but_no_op(qapp):
    """C3 gate: agent connected alone is not enough — op must also be live."""
    window, tabs = _gate_window(
        qapp, op_count=0, agent_connected=True, active_tab_id="tab-1"
    )
    window.refresh_feedback_widget()
    assert _feedback_host_tab(window) is None
    assert tabs["tab-1"]._plot_layout.indexOf(_feedback_panel(window)) == -1


def test_feedback_panel_unmounted_when_no_op_and_no_agent(qapp):
    """Both conditions false: panel stays unmounted."""
    window, tabs = _gate_window(
        qapp, op_count=0, agent_connected=False, active_tab_id="tab-1"
    )
    window.refresh_feedback_widget()
    assert _feedback_host_tab(window) is None


def test_feedback_panel_unmounted_when_no_tabs(qapp):
    """Edge case: gate true but no target tab → panel stays unmounted."""
    window, _ = _gate_window(qapp, op_count=1, agent_connected=True)
    window.refresh_feedback_widget()
    assert _feedback_host_tab(window) is None


def test_feedback_panel_mounted_below_figure_when_gate_true(qapp):
    """C3 gate satisfied: panel docks into the active tab's plot_layout at
    index 1 (directly below the plot stack), visible and expanded."""
    window, tabs = _gate_window(
        qapp, op_count=1, agent_connected=True, active_tab_id="tab-1"
    )
    window.refresh_feedback_widget()

    tab_w = tabs["tab-1"]
    panel = _feedback_panel(window)
    assert _feedback_host_tab(window) is tab_w
    assert _panel_docked_below_stack(window, tab_w)
    # Default EXPANDED: the collapsible body is not collapsed (toggle checked,
    # body visible relative to the panel — the window itself is not shown in the
    # test, so absolute isVisible() would be False).
    assert panel._toggle_btn is not None
    assert panel._toggle_btn.isChecked()
    assert panel._body.isVisibleTo(panel)


def test_feedback_panel_targets_running_tab_over_active(qapp):
    """Target tab = running tab if one is running, else active tab."""
    window, tabs = _gate_window(
        qapp,
        op_count=1,
        agent_connected=True,
        running_tab_id="run-tab",
        active_tab_id="act-tab",
    )
    window.refresh_feedback_widget()
    assert _feedback_host_tab(window) is tabs["run-tab"]
    assert tabs["act-tab"]._plot_layout.indexOf(_feedback_panel(window)) == -1


def test_feedback_panel_unmounts_and_clears_input_when_gate_drops(qapp):
    """Gate flips false (agent disconnects): panel unmounts and input clears."""
    window, tabs = _gate_window(
        qapp, op_count=1, agent_connected=True, active_tab_id="tab-1"
    )
    window.refresh_feedback_widget()
    panel = _feedback_panel(window)
    panel._input.setText("pending message")
    assert _feedback_host_tab(window) is tabs["tab-1"]

    cast(MagicMock, window._ctrl).has_agent_connected.return_value = False
    window.refresh_feedback_widget()

    assert _feedback_host_tab(window) is None
    assert tabs["tab-1"]._plot_layout.indexOf(panel) == -1
    assert panel._input.text() == ""


def test_feedback_panel_remounts_on_target_tab_change(qapp):
    """If the target tab changes while visible, the panel re-mounts under the
    new tab (and is removed from the old one)."""
    window, tabs = _gate_window(
        qapp,
        op_count=1,
        agent_connected=True,
        running_tab_id="tab-a",
        active_tab_id="tab-b",
    )
    # Add tab-b as a real tab too (active fallback target after run finishes).
    from zcu_tools.gui.app.main.ui.main_window import ExpTabWidget

    window.refresh_feedback_widget()
    assert _feedback_host_tab(window) is tabs["tab-a"]

    # Run finishes: no running tab now, active tab becomes the target.
    cast(MagicMock, window._ctrl).get_running_tab_id.return_value = None
    if "tab-b" not in tabs:
        tab_b = ExpTabWidget("tab-b", window._ctrl)
        tabs["tab-b"] = tab_b
        window._tab_widgets["tab-b"] = tab_b
    window.refresh_feedback_widget()

    assert _feedback_host_tab(window) is tabs["tab-b"]
    assert tabs["tab-a"]._plot_layout.indexOf(_feedback_panel(window)) == -1
    assert _panel_docked_below_stack(window, tabs["tab-b"])
