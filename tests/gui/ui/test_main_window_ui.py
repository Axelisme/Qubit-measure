"""UI structure tests for ExpTabWidget layout decisions."""

from __future__ import annotations

from unittest.mock import MagicMock

from qtpy.QtCore import Qt
from zcu_tools.gui.event_bus import EventBus, GuiEvent, SocChangedPayload
from zcu_tools.gui.state import TabInteractionState


def _mock_ctrl() -> MagicMock:
    ctrl = MagicMock()
    ctrl.get_persisted_left_panel_width.return_value = 500
    return ctrl


def test_left_panel_toggle_is_attached_to_tab_bar(qapp):
    from qtpy.QtWidgets import QApplication
    from zcu_tools.gui.ui.main_window import ExpTabWidget

    tab = ExpTabWidget("tab-1", _mock_ctrl())
    tab.show()
    QApplication.processEvents()

    corner = tab._left_tabs.cornerWidget(Qt.TopLeftCorner)  # type: ignore[attr-defined]
    assert corner is None
    assert tab._left_edge_handle.isVisible() is True


def test_left_panel_toggle_uses_collapsed_boundary_handle(qapp):
    from qtpy.QtWidgets import QApplication
    from zcu_tools.gui.ui.main_window import ExpTabWidget

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
    from zcu_tools.gui.ui.main_window import ExpTabWidget

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
    from zcu_tools.gui.ui.main_window import ExpTabWidget

    tab = ExpTabWidget("tab-1", _mock_ctrl())
    tab.update_writeback_items([MagicMock(selected=True)])
    tab.update_interaction_state(
        TabInteractionState(
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
    assert tab.writeback_widget.isEnabled() is False
    assert tab.save_image_btn.isEnabled() is False  # disabled because is_analyzing


def test_exp_tab_keeps_analyze_enabled_while_other_tab_running(qapp):
    from zcu_tools.gui.ui.main_window import ExpTabWidget

    tab = ExpTabWidget("tab-1", _mock_ctrl())
    tab.update_interaction_state(
        TabInteractionState(
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
    from zcu_tools.gui.ui.main_window import ExpTabWidget

    tab = ExpTabWidget("tab-1", _mock_ctrl())
    tab.update_interaction_state(
        TabInteractionState(
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
    assert tab.save_both_btn.isEnabled() is False
    assert tab.run_btn.text() == "Run"
    assert tab.run_btn.toolTip() == "Tab is busy"


def test_exp_tab_run_tooltip_shows_no_soc_reason(qapp):
    from zcu_tools.gui.ui.main_window import ExpTabWidget

    tab = ExpTabWidget("tab-1", _mock_ctrl())
    tab.update_interaction_state(
        TabInteractionState(
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
    from zcu_tools.gui.ui.main_window import ExpTabWidget

    tab = ExpTabWidget("tab-1", _mock_ctrl())
    tab.cfg_form.first_invalid_reason = MagicMock(
        return_value="modules.readout: invalid"
    )
    tab.cfg_form.is_valid = MagicMock(return_value=False)
    tab.update_interaction_state(
        TabInteractionState(
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
    from zcu_tools.gui.ui.main_window import ExpTabWidget

    tab = ExpTabWidget("tab-1", _mock_ctrl())
    tab.update_writeback_items([MagicMock(selected=True)])
    tab.update_interaction_state(
        TabInteractionState(
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
    assert tab.analyze_btn.isEnabled() is True
    assert tab.writeback_widget.isEnabled() is True
    assert tab.save_data_btn.isEnabled() is False
    assert tab.save_image_btn.isEnabled() is False
    assert tab.save_both_btn.isEnabled() is False


def test_main_window_run_lock_disables_only_new_tab_and_run(qapp):
    from zcu_tools.gui.ui.main_window import MainWindow

    ctrl = MagicMock()
    ctrl.get_bus.return_value = EventBus()
    ctrl.is_run_active.return_value = True
    ctrl.has_context.return_value = True
    ctrl.has_active_context.return_value = True
    ctrl.has_soc.return_value = True
    ctrl.has_tab.return_value = True
    ctrl.has_run_result.return_value = True
    ctrl.has_analyze_result.return_value = True
    ctrl.is_tab_running.side_effect = lambda tab_id: tab_id == "tab-1"
    ctrl.is_tab_analyzing.return_value = False
    ctrl.is_tab_saving_data.return_value = False

    window = MainWindow(ctrl)
    tab_one = MagicMock()
    tab_two = MagicMock()
    window._tab_widgets["tab-1"] = tab_one
    window._tab_widgets["tab-2"] = tab_two

    window.refresh_run_lock("tab-1")

    assert window._new_tab_btn.isEnabled() is False
    tab_one.update_interaction_state.assert_called_once()
    tab_two.update_interaction_state.assert_called_once()


def test_main_window_soc_changed_refreshes_run_lock(qapp):
    from zcu_tools.gui.ui.main_window import MainWindow

    ctrl = MagicMock()
    bus = EventBus()
    ctrl.get_bus.return_value = bus
    ctrl.is_run_active.return_value = False
    ctrl.has_context.return_value = True
    ctrl.has_active_context.return_value = True
    ctrl.has_soc.return_value = False
    ctrl.has_tab.return_value = True
    ctrl.has_run_result.return_value = False
    ctrl.has_analyze_result.return_value = False
    ctrl.is_tab_running.return_value = False
    ctrl.is_tab_analyzing.return_value = False
    ctrl.is_tab_saving_data.return_value = False
    ctrl.has_figure.return_value = False

    window = MainWindow(ctrl)
    tab = MagicMock()
    window._tab_widgets["tab-1"] = tab

    bus.emit(GuiEvent.SOC_CHANGED, SocChangedPayload(soc=None, soccfg=None))

    tab.update_interaction_state.assert_called()


def test_main_window_cancel_setup_before_closing(qapp, monkeypatch):
    from qtpy.QtGui import QCloseEvent
    from qtpy.QtWidgets import QMessageBox
    from zcu_tools.gui.services.device import DeviceSetupSnapshot
    from zcu_tools.gui.ui.main_window import MainWindow

    ctrl = MagicMock()
    ctrl.get_bus.return_value = EventBus()
    ctrl.get_active_device_setup.return_value = DeviceSetupSnapshot(
        device_name="flux", progress=()
    )
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *args, **kwargs: QMessageBox.StandardButton.Yes,
    )
    window = MainWindow(ctrl)
    event = QCloseEvent()

    window.closeEvent(event)

    assert event.isAccepted() is False
    assert window._shutdown_waiting_for_device_setup is True
    ctrl.cancel_device_operation.assert_called_once_with("flux")


def test_main_window_persists_session_on_close_when_idle(qapp):
    from qtpy.QtGui import QCloseEvent
    from zcu_tools.gui.ui.main_window import MainWindow

    ctrl = MagicMock()
    ctrl.get_bus.return_value = EventBus()
    ctrl.get_active_device_setup.return_value = None
    window = MainWindow(ctrl)
    event = QCloseEvent()

    window.closeEvent(event)

    ctrl.persist_tabs_session.assert_called_once_with()


def test_new_tab_menu_supports_nested_paths(qapp, monkeypatch):
    from qtpy.QtWidgets import QMenu
    from zcu_tools.gui.ui.main_window import MainWindow

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
            if child is not None:
                found = _find_action_by_data(child, target)
                if found is not None:
                    return found
        return None

    def _fake_exec(self, *_args, **_kwargs):
        action = _find_action_by_data(self, "twotone/rabi/length")
        assert action is not None
        return action

    monkeypatch.setattr(QMenu, "exec", _fake_exec)

    window._on_new_tab_requested()

    ctrl.new_tab.assert_called_once_with("twotone/rabi/length")


def test_show_analysis_figure_draws_canvas(qapp, monkeypatch):
    from matplotlib.figure import Figure
    from zcu_tools.gui.ui.main_window import ExpTabWidget

    del qapp
    tab = ExpTabWidget("tab-1", _mock_ctrl())
    canvas = MagicMock()

    monkeypatch.setattr(
        "zcu_tools.gui.ui.main_window.attach_existing_figure_to_container",
        lambda fig, container: canvas,
    )

    tab.show_analysis_figure(Figure())

    canvas.draw.assert_called_once_with()
    assert tab._canvas_widget is canvas


def test_show_analysis_figure_keeps_new_canvas_current_when_replacing_old(qapp):
    from matplotlib.figure import Figure
    from qtpy.QtWidgets import QApplication
    from zcu_tools.gui.ui.main_window import ExpTabWidget

    del qapp
    tab = ExpTabWidget("tab-1", _mock_ctrl())
    tab.show()
    QApplication.processEvents()

    fig1 = Figure()
    fig2 = Figure()

    tab.show_analysis_figure(fig1)
    first_canvas = tab._canvas_widget
    assert first_canvas is not None
    assert tab._figure_container._stack.currentWidget() is first_canvas

    tab.show_analysis_figure(fig2)
    second_canvas = tab._canvas_widget
    assert second_canvas is not None
    assert second_canvas is not first_canvas
    assert tab._figure_container._stack.currentWidget() is second_canvas
