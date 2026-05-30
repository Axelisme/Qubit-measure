"""UI structure tests for ExpTabWidget layout decisions."""

from __future__ import annotations

from unittest.mock import MagicMock

from qtpy.QtCore import Qt
from zcu_tools.gui.adapter import AdapterCapabilities
from zcu_tools.gui.event_bus import EventBus, GuiEvent, SocChangedPayload
from zcu_tools.gui.services import TabViewSnapshot
from zcu_tools.gui.state import TabInteractionState


def _mock_ctrl() -> MagicMock:
    ctrl = MagicMock()
    ctrl.get_persisted_left_panel_width.return_value = 500
    return ctrl


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
) -> TabViewSnapshot:
    return TabViewSnapshot(
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
        ),
        cfg_schema=MagicMock(),
        capabilities=AdapterCapabilities(),
        analyze_params=MagicMock(),
        writeback_items=(),
        save_paths=None,
        figure=None,
    )


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
    assert tab.writeback_widget.isEnabled() is False
    assert tab.save_image_btn.isEnabled() is False  # disabled because is_analyzing


def test_exp_tab_keeps_analyze_enabled_while_other_tab_running(qapp):
    from zcu_tools.gui.ui.main_window import ExpTabWidget

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
    from zcu_tools.gui.ui.main_window import ExpTabWidget

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
    assert tab.save_both_btn.isEnabled() is False
    assert tab.run_btn.text() == "Run"
    assert tab.run_btn.toolTip() == "Tab is busy"


def test_exp_tab_run_tooltip_shows_no_soc_reason(qapp):
    from zcu_tools.gui.ui.main_window import ExpTabWidget

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
    from zcu_tools.gui.ui.main_window import ExpTabWidget

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
    from zcu_tools.gui.ui.main_window import ExpTabWidget

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
    assert tab.analyze_btn.isEnabled() is True
    assert tab.writeback_widget.isEnabled() is True
    assert tab.save_data_btn.isEnabled() is False
    assert tab.save_image_btn.isEnabled() is False
    assert tab.save_both_btn.isEnabled() is False


def test_main_window_run_lock_disables_only_new_tab_and_run(qapp):
    from zcu_tools.gui.ui.main_window import MainWindow

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

    assert window._new_tab_btn.isEnabled() is False
    tab_one.update_interaction_state.assert_called_once()
    tab_two.update_interaction_state.assert_called_once()


def test_main_window_soc_changed_refreshes_run_lock(qapp):
    from zcu_tools.gui.ui.main_window import MainWindow

    ctrl = MagicMock()
    bus = EventBus()
    ctrl.get_bus.return_value = bus
    ctrl.get_running_tab_id.return_value = None
    ctrl.has_tab.return_value = True
    ctrl.get_tab_snapshot.return_value = _snapshot("tab-1", has_soc=False)

    window = MainWindow(ctrl)
    tab = MagicMock()
    window._tab_widgets["tab-1"] = tab

    bus.emit(GuiEvent.SOC_CHANGED, SocChangedPayload(soc=None, soccfg=None))

    tab.update_interaction_state.assert_called()


def test_main_window_content_event_queries_single_tab_snapshot(qapp):
    from zcu_tools.gui.event_bus import TabContentChangedPayload
    from zcu_tools.gui.ui.main_window import MainWindow

    ctrl = MagicMock()
    bus = EventBus()
    ctrl.get_bus.return_value = bus
    ctrl.get_tab_snapshot.return_value = _snapshot("tab-1")
    window = MainWindow(ctrl)
    window._tab_widgets["tab-1"] = MagicMock()

    bus.emit(GuiEvent.TAB_CONTENT_CHANGED, TabContentChangedPayload(tab_id="tab-1"))

    ctrl.get_tab_snapshot.assert_called_once_with("tab-1")


def _editor_wiring_ctrl() -> MagicMock:
    """Mock ctrl that also satisfies LiveModelEnv for a real populate()."""
    ctrl = MagicMock()
    ctrl.get_persisted_left_panel_width.return_value = 500
    ctrl.get_bus.return_value = EventBus()
    ctrl.get_current_md.return_value = MagicMock()
    ctrl.get_current_ml.return_value = MagicMock()
    ctrl.list_device_names.return_value = []
    ctrl.has_soc.return_value = False

    # populate_cfg now opens a service-owned (gc=False) seeded session and
    # attaches the widget to the service-owned model (ADR-0010). Build a real
    # SectionLiveField for get_cfg_editor_root so attach() works.
    from zcu_tools.gui.adapter import make_default_value
    from zcu_tools.gui.cfg_schemas import _MODULE_SPEC_FACTORIES
    from zcu_tools.gui.live_model import LiveModelEnv, SectionLiveField

    spec = _MODULE_SPEC_FACTORIES["pulse"]()
    model = SectionLiveField(spec, LiveModelEnv(ctrl=ctrl), make_default_value(spec))
    ctrl.open_seeded_cfg_editor.return_value = ("editor-tab1", [])
    ctrl.get_cfg_editor_root.return_value = model
    return ctrl


def _pulse_schema():
    from zcu_tools.gui.adapter import CfgSchema, make_default_value
    from zcu_tools.gui.cfg_schemas import _MODULE_SPEC_FACTORIES

    spec = _MODULE_SPEC_FACTORIES["pulse"]()
    return CfgSchema(spec=spec, value=make_default_value(spec))


def test_exp_tab_opens_cfg_editor_on_populate(qapp):
    from zcu_tools.gui.ui.main_window import ExpTabWidget, MainWindow

    ctrl = _editor_wiring_ctrl()
    tab = ExpTabWidget("tab-1", ctrl)
    tab.populate_cfg(_pulse_schema(), ctrl)
    tab.bind_to_controller(MainWindow(ctrl))

    # Opened a gc=False seeded session keyed by the tab id, and attached the
    # widget to the service-owned model.
    ctrl.open_seeded_cfg_editor.assert_called_once()
    kwargs = ctrl.open_seeded_cfg_editor.call_args.kwargs
    assert kwargs["owner_key"] == "tab-1"
    assert kwargs["gc"] is False
    assert tab._cfg_editor_id == "editor-tab1"
    assert tab.cfg_form.get_live_root() is ctrl.get_cfg_editor_root.return_value


def test_exp_tab_tears_down_cfg_editor_on_unbind(qapp):
    from zcu_tools.gui.ui.main_window import ExpTabWidget, MainWindow

    ctrl = _editor_wiring_ctrl()
    tab = ExpTabWidget("tab-1", ctrl)
    tab.populate_cfg(_pulse_schema(), ctrl)
    tab.bind_to_controller(MainWindow(ctrl))
    tab.unbind_from_controller()

    ctrl.teardown_cfg_editor.assert_called_once_with("editor-tab1")
    assert tab._cfg_editor_id is None


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
