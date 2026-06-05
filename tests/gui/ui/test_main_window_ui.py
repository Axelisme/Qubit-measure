"""UI structure tests for ExpTabWidget layout decisions."""

from __future__ import annotations

from unittest.mock import MagicMock

from qtpy.QtCore import Qt
from zcu_tools.gui.app.main.adapter import AdapterCapabilities
from zcu_tools.gui.app.main.event_bus import EventBus, GuiEvent, SocChangedPayload
from zcu_tools.gui.app.main.services import TabSnapshot
from zcu_tools.gui.app.main.state import TabInteractionState


def _mock_ctrl() -> MagicMock:
    ctrl = MagicMock()
    ctrl.get_persisted_left_panel_width.return_value = 500
    return ctrl


# Distinguishes "caller did not specify analyze_params" (default to a MagicMock)
# from "caller explicitly wants None" (a non-analysis adapter's snapshot).
_DEFAULT_PARAMS = object()


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
    supports_analysis: bool = True,
    analyze_params: object = _DEFAULT_PARAMS,
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
        ),
        cfg_schema=MagicMock(),
        save_paths_override=None,
        capabilities=AdapterCapabilities(supports_analysis=supports_analysis),
        analyze_params=MagicMock()
        if analyze_params is _DEFAULT_PARAMS
        else analyze_params,
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
    assert tab.analyze_btn.isEnabled() is True
    assert tab.writeback_widget.isEnabled() is True
    assert tab.save_data_btn.isEnabled() is False
    assert tab.save_image_btn.isEnabled() is False
    assert tab.save_result_btn.isEnabled() is False


def test_non_analysis_adapter_hides_analysis_widgets_but_keeps_save(qapp):
    """flux_dep / power_dep adapters (supports_analysis=False) hide only the
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


def test_main_window_run_lock_disables_only_new_tab_and_run(qapp):
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

    assert window._new_tab_btn.isEnabled() is False
    tab_one.update_interaction_state.assert_called_once()
    tab_two.update_interaction_state.assert_called_once()


def test_main_window_soc_changed_refreshes_run_lock(qapp):
    from zcu_tools.gui.app.main.ui.main_window import MainWindow

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
    from zcu_tools.gui.app.main.event_bus import TabContentChangedPayload
    from zcu_tools.gui.app.main.ui.main_window import MainWindow

    ctrl = MagicMock()
    bus = EventBus()
    ctrl.get_bus.return_value = bus
    ctrl.get_tab_snapshot.return_value = _snapshot("tab-1")
    window = MainWindow(ctrl)
    window._tab_widgets["tab-1"] = MagicMock()

    bus.emit(GuiEvent.TAB_CONTENT_CHANGED, TabContentChangedPayload(tab_id="tab-1"))

    ctrl.get_tab_snapshot.assert_called_once_with("tab-1")


def _emit_run_finished(bus, tab_id: str, outcome: str) -> None:
    from zcu_tools.gui.app.main.event_bus import RunFinishedPayload

    bus.emit(GuiEvent.RUN_FINISHED, RunFinishedPayload(tab_id=tab_id, outcome=outcome))


def test_finished_run_auto_switches_to_analysis_tab(qapp):
    """RUN_FINISHED with outcome=finished switches the tab to Analysis — the
    decision reads the outcome straight off the RUN_FINISHED payload."""
    from zcu_tools.gui.app.main.ui.main_window import MainWindow

    ctrl = MagicMock()
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

    ctrl = MagicMock()
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
    """flux_dep / power_dep adapters (supports_analysis=False) keep the second
    tab — its analysis widgets are hidden but the Save section stays — so a
    finished run still switches there, landing the user on Save (where they save
    the 2D sweep). Regression: switching used to be skipped, and earlier the
    whole tab was hidden so the user could not save at all."""
    from zcu_tools.gui.app.main.ui.main_window import MainWindow

    ctrl = MagicMock()
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

    ctrl = MagicMock()
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
    from zcu_tools.gui.app.main.adapter import make_default_value
    from zcu_tools.gui.app.main.cfg_schemas import _MODULE_SPEC_FACTORIES
    from zcu_tools.gui.app.main.live_model import LiveModelEnv, SectionLiveField

    spec = _MODULE_SPEC_FACTORIES["pulse"]()
    model = SectionLiveField(spec, LiveModelEnv(ctrl=ctrl), make_default_value(spec))
    ctrl.open_seeded_cfg_editor.return_value = ("editor-tab1", [])
    ctrl.get_cfg_editor_root.return_value = model
    return ctrl


def _pulse_schema():
    from zcu_tools.gui.app.main.adapter import CfgSchema, make_default_value
    from zcu_tools.gui.app.main.cfg_schemas import _MODULE_SPEC_FACTORIES

    spec = _MODULE_SPEC_FACTORIES["pulse"]()
    return CfgSchema(spec=spec, value=make_default_value(spec))


def test_exp_tab_opens_cfg_editor_on_attach(qapp):
    import dataclasses

    from zcu_tools.gui.app.main.ui.main_window import ExpTabWidget, MainWindow

    ctrl = _editor_wiring_ctrl()
    tab = ExpTabWidget("tab-1", ctrl)
    snapshot = dataclasses.replace(_snapshot("tab-1"), cfg_schema=_pulse_schema())
    tab.attach(snapshot, MainWindow(ctrl))

    # Opened a gc=False seeded session keyed by the tab id, and attached the
    # widget to the service-owned model.
    ctrl.open_seeded_cfg_editor.assert_called_once()
    kwargs = ctrl.open_seeded_cfg_editor.call_args.kwargs
    assert kwargs["owner_key"] == "tab-1"
    assert kwargs["gc"] is False
    assert tab._cfg_editor_id == "editor-tab1"
    assert tab.cfg_form.get_live_root() is ctrl.get_cfg_editor_root.return_value


def test_exp_tab_tears_down_cfg_editor_on_detach(qapp):
    import dataclasses

    from zcu_tools.gui.app.main.ui.main_window import ExpTabWidget, MainWindow

    ctrl = _editor_wiring_ctrl()
    tab = ExpTabWidget("tab-1", ctrl)
    snapshot = dataclasses.replace(_snapshot("tab-1"), cfg_schema=_pulse_schema())
    tab.attach(snapshot, MainWindow(ctrl))
    tab.detach()

    ctrl.teardown_cfg_editor.assert_called_once_with("editor-tab1")
    assert tab._cfg_editor_id is None


def test_main_window_confirms_and_begins_shutdown_when_operations_active(
    qapp, monkeypatch
):
    """User close with work in progress: confirm, then begin_shutdown (which
    cancels-all and waits). The event is ignored now; begin_shutdown is deferred
    to the next event-loop turn and the coordinator drives the real close
    later."""
    from qtpy.QtCore import QCoreApplication
    from qtpy.QtGui import QCloseEvent
    from qtpy.QtWidgets import QMessageBox
    from zcu_tools.gui.app.main.ui.main_window import MainWindow

    ctrl = MagicMock()
    ctrl.get_bus.return_value = EventBus()
    ctrl.active_operation_count.return_value = 2
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *args, **kwargs: QMessageBox.StandardButton.Yes,
    )
    window = MainWindow(ctrl)
    event = QCloseEvent()

    window.closeEvent(event)

    assert event.isAccepted() is False  # async wait — not closed yet
    QCoreApplication.processEvents()  # drain the deferred singleShot(0)
    ctrl.begin_shutdown.assert_called_once_with(window._perform_close)


def test_main_window_declining_confirmation_keeps_window_open(qapp, monkeypatch):
    from qtpy.QtGui import QCloseEvent
    from qtpy.QtWidgets import QMessageBox
    from zcu_tools.gui.app.main.ui.main_window import MainWindow

    ctrl = MagicMock()
    ctrl.get_bus.return_value = EventBus()
    ctrl.active_operation_count.return_value = 1
    monkeypatch.setattr(
        QMessageBox,
        "question",
        lambda *args, **kwargs: QMessageBox.StandardButton.No,
    )
    window = MainWindow(ctrl)
    event = QCloseEvent()

    window.closeEvent(event)

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
    from zcu_tools.gui.app.main.ui.main_window import ExpTabWidget

    del qapp
    tab = ExpTabWidget("tab-1", _mock_ctrl())
    canvas = MagicMock()

    monkeypatch.setattr(
        "zcu_tools.gui.app.main.ui.main_window.attach_existing_figure_to_container",
        lambda fig, container: canvas,
    )

    tab.show_analysis_figure(Figure())

    canvas.draw.assert_called_once_with()
    assert tab._canvas_widget is canvas


def test_show_analysis_figure_keeps_new_canvas_current_when_replacing_old(qapp):
    from matplotlib.figure import Figure
    from qtpy.QtWidgets import QApplication
    from zcu_tools.gui.app.main.ui.main_window import ExpTabWidget

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
