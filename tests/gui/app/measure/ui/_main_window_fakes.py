"""Shared controller and snapshot specimens for MainWindow view contracts."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import replace
from unittest.mock import MagicMock

from qtpy.QtWidgets import (
    QTabWidget,
)
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    AdapterGuide,
    AnalysisMode,
)
from zcu_tools.gui.app.measure.artifacts import (
    ArtifactSectionSnapshot,
    ArtifactSnapshot,
    SaveStatus,
)
from zcu_tools.gui.app.measure.services import TabSnapshot
from zcu_tools.gui.app.measure.state import TabInteractionState
from zcu_tools.gui.app.measure.ui.exp_tab_widget import ExpTabWidget
from zcu_tools.gui.app.measure.ui.main_window import MainWindow
from zcu_tools.gui.app.measure.writeback import WritebackItem
from zcu_tools.gui.control_identity import ControlIdentity
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
from zcu_tools.gui.session.types import SessionEnv

from tests.gui._dialog_fakes import RecordingDialogPresenter
from tests.gui.app.measure._form_fakes import configure_form_lookup
from tests.gui.app.measure.artifacts._snapshots import ready_figures, with_artifacts


def mock_ctrl(control_factory: Callable[[], ControlIdentity]) -> MagicMock:
    """Build a controller fake with real form resources and retained authority.

    Pass the function-scoped authority factory supplied by the UI conftest so
    its ControlIdentity outlives every Qt view created with this fake.
    """
    ctrl = MagicMock()
    ctrl.control_identity = control_factory()
    ctrl.get_adapter_guide.return_value = AdapterGuide(
        markdown="# Synthetic guide\n", resource_dir=None
    )
    configure_form_lookup(ctrl)
    ctrl.get_left_panel_width.return_value = 500
    ctrl.writeback_control.get_writeback_values_for_pane.return_value.__getitem__.return_value.actual_path = "r_f"
    return ctrl


def experiment_tabs(window: MainWindow) -> QTabWidget:
    """Return the window's experiment tabs, asserting the widget exists."""
    tabs = window.findChild(QTabWidget)
    assert tabs is not None
    return tabs


def apply_window_defaults(ctrl: MagicMock) -> MagicMock:
    """Set the minimal return values required by MainWindow.__init__ on a mock ctrl.

    MainWindow uses active_operation_count() for shutdown confirmation.
    Stub the count to keep tests independent of live operations.
    """
    ctrl.active_operation_count.return_value = 0
    ctrl.get_active_context_label.return_value = None
    ctrl.has_project.return_value = False
    ctrl.context_control.get_alias_rows.return_value = ()
    ctrl.get_session_env.return_value = SessionEnv(
        parameters=None, soc=None, soccfg=None
    )
    return ctrl


# Distinguishes "caller did not specify analyze_params" (default to a MagicMock)
# from "caller explicitly wants None" (a non-analysis adapter's snapshot).
_DEFAULT_PARAMS = object()


def make_tab_snapshot(
    tab_id: str,
    *,
    global_run_active: bool = False,
    is_running: bool = False,
    is_analyzing: bool = False,
    has_context: bool = True,
    has_active_context: bool = True,
    has_soc: bool = True,
    has_run_result: bool = True,
    has_analyze_result: bool = True,
    has_figure: bool = True,
    has_post_analyze_result: bool = False,
    supports_analysis: bool = True,
    supports_post_analysis: bool = False,
    supports_load_data: bool = False,
    analyze_params: object = _DEFAULT_PARAMS,
    post_analyze_params: object | None = None,
    writeback_items: tuple[WritebackItem, ...] = (),
    figure: object = _DEFAULT_PARAMS,
    post_figure: object = _DEFAULT_PARAMS,
    data_status: SaveStatus | None = None,
) -> TabSnapshot:
    """Build one tab's pane and artifact projection for synthetic view tests.

    tab_id identifies the tab. The global_run_active, is_running, is_analyzing,
    has_context, has_active_context, has_soc and result/figure booleans set the
    interaction gates. supports_analysis, supports_post_analysis and
    supports_load_data select capabilities. Run and analysis result flags also
    create distinct opaque pane results; no live acquisition is performed.

    Omitted analyze_params creates a mock; explicit None remains None.
    post_analyze_params is passed through. Omitted figure creates a Figure only
    when has_figure and has_analyze_result are true; omitted post_figure creates
    one when has_post_analyze_result is true. Explicit Figures retain identity,
    explicit None clears them, and other figure values fail an assertion.
    writeback_items populates only the primary analysis pane. data_status None
    or NOT_SAVED keeps the planned artifacts; other statuses replace the Run
    artifact status and mark its existing output paths saved.
    """
    from zcu_tools.gui.app.measure.services.ports import (
        AnalysisPaneSnapshot,
        PostAnalysisPaneSnapshot,
        RunPaneSnapshot,
        SavePaneSnapshot,
    )

    # Resolve analysis params
    resolved_analyze_params = (
        MagicMock() if analyze_params is _DEFAULT_PARAMS else analyze_params
    )
    # Resolve figures: allow explicit None or Figure override
    if figure is _DEFAULT_PARAMS:
        from matplotlib.figure import Figure

        figure_obj: Figure | None = (
            Figure() if has_figure and has_analyze_result else None
        )
    else:
        from matplotlib.figure import Figure

        assert figure is None or isinstance(figure, Figure)
        figure_obj = figure
    if post_figure is _DEFAULT_PARAMS:
        from matplotlib.figure import Figure as _Fig2

        post_figure_obj: Figure | None = _Fig2() if has_post_analyze_result else None
    else:
        assert post_figure is None or isinstance(post_figure, Figure)
        post_figure_obj = post_figure

    # Capabilities include load_data
    caps = AdapterCapabilities(
        analysis=AnalysisMode.FIT if supports_analysis else AnalysisMode.NONE,
        post_analysis=supports_post_analysis,
        load_data=supports_load_data,
    )

    # Pane snapshots
    run_snap = RunPaneSnapshot(
        result=object() if has_run_result else None,
        source_path=None,
    )
    analysis_snap = AnalysisPaneSnapshot(
        params=resolved_analyze_params,
        result=object() if has_analyze_result else None,
        figures=ready_figures(figure_obj),
        writeback_items=tuple(writeback_items),
    )
    post_snap = PostAnalysisPaneSnapshot(
        params=post_analyze_params,
        result=object() if has_post_analyze_result else None,
        figures=ready_figures(post_figure_obj),
        writeback_items=(),
    )
    snapshot = TabSnapshot(
        adapter_name="fake",
        tab_id=tab_id,
        interaction=TabInteractionState(
            global_run_active=global_run_active,
            is_running=is_running,
            is_analyzing=is_analyzing,
            has_context=has_context,
            has_active_context=has_active_context,
            has_soc=has_soc,
            has_run_result=has_run_result,
            has_analyze_result=has_analyze_result,
            has_figure=has_figure,
            has_post_analyze_result=has_post_analyze_result,
        ),
        cfg_input={},
        capabilities=caps,
        run=run_snap,
        analysis=analysis_snap,
        post_analysis=post_snap,
        save=SavePaneSnapshot(),
    )
    snapshot = with_artifacts(snapshot)
    if data_status is None or data_status is SaveStatus.NOT_SAVED:
        return snapshot
    # Close-guard specimens retain planned paths and project the Run save facts.
    sections: list[ArtifactSectionSnapshot] = []
    for section in snapshot.artifacts:
        if section.section != "run":
            sections.append(section)
            continue
        artifacts: list[ArtifactSnapshot] = []
        for artifact in section.artifacts:
            members = tuple(
                replace(
                    member,
                    outputs=tuple(
                        replace(output, status="saved") for output in member.outputs
                    ),
                )
                for member in artifact.members
            )
            artifacts.append(replace(artifact, status=data_status, members=members))
        sections.append(replace(section, artifacts=tuple(artifacts)))
    return replace(snapshot, artifacts=tuple(sections))


def tab_pane_tabs(tab: ExpTabWidget) -> QTabWidget:
    """Return the left tab widget, failing if it is missing."""
    tabs = tab.findChild(QTabWidget)
    assert tabs is not None
    return tabs


def editor_wiring_ctrl(control_factory: Callable[[], ControlIdentity]) -> MagicMock:
    """Build an idle view facade using the retained authority factory.

    Its real tab form resources are owned by the returned controller fake.
    """
    ctrl = mock_ctrl(control_factory)
    ctrl.active_operation_count.return_value = 0
    return ctrl


def setup_window_with_tabs(
    control_factory: Callable[[], ControlIdentity],
    dialog_presenter: RecordingDialogPresenter | None = None,
    active_operations: int = 0,
) -> tuple[MainWindow, MagicMock]:
    """Return an empty window and fake controller using retained authority.

    control_factory must retain each ControlIdentity until Qt cleanup. Use
    dialog_presenter to record decisions, or omit it for a recording presenter.
    active_operations sets the fake controller's shutdown confirmation count.
    The caller owns the window and adds tab specimens with add_tab.
    """
    ctrl = apply_window_defaults(editor_wiring_ctrl(control_factory))
    ctrl.get_bus.return_value = EventBus()
    ctrl.active_operation_count.return_value = active_operations
    ctrl.has_tab.side_effect = lambda tid: True
    dialogs = dialog_presenter or RecordingDialogPresenter()
    window = MainWindow(ctrl, templates=lambda: {}, dialog_presenter=dialogs)
    return window, ctrl


def add_tab(
    window: MainWindow,
    tab_id: str,
    *,
    ctrl: MagicMock,
    has_run: bool = False,
    saved: bool = False,
) -> ExpTabWidget:
    """Add tab_id to window using its fake ctrl, preserving the selection.

    has_run creates a Run result; saved marks its planned artifacts saved.
    Restore ctrl's snapshot return value after construction and return the
    exact created view. The caller owns its removal and the window's cleanup.
    """
    tabs = experiment_tabs(window)
    selected = tabs.currentWidget()
    snap = make_tab_snapshot(
        tab_id,
        has_run_result=has_run,
        data_status=SaveStatus.SAVED if saved else None,
    )
    original_snapshot = ctrl.get_tab_snapshot.return_value
    ctrl.get_tab_snapshot.return_value = snap
    try:
        window.add_tab_widget(tab_id, tab_id)
    finally:
        ctrl.get_tab_snapshot.return_value = original_snapshot
    tab = next(tab for tab in window.findChildren(ExpTabWidget) if tab.tab_id == tab_id)
    if selected is not None:
        tabs.setCurrentWidget(selected)
    return tab
