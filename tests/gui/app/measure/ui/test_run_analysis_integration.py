"""Focused A1-A3 tests for run-analysis-integration (S1/S2)."""

from __future__ import annotations

import dataclasses
from collections.abc import Callable, Iterator
from unittest.mock import MagicMock

import pytest
from matplotlib.figure import Figure
from qtpy.QtCore import QEvent, QPointF, Qt
from qtpy.QtGui import QMouseEvent
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QDialog,
    QLabel,
    QPushButton,
    QScrollArea,
    QTreeWidget,
    QWidget,
)
from zcu_tools.cfg_model import ConstWaveformCfg
from zcu_tools.gui.app.measure.adapter import (
    AdapterCapabilities,
    AdapterGuide,
    AnalysisMode,
    ModuleWriteback,
    ParameterWriteback,
    SessionEnv,
)
from zcu_tools.gui.app.measure.services import TabSnapshot
from zcu_tools.gui.app.measure.state import TabInteractionState
from zcu_tools.gui.cfg import (
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    ScalarSpec,
)
from zcu_tools.gui.control_identity import ControlIdentity
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.plotting.plots import NonPresentingHost, Plots

from tests.gui.app.measure._cfg_fakes import parameter_capture
from tests.gui.app.measure._components import ResonatorSchema
from tests.gui.app.measure._form_fakes import configure_form_lookup
from tests.gui.app.measure.ui._artifact_snapshots import with_artifacts


def _parameters():
    return parameter_capture(
        {"R1": ResonatorSchema(kind="resonator", freq=6000.0)}, existing=123
    ).bind_aliases((AliasRow("resonator", "R1"),), {})


@pytest.fixture
def make_ctrl() -> Iterator[Callable[[], MagicMock]]:
    # Retain every authority through the test's owned view detach/close sequence.
    owners: list[ControlIdentity] = []

    def create() -> MagicMock:
        ctrl = MagicMock()
        authority = ControlIdentity()
        owners.append(authority)
        ctrl.control_identity = authority
        configure_form_lookup(ctrl)
        ctrl.get_left_panel_width.return_value = 500
        ctrl.get_tab_adapter_name.return_value = "fake"
        ctrl.get_adapter_guide.return_value = AdapterGuide(
            markdown="", resource_dir=None
        )
        ctrl.progress_control.attach_progress.return_value = lambda: None
        ctrl.progress_control.progress_bars.return_value = []
        parameters = _parameters()
        ctrl.read_context.return_value = parameters
        ctrl.get_session_env.return_value = SessionEnv(
            parameters=parameters, soc=None, soccfg=None
        )
        return ctrl

    yield create


def make_snapshot(tab_id, *, plots: Plots, analysis=AnalysisMode.FIT, post=False):
    from zcu_tools.gui.app.measure.services.ports import (
        AnalysisPaneSnapshot,
        PostAnalysisPaneSnapshot,
        RunPaneSnapshot,
        SavePaneSnapshot,
    )

    run_snap = RunPaneSnapshot(result=object(), source_path=None)
    plots.adopt("fit", Figure())
    plots.finish()
    analysis_snap = AnalysisPaneSnapshot(
        params=MagicMock(),
        result=object(),
        figures=plots,
        writeback_items=(),
        has_writeback_draft=False,
    )
    post_snap = PostAnalysisPaneSnapshot(
        params=None,
        result=None,
        figures=None,
        writeback_items=(),
        has_writeback_draft=False,
    )

    # params dataclass for analyze form
    @dataclasses.dataclass
    class P:
        thr: float = 0.5

    analysis_snap = dataclasses.replace(analysis_snap, params=P())
    snapshot = TabSnapshot(
        adapter_name="fake",
        cfg_input={},
        tab_id=tab_id,
        interaction=TabInteractionState(
            global_run_active=False,
            is_running=False,
            is_analyzing=False,
            has_context=True,
            has_active_context=True,
            has_soc=True,
            has_run_result=True,
            has_analyze_result=True,
            has_figure=True,
            has_post_analyze_result=False,
        ),
        capabilities=AdapterCapabilities(analysis=analysis, post_analysis=post),
        run=run_snap,
        analysis=analysis_snap,
        post_analysis=post_snap,
        save=SavePaneSnapshot(),
    )
    return with_artifacts(snapshot)


@pytest.fixture
def snapshots():
    owners: list[Plots] = []

    def create(tab_id, *, analysis=AnalysisMode.FIT, post=False):
        plots = Plots(NonPresentingHost())
        owners.append(plots)
        return make_snapshot(tab_id, plots=plots, analysis=analysis, post=post)

    try:
        yield create
    finally:
        for plots in owners:
            plots.finish()
            plots.release()


@pytest.fixture
def exp_tab_widget(qapp):
    from zcu_tools.gui.app.measure.ui.exp_tab_widget import ExpTabWidget

    return ExpTabWidget


def test_A1_Run_mounts_tree_with_visual_and_folding(
    qapp, exp_tab_widget, snapshots, make_ctrl
):
    """Run keeps its tree presentation, Reset, editing lock and busy state."""
    from zcu_tools.gui.app.measure.cfg_binding import MeasureCfgBindings
    from zcu_tools.gui.widgets.cfg import CfgFormWidget

    ctrl = make_ctrl()
    snap = snapshots("tab-1", analysis=AnalysisMode.FIT, post=False)
    tab = exp_tab_widget("tab-1", ctrl, snap.capabilities)
    w = CfgFormWidget(text_input_enhancer=None)
    try:
        tab.attach(snap, MagicMock())
        schema = CfgSchema(
            spec=CfgSectionSpec(fields={"reps": ScalarSpec(label="Reps", type=int)}),
            value=CfgSectionValue(fields={"reps": DirectValue(10)}),
        )
        draft = MeasureCfgBindings(make_ctrl()).new_draft(schema, parameters=None)
        w.attach(draft)
        tree = w.findChild(QTreeWidget)
        assert tree is not None
        assert tree.isHeaderHidden()
        assert tree.indentation() == 10
        assert tree.font().pixelSize() == 13
        assert tab.reset_btn.isEnabled() is True, ctrl.control_identity.status()
        assert isinstance(tab.reset_btn.toolTip(), str)
        busy_snap = snapshots("tab-1", analysis=AnalysisMode.FIT, post=False)
        assert busy_snap.interaction is not None
        busy_snap = dataclasses.replace(
            busy_snap,
            interaction=dataclasses.replace(busy_snap.interaction, is_running=True),
        )
        tab.update_interaction_state(busy_snap)
        assert tab.reset_btn.isEnabled() is False
    finally:
        w.detach()
        w.close()
        tab.detach()
        tab.close()
        qapp.processEvents()


def _click_header(qapp: QApplication, header: QWidget) -> None:
    center = header.rect().center()
    event = QMouseEvent(
        QEvent.Type.MouseButtonRelease,
        QPointF(center),
        QPointF(header.mapToGlobal(center)),
        Qt.MouseButton.LeftButton,
        Qt.MouseButton.NoButton,
        Qt.KeyboardModifier.NoModifier,
    )
    qapp.sendEvent(header, event)


def test_A2_Analysis_ledger_single_column_folding_and_fixed_bar(
    qapp, exp_tab_widget, snapshots, make_ctrl
):
    """Analysis uses a 13 px ledger, whole-header folding and inline Analyze."""
    ctrl = make_ctrl()
    snap = snapshots("tab-1", analysis=AnalysisMode.FIT, post=False)
    tab = exp_tab_widget("tab-1", ctrl, snap.capabilities)
    try:
        tab.attach(snap, MagicMock())
        assert tab.analyze_form.font().pixelSize() == 13
        header = next(
            header
            for header in tab.findChildren(QWidget, "ledgerHeader")
            if any(
                label.text() == "<b>Analysis parameters</b>"
                for label in header.findChildren(QLabel)
            )
        )
        section = header.parentWidget()
        if section is None:
            raise RuntimeError("Analysis header has no containing section")
        initially = tab.analyze_form.isVisibleTo(section)
        _click_header(qapp, header)
        assert tab.analyze_form.isVisibleTo(section) != initially
        _click_header(qapp, header)
        assert tab.analyze_form.isVisibleTo(section) == initially
        scroll = next(
            scroll
            for scroll in tab.findChildren(QScrollArea)
            if scroll.isAncestorOf(tab.analyze_form)
        )
        inner = scroll.widget()
        assert inner is not None
        layout = inner.layout()
        assert layout is not None
        assert tab.analyze_btn.parent() is not None
        assert inner.isAncestorOf(tab.analyze_btn), (
            "Analyze should be inside scroll area"
        )
        widgets = []
        for index in range(layout.count()):
            item = layout.itemAt(index)
            if item is not None and (widget := item.widget()) is not None:
                widgets.append(widget)
        idx_params = widgets.index(section)
        idx_analyze = widgets.index(tab.analyze_btn)
        idx_writeback = widgets.index(tab.writeback_section)
        assert idx_params < idx_analyze < idx_writeback, (
            f"order params {idx_params} analyze {idx_analyze} writeback {idx_writeback}"
        )
        assert scroll.findChild(type(tab.analyze_form)) is not None
        assert tab.analyze_btn.isEnabled() is True
    finally:
        tab.detach()
        tab.close()
        qapp.processEvents()


def test_A3_Writeback_items_show_current_proposed_and_edit(qapp, make_ctrl):
    """Native preview facts retain selection, editing and Apply interactions."""
    from zcu_tools.cfg_model import PulseReadoutCfg
    from zcu_tools.gui.app.measure.cfg_schemas import module_cfg_to_value
    from zcu_tools.gui.app.measure.services.writeback import WritebackValues
    from zcu_tools.gui.app.measure.ui.writeback_widget import WritebackWidget

    ctrl = make_ctrl()
    md_item = ParameterWriteback(
        target_name="$resonator.freq",
        description="Resonator freq",
        proposed_value=6100.0,
    )
    cfg = PulseReadoutCfg(
        gen_ch=0,
        nqz=1,
        gain=0.1,
        waveform=ConstWaveformCfg(length=1.0),
        ro_ch=0,
        ro_length=1.0,
        freq=0.0,
    )
    spec, value = module_cfg_to_value(cfg)
    ml_item = ModuleWriteback(
        target_name="readout_rf",
        description="Readout module",
        edit_schema=CfgSchema(spec, value),
    )
    ml_item2 = ModuleWriteback(
        target_name="ro_waveform", description="Waveform", edit_schema=None
    )
    md_item.session_id, ml_item.session_id, ml_item2.session_id = (
        "param-1",
        "ml-1",
        "wf-1",
    )
    values = {
        "param-1": WritebackValues(
            actual_path="R1.freq", current=6000.0, proposed=6100.0, unit=None
        ),
        "ml-1": WritebackValues(
            actual_path="R1.readout.readout_rf",
            current=None,
            proposed=cfg.to_dict(),
            unit=None,
        ),
        "wf-1": WritebackValues(
            actual_path="R1.readout.ro_waveform",
            current=None,
            proposed=cfg.to_dict(),
            unit=None,
        ),
    }
    ctrl.writeback_control.get_writeback_values_for_pane.side_effect = (
        lambda tab_id, pane, session_id=None: (
            values if session_id is None else {session_id: values[session_id]}
        )
    )
    widget = WritebackWidget(ctrl, tab_id="tab-1", pane="analysis")
    md_dialog = None
    try:
        widget.populate([md_item, ml_item, ml_item2], values)
        checks = widget.findChildren(QCheckBox)
        assert len(checks) == 3
        for target in ("$resonator.freq", "readout_rf", "ro_waveform"):
            assert any(target in check.text() for check in checks)
        assert "$resonator.freq" in checks[0].text()
        assert "readout_rf" in checks[1].text()
        cur = widget.findChildren(QLabel, "writebackCurrent")
        prop = widget.findChildren(QLabel, "writebackProposed")
        assert len(cur) == 3
        assert any("6000.0" in label.text() for label in cur)
        assert cur[1].text() == cur[2].text() == "—"
        assert len(prop) == 3
        assert any("6100.0" in label.text() for label in prop)
        assert prop[1].text() == prop[2].text() == "map[12]"
        edits = [
            button
            for button in widget.findChildren(QPushButton)
            if button.text() == "Edit"
        ]
        assert len(edits) == 2
        apply = next(
            button
            for button in widget.findChildren(QPushButton)
            if button.text() == "Apply Selected"
        )
        assert apply.text() == "Apply Selected"
        assert apply.isEnabled() is True
        for check in checks:
            check.setChecked(False)
        assert apply.isEnabled() is False
        checks[0].setChecked(True)
        assert apply.isEnabled() is True
        edits[0].click()
        md_dialog = next(
            dialog
            for dialog in QApplication.topLevelWidgets()
            if isinstance(dialog, QDialog)
            and dialog.isVisible()
            and dialog.windowTitle().startswith("Edit Value:")
        )
        assert md_dialog is not None
        assert md_dialog.findChild(QLabel, "writebackCurrentReadonly") is not None
    finally:
        if md_dialog is not None:
            md_dialog.close()
        widget.deleteLater()
        qapp.processEvents()


def test_writeback_baseline_captured_via_service(qapp, make_ctrl):
    """The service captures baseline summaries from a fixed ParameterContext."""
    from zcu_tools.cfg_model import PulseReadoutCfg
    from zcu_tools.gui.app.measure.cfg_schemas import module_cfg_to_value
    from zcu_tools.gui.app.measure.services.cfg_editor import CfgEditorService
    from zcu_tools.gui.app.measure.services.writeback import WritebackService
    from zcu_tools.gui.app.measure.state import State
    from zcu_tools.gui.event_bus import BaseEventBus

    from tests.gui.app.measure._cfg_fakes import PublishedHost

    parameters = _parameters()
    state = State(SessionEnv(parameters=parameters, soc=None, soccfg=None))
    ctrl = make_ctrl()
    cfg_editor = CfgEditorService(
        PublishedHost(state), ctrl, MagicMock(), MagicMock(), BaseEventBus()
    )
    svc = WritebackService(cfg_editor, ctrl)
    md_item = ParameterWriteback(
        target_name="$resonator.freq", description="freq", proposed_value=6100.0
    )
    md_item2 = ParameterWriteback(
        target_name="$resonator.ext.missing_key",
        description="missing",
        proposed_value=1,
    )
    spec, value = module_cfg_to_value(
        PulseReadoutCfg(
            gen_ch=0,
            nqz=1,
            gain=0.1,
            waveform=ConstWaveformCfg(length=1.0),
            ro_ch=0,
            ro_length=1.0,
            freq=0.0,
        )
    )
    module_item = ModuleWriteback(
        target_name="$resonator.readout.new_mod",
        description="new",
        edit_schema=CfgSchema(spec, value),
    )
    draft = svc.create_draft(
        [md_item, md_item2, module_item],
        parameters=parameters,
        experiment_name="test/writeback",
    )
    try:
        items = draft.preview()
        r_f_item = next(i for i in items if i.target_name == "$resonator.freq")
        missing = next(
            i for i in items if i.target_name == "$resonator.ext.missing_key"
        )
        mod = next(i for i in items if i.target_name == "$resonator.readout.new_mod")
        cur_rf, prop_rf = svc.get_summaries(draft, r_f_item.session_id)
        cur_missing, _ = svc.get_summaries(draft, missing.session_id)
        cur_mod, prop_mod = svc.get_summaries(draft, mod.session_id)
        assert cur_rf is not None and "6000" in cur_rf
        assert prop_rf is not None and "6100" in prop_rf
        assert cur_missing == "—"
        assert cur_mod == "— not present"
        assert prop_mod is not None and "create" in prop_mod
    finally:
        draft.teardown()
