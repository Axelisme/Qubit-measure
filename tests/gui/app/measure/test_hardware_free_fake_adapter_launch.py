"""Hardware-free Form adapter Run and native writeback ledger presentation."""

from __future__ import annotations

from collections.abc import Sequence

import pytest
from qtpy.QtWidgets import QApplication, QCheckBox, QLabel, QScrollArea
from zcu_tools.experiment.context import RunContext
from zcu_tools.gui.app.measure.adapter import (
    ParameterWriteback,
    RunRequest,
    SessionEnv,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.app.measure.controller import Controller
from zcu_tools.gui.app.measure.registry import Registry
from zcu_tools.gui.app.measure.services import create_persistence_caretaker
from zcu_tools.gui.app.measure.state import State
from zcu_tools.gui.app.measure.ui.exp_tab_widget import ExpTabWidget
from zcu_tools.gui.app.measure.ui.main_window import MainWindow
from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_types import (
    CommandOrigin,
    EntryRequest,
    PointCreateRequest,
)
from zcu_tools.resources.entry.registry import component_registry

from tests.gui._qt_waits import qt_until
from tests.gui.app.measure._components import ResonatorSchema
from tests.gui.app.measure._synthetic import (
    PairAdapter,
    SampleAnalysis,
    SampleCfg,
    SampleRun,
)
from tests.gui.app.measure._templates import TEMPLATES


class LedgerAdapter(PairAdapter):
    """Supply a fixed native proposal over deterministic sample acquisition."""

    def run(
        self,
        req: RunRequest,
        cfg: SampleCfg,
        *,
        context: RunContext,
    ) -> SampleRun:
        return super().run(req, cfg, context=context)

    def get_writeback_items(
        self, req: WritebackRequest[SampleRun, SampleAnalysis]
    ) -> Sequence[WritebackItem]:
        return [
            ParameterWriteback(
                target_name="R1.freq", description="freq", proposed_value=6100.0
            )
        ]


@pytest.fixture
def hw_fixture(qapp, tmp_path):
    """Compose real Qt/domain owners over an isolated entry, without hardware."""
    before = component_registry.kinds()
    added = "resonator" not in before
    if added:
        component_registry.register("resonator", ResonatorSchema)
    registry = Registry(templates=TEMPLATES)
    registry.register("fake/freq", LedgerAdapter)
    state, bus = (
        State(SessionEnv(parameters=None, soc=None, soccfg=None)),
        BaseEventBus(),
    )
    ctrl = Controller(
        state=state,
        registry=registry,
        view=None,
        bus=bus,
        project_root=str(tmp_path),
    )
    window = MainWindow(ctrl, templates=lambda: {})
    ctrl.add_view(window)
    caretaker = create_persistence_caretaker(ctrl, cache_dir=tmp_path / "cache")
    ctrl.attach_caretaker(caretaker)
    ctrl.setup_control.new_entry(EntryRequest("lab"))
    ctrl.setup_control.new_point(PointCreateRequest("ctx001", "", CommandOrigin("gui")))
    ctrl.context_control.add_component(
        "R1",
        "resonator",
        {
            "freq": 6000.0,
            "kappa": 1.0,
            "wiring": {"ch": 0, "ro_ch": 0},
            "ext": {"time_of_fly": 0.5},
        },
    )
    ctrl.context_control.set_alias_rows((AliasRow("resonator", "R1"),))
    window.show()
    qapp.processEvents()
    try:
        yield ctrl, window
    finally:
        ctrl.close()
        window.deleteLater()
        qapp.processEvents()
        if added:
            component_registry.unregister("resonator")
        assert component_registry.kinds() == before


def test_hardware_free_fake_shows_run_tree_and_analysis_ledger(hw_fixture):
    """Public Form tab Run/analysis renders a native 6100 MHz writeback ledger."""
    ctrl, window = hw_fixture
    assert "fake/freq" in ctrl.get_adapter_names()
    assert ctrl.get_session_env().soc is None
    tab_id = ctrl.new_tab("fake/freq")
    QApplication.processEvents()
    (tab,) = window.findChildren(ExpTabWidget)
    assert tab.analyze_form.font().pixelSize() == 13

    def is_descendant(widget, ancestor):
        current = widget
        while current is not None:
            if current is ancestor:
                return True
            current = current.parent()
        return False

    scroll = next(
        scroll
        for scroll in tab.findChildren(QScrollArea)
        if is_descendant(tab.analyze_btn, scroll)
    )
    inner = scroll.widget()
    assert inner is not None
    assert is_descendant(tab.analyze_btn, inner)
    layout = inner.layout()
    assert layout is not None
    widgets = [
        layout.itemAt(i).widget()
        for i in range(layout.count())
        if layout.itemAt(i).widget() is not None
    ]
    idx_params = next(
        i for i, widget in enumerate(widgets) if is_descendant(tab.analyze_form, widget)
    )
    idx_analyze = next(
        i for i, widget in enumerate(widgets) if is_descendant(tab.analyze_btn, widget)
    )
    idx_writeback = widgets.index(tab.writeback_section)
    assert idx_params < idx_analyze < idx_writeback

    cfg = ctrl.cfg_resources.lookup(tab_id)
    cfg.write({("rounds",): 1})
    ctrl.start_run(tab_id)

    qt_until(lambda: ctrl.get_tab_snapshot(tab_id).run.result is not None)
    qt_until(lambda: not ctrl.get_tab_snapshot(tab_id).interaction.is_running)
    ctrl.analyze(tab_id, tab.analyze_form.read_params())
    qt_until(lambda: ctrl.get_tab_snapshot(tab_id).analysis.has_writeback_draft)
    items = ctrl.get_tab_snapshot(tab_id).analysis.writeback_items
    assert len(items) == 1
    current, proposed = ctrl.get_writeback_summaries_for_pane(tab_id, "analysis")[
        items[0].session_id
    ]
    assert current is not None and proposed is not None
    window.refresh_tab_writeback(tab_id, ctrl.get_tab_snapshot(tab_id))
    QApplication.processEvents()
    current_labels = tab.writeback_widget.findChildren(QLabel, "writebackCurrent")
    proposed_labels = tab.writeback_widget.findChildren(QLabel, "writebackProposed")
    assert len(current_labels) >= 1
    assert len(proposed_labels) >= 1
    assert any(current in label.text() for label in current_labels)
    assert any(proposed in label.text() for label in proposed_labels)
    assert len(tab.writeback_widget.findChildren(QCheckBox)) >= 1
    ctrl.apply_writeback_for_pane(tab_id, "analysis")
    assert ctrl.read_context().value("R1.freq") == 6100.0
    ctrl.close_tab(tab_id)
    QApplication.processEvents()
