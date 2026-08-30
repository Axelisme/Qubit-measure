"""Hardware-free fake-adapter production-path launch preparation (A6 non-claim).

Reproducible entry point for Orchestrator A6 observation: launch the shipped
measure-gui composition with a fake adapter (requires_soc=False) via the normal
Controller/MainWindow composition path and verify the approved Run tree and
Analysis ledger are mounted without hardware.

Run with:
  PYTHONPATH=lib QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/gui/app/main/test_hardware_free_fake_adapter_launch.py -xvs
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from qtpy.QtWidgets import QApplication, QLabel, QCheckBox
from zcu_tools.gui.app.main.adapter import AnalysisMode


@pytest.fixture
def hw_fixture(qapp, tmp_path):
    """Real Controller + MainWindow via shipped composition, no hardware."""
    from tests.gui.test_controller import ControllerFixture
    from zcu_tools.gui.app.main.ui.main_window import MainWindow

    fixture = ControllerFixture(cache_dir=tmp_path)
    window = MainWindow(fixture.ctrl)
    # ControllerFixture's view is a MagicMock; add the real MainWindow as well.
    # MainWindow already bound its event coordinator to fixture.ctrl's bus.
    yield fixture, window
    # Teardown: close window and quiesce background
    try:
        window.close()
        window.deleteLater()
        QApplication.processEvents()
    except Exception:
        pass
    fixture.quiesce()


def test_hardware_free_fake_shows_run_tree_and_analysis_ledger(hw_fixture):
    """Shipped composition path: fake/freq opens without SoC and shows validated layouts."""
    fixture, window = hw_fixture
    ctrl = fixture.ctrl

    # Verify fake/freq is available and hardware-free
    from zcu_tools.experiment.v2_gui.adapters.fake.freq import FakeFreqAdapter

    caps = FakeFreqAdapter.capabilities
    assert caps.requires_soc is False
    assert caps.analysis is not AnalysisMode.NONE
    assert "fake/freq" in ctrl.get_adapter_names()

    # Open tab via normal Controller path (not direct ExpTabWidget instantiation)
    tab_id = ctrl.new_tab("fake/freq")
    QApplication.processEvents()
    QApplication.processEvents()

    # MainWindow should have created the ExpTabWidget via TabAddedPayload
    assert tab_id in window._tab_widgets
    tab = window._tab_widgets[tab_id]

    # A1: Run tree selected (shared adapter)
    from zcu_tools.gui.widgets.cfg import tree_structure

    assert tab.cfg_form._structure is tree_structure

    # Verify tree visuals via direct CfgFormWidget check (shared tree tests cover depth etc.)
    from zcu_tools.gui.widgets.cfg.structure import TREE_DEPTH_COLORS

    assert len(TREE_DEPTH_COLORS) == 5
    # Tree should have been attached via real CfgDraft (not stubbed)
    from zcu_tools.gui.widgets.cfg.structure import TreeCfgWidget

    assert tab.cfg_form._root_widget is not None
    assert isinstance(tab.cfg_form._root_widget, TreeCfgWidget)
    tree = tab.cfg_form._root_widget._tree
    assert tree.isHeaderHidden()
    assert tree.indentation() == 10
    assert tree.font().pixelSize() == 13

    # A2: ledger + fixed bar (Primary Analysis)
    from zcu_tools.gui.app.main.ui.exp_tab_widget import _LedgerSection

    assert isinstance(tab._analyze_section, _LedgerSection)
    assert hasattr(tab, "_analysis_action_bar")
    assert tab.analyze_btn.parent() is tab._analysis_action_bar
    assert tab.analyze_form.font().pixelSize() == 13
    # Post-Analysis should remain baseline _CollapsibleSection (not ledger)
    from zcu_tools.gui.widgets.cfg.fields import _CollapsibleSection

    # fake/freq does not have post_analysis, so post widgets should not exist
    assert not hasattr(tab, "post_writeback_widget") or tab._has_post is False

    # Populate writeback via service projection (S2) and verify ledger columns
    # Trigger an analyze to get a real draft with baseline summaries, then check widget
    # For hardware-free we can run a fake analyze via controller
    # Use the adapter's analyze path directly via service to avoid hardware
    # Instead, manually create a draft via service to test display
    from zcu_tools.gui.app.main.adapter import MetaDictWriteback
    from zcu_tools.gui.app.main.services.writeback import WritebackService

    # Create a draft with a real context snapshot to get summaries
    # Use the controller's writeback service
    writeback_svc = (
        ctrl._writeback_svc if hasattr(ctrl, "_writeback_svc") else ctrl._writeback
    )
    # The WritebackService is available as ctrl._writeback (check fixture)
    # In ControllerFixture, ctrl._writeback is the service
    svc = getattr(ctrl, "_writeback", None) or getattr(ctrl, "_writeback_svc", None)
    # Fallback: get via app_services
    if svc is None:
        svc = ctrl._writeback  # type: ignore[attr-defined]

    # Create a simple MetaDict item and get summaries via service
    md_item = MetaDictWriteback(
        target_name="r_f", description="freq", proposed_value=6100.0
    )
    # Use the service to create draft so summaries are captured in service-owned model
    draft = svc.create_draft([md_item])
    # The draft's item should be available via preview, but summaries are in service
    items = svc.preview_draft(draft)
    assert len(items) == 1
    session_id = items[0].session_id
    cur, prop = svc.get_summaries(draft, session_id)
    assert cur is not None
    assert prop is not None
    # Now set this draft as the tab's analysis draft (simulate analyze completion)
    # Use State to set writeback draft
    from zcu_tools.gui.app.main.state import State

    # Directly update tab's analysis pane with the draft (simulate service record)
    # Use controller's state
    ctrl._state.update_tab_analyze(
        tab_id,
        object(),
        None,
        writeback_draft=draft,
        analyze_params_instance=tab.analyze_form.read_params()
        if tab.has_analyze_params()
        else object(),
    )
    # Refresh the tab's writeback widget via MainWindow path
    snapshot = ctrl.get_tab_snapshot(tab_id)
    window.refresh_tab_writeback(tab_id, snapshot)
    QApplication.processEvents()
    # Now widget should show current/proposed via service projection
    cur_labels = tab.writeback_widget.findChildren(QLabel, "writebackCurrent")
    prop_labels = tab.writeback_widget.findChildren(QLabel, "writebackProposed")
    assert len(cur_labels) >= 1
    assert len(prop_labels) >= 1
    assert any(cur in lbl.text() for lbl in cur_labels)  # type: ignore[arg-type]
    assert any(prop in lbl.text() for lbl in prop_labels)  # type: ignore[arg-type]
    checks = tab.writeback_widget.findChildren(QCheckBox)
    assert len(checks) >= 1

    # Cleanup: teardown draft via state removal will be handled by fixture quiesce
    # Remove tab to avoid leaking
    ctrl.close_tab(tab_id)
    QApplication.processEvents()
