"""Hardware-free fake-adapter production-path launch preparation (A6 non-claim).

Reproducible entry point for Orchestrator A6 observation: launch the shipped
measure-gui composition with a fake adapter (requires_soc=False) via the normal
Controller/MainWindow composition path and verify the approved Run tree and
Analysis ledger are mounted without hardware.

Run with:
  PYTHONPATH=lib QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/gui/app/main/test_hardware_free_fake_adapter_launch.py -xvs
"""

from __future__ import annotations

import pytest
from qtpy.QtWidgets import QApplication, QCheckBox, QLabel
from zcu_tools.gui.app.main.adapter import AnalysisMode


@pytest.fixture
def hw_fixture(qapp, tmp_path):
    """Real Controller + MainWindow via shipped composition, no hardware."""
    from unittest.mock import MagicMock

    from zcu_tools.experiment.v2_gui.registry import register_all, register_all_roles
    from zcu_tools.gui.app.main.app import _build_window, _make_empty_ctx
    from zcu_tools.gui.app.main.registry import Registry
    from zcu_tools.gui.app.main.role_catalog import RoleCatalog
    from zcu_tools.gui.app.main.state import State
    from zcu_tools.gui.session.services.io_manager import IOManager

    state = State(_make_empty_ctx())  # soc=None, no hardware
    registry = Registry()
    register_all(registry)
    role_catalog = RoleCatalog()
    register_all_roles(role_catalog)
    io_manager = IOManager()
    ctrl, window = _build_window(
        state, registry, role_catalog, io_manager, project_root=str(tmp_path)
    )
    # Attach a no-op caretaker so MainWindow close does not assert (production
    # attaches it in MeasureGuiBehavior.before_show, which we do not run here).
    ctrl._caretaker = MagicMock()  # type: ignore[attr-defined]
    ctrl._caretaker.flush = MagicMock()  # type: ignore[attr-defined]
    window.show()
    QApplication.processEvents()
    QApplication.processEvents()
    yield ctrl, window
    # Teardown: close window and quiesce background
    try:
        # Avoid triggering persist path that expects a real caretaker; just delete.
        window.deleteLater()
        QApplication.processEvents()
        QApplication.processEvents()
    except Exception:
        pass
    try:
        ctrl._background_svc.quiesce()  # type: ignore[attr-defined]
    except Exception:
        try:
            ctrl._app_services.background.quiesce()  # type: ignore[attr-defined]
        except Exception:
            pass


def test_hardware_free_fake_shows_run_tree_and_analysis_ledger(hw_fixture):
    """Shipped composition path: fake/freq opens without SoC and shows validated layouts."""
    ctrl, window = hw_fixture

    # Verify fake/freq is available and hardware-free
    from zcu_tools.experiment.v2_gui.adapters.fake.freq import FakeFreqAdapter

    caps = FakeFreqAdapter.capabilities
    assert caps.requires_soc is False
    assert caps.analysis is not AnalysisMode.NONE
    assert "fake/freq" in ctrl.get_adapter_names()

    # Open tab via normal Controller path (not direct ExpTabWidget instantiation)
    # This goes through the production TabAddedPayload -> MainWindow.add_tab_widget path,
    # with MainWindow as the RenderHost (via ctrl.add_view(window) done in _build_window).
    assert ctrl.get_exp_context().soc is None
    tab_id = ctrl.new_tab("fake/freq")
    QApplication.processEvents()
    QApplication.processEvents()

    # MainWindow should have created the ExpTabWidget via TabAddedPayload
    assert tab_id in window._tab_widgets
    tab = window._tab_widgets[tab_id]
    # Controller's RenderHost should be the MainWindow, not a MagicMock
    assert ctrl._render_host is window  # type: ignore[attr-defined]
    assert window in ctrl._diag_sinks  # type: ignore[attr-defined]

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

    # A2/A4: ledger with Analyze between params and writeback (not fixed bar)
    from zcu_tools.gui.app.main.ui.exp_tab_widget import _LedgerSection
    from qtpy.QtWidgets import QScrollArea

    assert isinstance(tab._analyze_section, _LedgerSection)
    assert not hasattr(tab, "_analysis_action_bar"), "fixed action bar should be removed for A4"
    assert tab.analyze_form.font().pixelSize() == 13
    # Verify Analyze is inside scroll area between params and writeback
    scroll = tab._analysis_panel.findChild(QScrollArea)
    assert scroll is not None
    inner = scroll.widget()
    assert inner is not None
    # Check Analyze is descendant of inner (scroll content) not fixed bar
    def is_descendant(widget, ancestor):
        cur = widget
        while cur is not None:
            if cur is ancestor:
                return True
            cur = cur.parent()
        return False
    assert is_descendant(tab.analyze_btn, inner)
    # Verify ordering: params < analyze < writeback
    layout = inner.layout()
    assert layout is not None
    widgets = [layout.itemAt(i).widget() for i in range(layout.count()) if layout.itemAt(i).widget() is not None]
    idx_params = widgets.index(tab._analyze_section)
    # Find container holding analyze_btn
    idx_analyze = None
    for idx, w in enumerate(widgets):
        if is_descendant(tab.analyze_btn, w):
            idx_analyze = idx
            break
    assert idx_analyze is not None
    idx_writeback = widgets.index(tab.writeback_section)
    assert idx_params < idx_analyze < idx_writeback
    # Post-Analysis should remain baseline _CollapsibleSection (not ledger)
    # fake/freq does not have post_analysis, so post widgets should not exist
    assert not hasattr(tab, "post_writeback_widget") or tab._has_post is False

    # Populate writeback via service projection (S2) and verify ledger columns
    from zcu_tools.gui.app.main.adapter import MetaDictWriteback

    # Use the controller's writeback service to create a draft so summaries are
    # captured in the service-owned model (not on the public WritebackItem).
    svc = ctrl._writeback_svc  # type: ignore[attr-defined]

    md_item = MetaDictWriteback(
        target_name="r_f", description="freq", proposed_value=6100.0
    )
    draft = svc.create_draft([md_item])
    items = svc.preview_draft(draft)
    assert len(items) == 1
    session_id = items[0].session_id
    cur, prop = svc.get_summaries(draft, session_id)
    assert cur is not None
    assert prop is not None
    # Now set this draft as the tab's analysis draft (simulate analyze completion)
    ctrl._state.update_tab_analyze(  # type: ignore[attr-defined]
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
    ctrl.close_tab(tab_id)
    QApplication.processEvents()
