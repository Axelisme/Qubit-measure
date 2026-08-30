"""Hardware-free fake-adapter production-path launch preparation (A6 non-claim).

Reproducible entry point for Orchestrator A6 observation: launch the shipped
measure-gui composition with a fake adapter (requires_soc=False) and verify the
approved Run tree and Analysis ledger are mounted without hardware.

Run with:
  PYTHONPATH=lib QT_QPA_PLATFORM=offscreen .venv/bin/python -m pytest tests/gui/app/main/test_hardware_free_fake_adapter_launch.py -xvs
"""

from __future__ import annotations

from unittest.mock import MagicMock

from qtpy.QtWidgets import QCheckBox, QLabel
from zcu_tools.gui.app.main.adapter import AdapterCapabilities, AnalysisMode
from zcu_tools.gui.app.main.services import PersistedStartup
from zcu_tools.gui.app.main.ui.exp_tab_widget import ExpTabWidget


def _ctrl():
    from zcu_tools.gui.app.main.services import PersistedStartup

    ctrl = MagicMock()
    ctrl.get_persisted_startup.return_value = PersistedStartup(left_panel_width=500)
    ctrl.get_tab_adapter_name.return_value = "fake/freq"
    ctrl.get_adapter_guide.return_value = {}
    ctrl.progress_control.attach_progress.return_value = lambda: None
    ctrl.progress_control.progress_bars.return_value = []
    ctrl.get_exp_context.return_value = MagicMock()
    ctrl.open_seeded_cfg_editor.return_value = ("editor-1", ())
    ctrl.get_cfg_editor_draft.return_value = MagicMock()
    return ctrl


def test_hardware_free_fake_shows_run_tree_and_analysis_ledger(qapp, monkeypatch):
    """Shipped composition path: fake/freq opens without SoC and shows validated layouts."""
    import zcu_tools.gui.app.main.ui.exp_tab_widget as mod

    orig = mod.ExpTabWidget._populate_cfg

    def stub(self, schema, ctrl):
        self._cfg_editor_id = "probe-editor"
        self.cfg_form.is_valid = lambda: True
        self.cfg_form.first_invalid_reason = lambda: None

    monkeypatch.setattr(mod.ExpTabWidget, "_populate_cfg", stub)
    orig_attach = mod.attach_existing_figure_to_container

    def mock_attach(fig, container):
        from qtpy.QtWidgets import QWidget

        w = QWidget()
        w.figure = fig  # type: ignore[attr-defined]
        container.attach_canvas(w)
        w.draw = lambda: None  # type: ignore[attr-defined]
        return w

    monkeypatch.setattr(mod, "attach_existing_figure_to_container", mock_attach)

    # Use the real FakeFreqAdapter capabilities (requires_soc=False)
    from zcu_tools.experiment.v2_gui.adapters.fake.freq import FakeFreqAdapter

    caps = FakeFreqAdapter.capabilities
    assert caps.requires_soc is False
    assert caps.analysis is not AnalysisMode.NONE

    ctrl = _ctrl()
    # Build minimal snapshot matching FakeFreqAdapter's capabilities
    from zcu_tools.gui.app.main.services.ports import (
        AnalysisPaneSnapshot,
        PathResourceSnapshot,
        PostAnalysisPaneSnapshot,
        RunPaneSnapshot,
        SavePaneSnapshot,
        TabPathsSnapshot,
        TabSnapshot,
    )
    from zcu_tools.gui.app.main.state import TabInteractionState
    from zcu_tools.gui.cfg import CfgSchema, CfgSectionSpec, CfgSectionValue

    snap = TabSnapshot(
        adapter_name="fake/freq",
        cfg_schema=CfgSchema(spec=CfgSectionSpec(), value=CfgSectionValue()),
        tab_id="tab-1",
        interaction=TabInteractionState(
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
            has_post_analyze_result=False,
        ),
        capabilities=caps,
        run=RunPaneSnapshot(result=None, source_path=None),
        analysis=AnalysisPaneSnapshot(
            params=FakeFreqAdapter().get_analyze_params(MagicMock(), MagicMock()),  # type: ignore
            result=None,
            figure=None,
            writeback_items=(),
            image_path=PathResourceSnapshot(override=None, path=None),
        ),
        post_analysis=PostAnalysisPaneSnapshot(
            params=None,
            result=None,
            figure=None,
            writeback_items=(),
            image_path=PathResourceSnapshot(override=None, path=None),
        ),
        save=SavePaneSnapshot(data_path=PathResourceSnapshot(override=None, path=None)),
        paths=TabPathsSnapshot(
            data=PathResourceSnapshot(override=None, path=None),
            analysis_image=PathResourceSnapshot(override=None, path=None),
            post_analysis_image=PathResourceSnapshot(override=None, path=None),
        ),
    )
    # Ensure params is dataclass for ledger
    try:
        tab = ExpTabWidget("tab-1", ctrl, caps)
        tab.attach(snap, MagicMock())
        # A1: Run tree
        from zcu_tools.gui.widgets.cfg import tree_structure

        assert tab.cfg_form._structure is tree_structure
        # A2: ledger + fixed bar
        from zcu_tools.gui.app.main.ui.exp_tab_widget import _LedgerSection

        assert isinstance(tab._analyze_section, _LedgerSection)
        assert hasattr(tab, "_analysis_action_bar")
        assert tab.analyze_btn.parent() is tab._analysis_action_bar
        # 13 px
        assert tab.analyze_form.font().pixelSize() == 13
        # Writeback ledger shows current/proposed columns (even when empty, widget exists)
        assert tab.writeback_widget is not None
        # Populate with fake writeback items and verify current/proposed appear
        from zcu_tools.gui.app.main.adapter import MetaDictWriteback

        item = MetaDictWriteback(
            target_name="r_f", description="freq", proposed_value=6100.0
        )
        item.session_id = "md-1"
        item.current_summary = "6000.0"
        item.proposed_summary = "6100.0"
        tab.writeback_widget.populate([item])
        cur = tab.writeback_widget.findChildren(QLabel, "writebackCurrent")
        prop = tab.writeback_widget.findChildren(QLabel, "writebackProposed")
        assert any("6000" in c.text() for c in cur)
        assert any("6100" in p.text() for p in prop)
        # Selection + Edit present
        checks = tab.writeback_widget.findChildren(QCheckBox)
        assert len(checks) == 1
        tab.detach()
    finally:
        monkeypatch.setattr(mod.ExpTabWidget, "_populate_cfg", orig)
        monkeypatch.setattr(mod, "attach_existing_figure_to_container", orig_attach)
