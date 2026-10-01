"""Host coverage: every existing CfgFormWidget host uses sole tree (A3)."""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.cfg import (
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    ScalarSpec,
)
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
from zcu_tools.gui.widgets.cfg.structure import TreeCfgWidget
from zcu_tools.plotting.figures import NamedFigures
from zcu_tools.resources.context import MetaDict, ModuleLibrary

from tests.gui.app.measure.ui._artifact_snapshots import with_artifacts

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _fake_ctrl():
    c = MagicMock()
    c.get_bus.return_value = EventBus()
    c.get_current_md.return_value = MetaDict()
    c.get_current_ml.return_value = MagicMock(modules={}, waveforms={})
    c.arb_waveforms.list_data_keys.return_value = []
    c.list_device_names.return_value = []
    return c


def test_measure_gui_run_uses_sole_tree(qapp, monkeypatch):
    """measure-gui Run renders through sole tree."""
    import zcu_tools.gui.app.measure.ui.exp_tab_widget as mod
    from matplotlib.figure import Figure
    from zcu_tools.gui.app.measure.adapter import AdapterCapabilities, AnalysisMode
    from zcu_tools.gui.app.measure.services import TabSnapshot
    from zcu_tools.gui.app.measure.services.ports import (
        AnalysisPaneSnapshot,
        PathResourceSnapshot,
        PostAnalysisPaneSnapshot,
        RunPaneSnapshot,
        SavePaneSnapshot,
        TabPathsSnapshot,
    )
    from zcu_tools.gui.app.measure.state import TabInteractionState
    from zcu_tools.gui.app.measure.ui.exp_tab_widget import ExpTabWidget

    orig_attach = mod.attach_existing_figure_to_container
    monkeypatch.setattr(
        mod, "attach_existing_figure_to_container", lambda fig, container: MagicMock()
    )

    ctrl = MagicMock()
    ctrl.get_left_panel_width.return_value = 500
    ctrl.get_tab_adapter_name.return_value = "fake"
    ctrl.get_adapter_guide.return_value = {}
    ctrl.progress_control.attach_progress.return_value = lambda: None
    ctrl.progress_control.progress_bars.return_value = []
    ctrl.get_session_env.return_value = MagicMock(md=MetaDict(), ml=ModuleLibrary())

    caps = AdapterCapabilities(analysis=AnalysisMode.FIT, post_analysis=False)
    spec = CfgSectionSpec(
        label="root", fields={"reps": ScalarSpec(label="Reps", type=int)}
    )
    schema = CfgSchema(
        spec=spec, value=CfgSectionValue(fields={"reps": DirectValue(10)})
    )
    # params dataclass
    import dataclasses

    @dataclasses.dataclass
    class P:
        thr: float = 0.5

    snap = TabSnapshot(
        adapter_name="fake",
        cfg_schema=schema,
        tab_id="t1",
        interaction=TabInteractionState(
            global_run_active=False,
            is_running=False,
            is_analyzing=False,
            is_saving_data=False,
            has_context=True,
            has_active_context=True,
            has_soc=True,
            has_run_result=True,
            has_analyze_result=True,
            has_figure=True,
            has_post_analyze_result=False,
        ),
        capabilities=caps,
        run=RunPaneSnapshot(result=object(), source_path=None),
        analysis=AnalysisPaneSnapshot(
            params=P(),
            result=object(),
            figures=NamedFigures({"fit": Figure()}),
            writeback_items=(),
            image_paths={"fit": PathResourceSnapshot(override=None, path="/tmp/a")},
            has_writeback_draft=False,
        ),
        post_analysis=PostAnalysisPaneSnapshot(
            params=None,
            result=None,
            figures=None,
            writeback_items=(),
            image_paths={},
            has_writeback_draft=False,
        ),
        save=SavePaneSnapshot(
            data_path=PathResourceSnapshot(override=None, path="/tmp/c")
        ),
        paths=TabPathsSnapshot(
            data=PathResourceSnapshot(override=None, path="/tmp/c"),
            analysis_images={"fit": PathResourceSnapshot(override=None, path="/tmp/a")},
            post_analysis_images={},
        ),
    )

    from qtpy.QtWidgets import QTreeWidget
    from zcu_tools.gui.widgets.cfg.resource_form import ResourceCfgFormWidget

    from tests.gui.app.measure._cfg_fakes import make_cfg

    cfg = make_cfg(schema)
    ctrl.cfg_resources.lookup.return_value = cfg
    tab = ExpTabWidget("t1", ctrl, caps)
    tab.attach(with_artifacts(snap), MagicMock())
    assert isinstance(tab.cfg_form, ResourceCfgFormWidget)
    tree = tab.cfg_form.findChild(QTreeWidget)
    assert tree is not None and tree.topLevelItemCount() == 1
    item = tree.topLevelItem(0)
    assert item is not None and item.text(0) == "Reps"
    tab.detach()
    monkeypatch.setattr(mod, "attach_existing_figure_to_container", orig_attach)


def test_autofluxdep_default_and_generation_use_sole_tree(qapp):
    """autofluxdep Default cfg and Generation overrides both use sole tree."""
    from zcu_tools.gui.app.autofluxdep.app import build_core
    from zcu_tools.gui.app.autofluxdep.ui.node_cfg_form import NodeCfgForm

    ctrl = build_core()
    try:
        node = ctrl.add_node_by_type("qubit_freq")
        idx = ctrl.state.nodes.index(node)
        form = NodeCfgForm(ctrl, node, idx)
        try:
            assert isinstance(form._default_form._root_widget, TreeCfgWidget)
            assert form._default_form._root_widget is not None
            # Generation overrides should also be tree when present
            if form._generation_form is not None:
                assert isinstance(form._generation_form._root_widget, TreeCfgWidget)
        finally:
            form.teardown()
    finally:
        ctrl._background_svc.quiesce()


def test_writeback_edit_uses_sole_tree(qapp, monkeypatch):
    """writeback module/waveform Edit dialog CfgFormWidget is sole tree."""
    from zcu_tools.gui.app.measure.cfg_binding import MeasureCfgBindings
    from zcu_tools.gui.cfg import (
        CfgSectionSpec,
        CfgSectionValue,
        DirectValue,
        ScalarSpec,
    )
    from zcu_tools.gui.widgets.cfg import CfgFormWidget

    ctrl = _fake_ctrl()
    # Create a realistic schema for edit
    inner = CfgSectionSpec(
        label="Inner", fields={"gain": ScalarSpec(label="Gain", type=float)}
    )
    schema = CfgSchema(
        spec=inner, value=CfgSectionValue(fields={"gain": DirectValue(0.5)})
    )
    draft = MeasureCfgBindings(ctrl).new_draft(schema)

    # Mock controller to return this draft for writeback edit
    ctrl.get_writeback_item_draft_for_pane = MagicMock(return_value=draft)
    ctrl.get_session_env.return_value = MagicMock(md=MetaDict(), ml=MagicMock())
    # We need to directly test that WritebackWidget creates a CfgFormWidget that is tree
    # The edit dialog creates CfgFormWidget internally; we verify a standalone CfgFormWidget used there is tree
    w = CfgFormWidget()
    w.attach(draft)
    assert isinstance(w._root_widget, TreeCfgWidget)
    w.detach()
    draft.close()


def test_module_library_cfg_form_uses_sole_tree(qapp, monkeypatch):
    """ModuleLibrary cfg forms use the shared tree widget."""
    from zcu_tools.gui.app.measure.cfg_binding import MeasureCfgBindings
    from zcu_tools.gui.cfg import (
        CfgSectionSpec,
        CfgSectionValue,
        DirectValue,
        ScalarSpec,
    )
    from zcu_tools.gui.widgets.cfg import CfgFormWidget

    ctrl = _fake_ctrl()
    schema = CfgSchema(
        spec=CfgSectionSpec(fields={"gain": ScalarSpec(label="Gain", type=float)}),
        value=CfgSectionValue(fields={"gain": DirectValue(1.0)}),
    )
    draft = MeasureCfgBindings(ctrl).new_draft(schema)
    w = CfgFormWidget()
    w.attach(draft)
    assert isinstance(w._root_widget, TreeCfgWidget)
    w.detach()
    draft.close()
    # The embedded Inspect and writeback editors use this same sole-tree form;
    # the widget no longer accepts a separate structure parameter.
    with pytest.raises(TypeError):
        CfgFormWidget(structure=object())  # type: ignore[call-arg]
