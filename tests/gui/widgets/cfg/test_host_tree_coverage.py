"""Measure tab cfg resource rendering through shared Qt widgets."""

from __future__ import annotations

from unittest.mock import MagicMock

from zcu_tools.gui.cfg import (
    CfgSchema,
    CfgSectionSpec,
    CfgSectionValue,
    DirectValue,
    ScalarSpec,
)
from zcu_tools.plotting.figures import FigureCollection
from zcu_tools.resources.context import MetaDict, ModuleLibrary

from tests.gui.app.measure.ui._artifact_snapshots import with_artifacts


def test_measure_tab_renders_cfg_resource_rows(qapp, monkeypatch):
    """The measure tab renders the attached cfg resource as editable rows."""
    import zcu_tools.gui.app.measure.ui.exp_tab_widget as mod
    from matplotlib.figure import Figure
    from zcu_tools.gui.app.measure.adapter import AdapterCapabilities, AnalysisMode
    from zcu_tools.gui.app.measure.services import TabSnapshot
    from zcu_tools.gui.app.measure.services.ports import (
        AnalysisPaneSnapshot,
        PostAnalysisPaneSnapshot,
        RunPaneSnapshot,
        SavePaneSnapshot,
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

    figures = FigureCollection()
    figures.adopt("fit", Figure())
    snap = TabSnapshot(
        adapter_name="fake",
        cfg_schema=schema,
        tab_id="t1",
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
        capabilities=caps,
        run=RunPaneSnapshot(result=object(), source_path=None),
        analysis=AnalysisPaneSnapshot(
            params=P(),
            result=object(),
            figures=figures,
            writeback_items=(),
            has_writeback_draft=False,
        ),
        post_analysis=PostAnalysisPaneSnapshot(
            params=None,
            result=None,
            figures=None,
            writeback_items=(),
            has_writeback_draft=False,
        ),
        save=SavePaneSnapshot(),
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


def test_resource_form_shows_device_name_preview_and_unbound_diagnostic(qapp) -> None:
    from qtpy.QtWidgets import QLabel, QLineEdit
    from zcu_tools.gui.app.measure.cfg_binding import MeasureCfgBindings
    from zcu_tools.gui.app.measure.state import State
    from zcu_tools.gui.cfg import EvalValue
    from zcu_tools.gui.cfg.resource import CfgResource
    from zcu_tools.gui.session.aliases import AliasRow
    from zcu_tools.gui.session.types import SessionEnv
    from zcu_tools.gui.widgets.cfg.resource_form import ResourceCfgFormWidget

    from tests.gui.app.measure._cfg_fakes import PublishedHost, parameter_capture

    base = parameter_capture()
    current = base.bind_aliases((AliasRow("flux_dev", "yoko1"),), {})
    state = State(SessionEnv(parameters=current, soc=None, soccfg=None))
    bindings = MeasureCfgBindings(PublishedHost(state))
    schema = CfgSchema(
        CfgSectionSpec(fields={"flux_dev": ScalarSpec("Flux", str, choices_source="devices")}),
        CfgSectionValue(fields={"flux_dev": EvalValue("$flux_dev")}),
    )
    resource = CfgResource(lambda: schema, resolution=lambda: bindings.snapshot((), captured_values={}, parameters=current))
    form = ResourceCfgFormWidget()
    try:
        form.attach(resource)
        assert form.is_valid()
        assert any(label.text() == '= "yoko1"' for label in form.findChildren(QLabel))
        line = next(line for line in form.findChildren(QLineEdit) if line.text() == "$flux_dev")
        line.setText("$unknown")
        assert not form.is_valid()
        assert "unknown" in (form.first_invalid_reason() or "")
        line.setText("$flux_dev")
        assert form.is_valid()
        current = base.bind_aliases((AliasRow("flux_dev", "yoko2"),), {})
        resource.refresh(resource.observe().ref.revision)
        assert any(label.text() == '= "yoko2"' for label in form.findChildren(QLabel))
    finally:
        form.detach()
        resource.revoke()
        form.deleteLater()
        qapp.processEvents()
