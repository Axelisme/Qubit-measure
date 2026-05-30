from __future__ import annotations

from unittest.mock import MagicMock

from qtpy.QtWidgets import QPushButton
from zcu_tools.experiment.v2_gui.adapters.onetone.fakefreq import (
    FakeFreqAdapter,
    FakeFreqAnalyzeParams,
    FakeFreqRunResult,
)
from zcu_tools.gui.adapter import (
    AnalyzeRequest,
    ExpContext,
    RunRequest,
    WritebackRequest,
)
from zcu_tools.gui.event_bus import EventBus
from zcu_tools.gui.ui.cfg_form import CfgFormWidget
from zcu_tools.gui.ui.writeback_widget import WritebackWidget
from zcu_tools.meta_tool import MetaDict, ModuleLibrary


def _make_ctx() -> ExpContext:
    return ExpContext(
        md=MetaDict(),
        ml=ModuleLibrary(),
        soc=None,
        soccfg=None,
    )


def _default_analyze_params(
    adapter: FakeFreqAdapter, result: FakeFreqRunResult, ctx: ExpContext
) -> FakeFreqAnalyzeParams:
    return adapter.get_analyze_params(result, ctx)


def test_writeback_widget_lists_items_and_edit_buttons(qapp):
    ctx = _make_ctx()
    adapter = FakeFreqAdapter(fast_mode=True)
    schema = adapter.make_default_cfg(ctx)
    result = adapter.run(
        RunRequest(md=ctx.md, ml=ctx.ml, soc=ctx.soc, soccfg=ctx.soccfg), schema
    )
    analyze_result = adapter.analyze(
        AnalyzeRequest(
            run_result=result,
            analyze_params=_default_analyze_params(adapter, result, ctx),
            md=ctx.md,
            ml=ctx.ml,
            predictor=ctx.predictor,
        )
    )
    items = adapter.get_writeback_items(
        WritebackRequest(run_result=result, analyze_result=analyze_result, ctx=ctx)
    )

    widget = WritebackWidget(MagicMock())
    widget.populate(items)
    selected = widget.get_selected_items()
    edit_buttons = [w for w in widget.findChildren(QPushButton) if w.text() == "Edit"]

    assert len(selected) == len(items)
    assert len(edit_buttons) >= 4


def test_readout_writeback_drag_schema_is_initially_valid(qapp):
    ctx = _make_ctx()
    ctrl = MagicMock()
    ctrl.get_bus.return_value = EventBus()
    ctrl.get_current_md.return_value = ctx.md
    ctrl.get_current_ml.return_value = ctx.ml
    ctrl.is_running.return_value = False
    ctrl.has_soc.return_value = False

    adapter = FakeFreqAdapter(fast_mode=True)
    schema = adapter.make_default_cfg(ctx)
    result = adapter.run(
        RunRequest(md=ctx.md, ml=ctx.ml, soc=ctx.soc, soccfg=ctx.soccfg), schema
    )
    analyze_result = adapter.analyze(
        AnalyzeRequest(
            run_result=result,
            analyze_params=_default_analyze_params(adapter, result, ctx),
            md=ctx.md,
            ml=ctx.ml,
            predictor=ctx.predictor,
        )
    )
    items = adapter.get_writeback_items(
        WritebackRequest(run_result=result, analyze_result=analyze_result, ctx=ctx)
    )
    from zcu_tools.gui.adapter import ModuleWriteback

    readout_item = next(
        item
        for item in items
        if isinstance(item, ModuleWriteback) and item.key == "readout_rf"
    )

    from zcu_tools.gui.live_model import LiveModelEnv, SectionLiveField

    schema_r = readout_item.edit_schema
    assert schema_r is not None
    model = SectionLiveField(schema_r.spec, LiveModelEnv(ctrl=ctrl), schema_r.value)
    form = CfgFormWidget()
    form.attach(model)

    assert form.is_valid() is True
