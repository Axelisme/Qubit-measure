from __future__ import annotations

from unittest.mock import MagicMock

from qtpy.QtWidgets import QPushButton
from zcu_tools.experiment.v2_gui.adapters.fake.freq import (
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
    items = list(
        adapter.get_writeback_items(
            WritebackRequest(run_result=result, analyze_result=analyze_result, ctx=ctx)
        )
    )
    # The service stamps session_ids at compute time; do it here so the widget's
    # per-id checkbox map is unambiguous.
    for i, item in enumerate(items):
        item.session_id = f"id-{i}"

    widget = WritebackWidget(MagicMock())
    widget.populate(items)
    selected = [it for it in items if it.selected]
    edit_buttons = [w for w in widget.findChildren(QPushButton) if w.text() == "Edit"]

    # The one-tone freq fit proposes only r_f / rf_w (two MetaDict items, both
    # editable) — no readout module / waveform writeback.
    assert len(selected) == len(items)  # all selected by default
    assert {it.target_name for it in items} == {"r_f", "rf_w"}
    assert len(edit_buttons) == 2
