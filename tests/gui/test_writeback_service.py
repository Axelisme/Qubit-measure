from __future__ import annotations

from unittest.mock import MagicMock

from zcu_tools.experiment.v2_gui.adapters.fake import FakeAdapter
from zcu_tools.experiment.v2_gui.adapters.onetone.fakefreq import FakeFreqAdapter
from zcu_tools.gui.adapter import (
    AnalyzeRequest,
    ExpContext,
    ModuleWriteback,
    RunRequest,
)
from zcu_tools.gui.event_bus import GuiEvent
from zcu_tools.gui.services.writeback import WritebackService
from zcu_tools.gui.state import State
from zcu_tools.meta_tool import MetaDict, ModuleLibrary


def _make_ctx() -> ExpContext:
    return ExpContext(
        md=MetaDict(),
        ml=ModuleLibrary(),
        soc=None,
        soccfg=None,
        res_name="test_res",
        result_dir="/tmp/zcu_result",
        database_path="/tmp/zcu_db/test_chip/test_qubit",
        active_label="ctx001",
    )


def _default_analyze_params(adapter, result, ctx) -> dict[str, object]:
    params = adapter.get_analyze_params(result, ctx)
    return {param.key: param.default for param in params}


def test_writeback_service_applies_md_items():
    ctx = _make_ctx()
    state = State(ctx)
    bus = MagicMock()
    adapter = FakeAdapter()
    tab_id = "fake"

    state.add_tab(tab_id, adapter, ctx)
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
    state.update_tab_result(tab_id, result)
    state.update_tab_analyze(tab_id, analyze_result, analyze_result.figure)

    svc = WritebackService(state, bus)
    items = svc.get_tab_writeback_items(tab_id)
    applied = svc.apply_tab_writeback_items(tab_id, items)

    assert applied == ["fake_peak"]
    assert ctx.md.fake_peak is not None
    bus.emit.assert_any_call(GuiEvent.MD_CHANGED, ctx.md)

    items_after = svc.get_tab_writeback_items(tab_id)
    assert len(items_after) == 1
    assert items_after[0].selected is False


def test_writeback_service_applies_module_and_waveform_items():
    ctx = _make_ctx()
    state = State(ctx)
    bus = MagicMock()
    adapter = FakeFreqAdapter(fast_mode=True)
    tab_id = "fakefreq"

    state.add_tab(tab_id, adapter, ctx)
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
    state.update_tab_result(tab_id, result)
    state.update_tab_analyze(tab_id, analyze_result, analyze_result.figure)

    svc = WritebackService(state, bus)
    items = svc.get_tab_writeback_items(tab_id)
    for item in items:
        item.selected = item.key in {"readout_rf", "ro_waveform"}
        if isinstance(item, ModuleWriteback):
            item.edited_schema = item.edit_schema

    applied = svc.apply_tab_writeback_items(tab_id, items)

    assert applied == ["readout_rf", "ro_waveform"]
    assert "readout_rf" in ctx.ml.modules
    assert "ro_waveform" in ctx.ml.waveforms
    bus.emit.assert_any_call(GuiEvent.INSPECT_CHANGED, ctx.md)

    items_after = svc.get_tab_writeback_items(tab_id)
    selected_keys = {item.key for item in items_after if item.selected}
    assert "readout_rf" not in selected_keys
    assert "ro_waveform" not in selected_keys
