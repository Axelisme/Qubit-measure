from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from zcu_tools.experiment.v2.twotone.reset.single_tone.freq import (
    FreqCfg,
    FreqResult,
)
from zcu_tools.experiment.v2.twotone.reset.single_tone.length import (
    LengthCfg,
    LengthResult,
)
from zcu_tools.experiment.v2_gui.adapters.twotone import (
    SingleToneFreqAdapter,
    SingleToneLengthAdapter,
)
from zcu_tools.gui.app.main.adapter import (
    CfgSchema,
    MetaDictWriteback,
    RunRequest,
    WritebackRequest,
)
from zcu_tools.meta_tool import MetaDict
from zcu_tools.program.v2 import SweepCfg


def _make_ml() -> MagicMock:
    ml = MagicMock()
    ml.modules = {}
    ml.waveforms = {}
    ml.make_cfg.return_value = MagicMock()
    return ml


def _make_ctx(ml: MagicMock | None = None) -> MagicMock:
    ctx = MagicMock()
    ctx.ml = ml or _make_ml()
    ctx.md = MetaDict()
    ctx.qub_name = "Q1"
    return ctx


def _make_req(ml: MagicMock | None = None) -> RunRequest:
    return RunRequest(
        md=MagicMock(),
        ml=ml or _make_ml(),
        soc=None,
        soccfg=None,
    )


def _lower(schema: CfgSchema, req: RunRequest) -> dict[str, object]:
    return schema.to_raw_dict(None, req.ml)


@pytest.mark.parametrize(
    ("adapter", "cfg_model", "sweep_axis"),
    [
        (SingleToneFreqAdapter(), FreqCfg, "freq"),
        (SingleToneLengthAdapter(), LengthCfg, "length"),
    ],
)
def test_reset_build_exp_cfg_delegates_to_ml_make_cfg(
    adapter: Any, cfg_model: type, sweep_axis: str
) -> None:
    ml = _make_ml()
    # make_default_cfg validates the value tree (ADR-0011) before lowering.
    raw = _lower(adapter.make_default_cfg(_make_ctx(ml)), _make_req(ml))

    assert "modules" in raw
    assert "sweep" in raw
    modules = cast(dict[str, Any], raw["modules"])
    assert "tested_reset" in modules
    assert "readout" in modules
    # optional reset / init_pulse disabled by default (no library entries)
    assert "reset" not in modules
    assert "init_pulse" not in modules

    sweep = cast(dict[str, Any], raw["sweep"])
    assert isinstance(sweep[sweep_axis], SweepCfg)

    adapter.build_exp_cfg(raw, _make_req(ml))
    ml.make_cfg.assert_called_once_with(raw, cfg_model)


def test_reset_freq_sweep_centres_on_reset_f() -> None:
    from zcu_tools.gui.app.main.adapter import (
        CfgSectionValue,
        EvalValue,
        SweepValue,
    )

    ctx = _make_ctx(_make_ml())
    ctx.md.reset_f = 1500.0
    schema = SingleToneFreqAdapter().make_default_cfg(ctx)
    sweep = schema.value.fields["sweep"]
    assert isinstance(sweep, CfgSectionValue)
    freq_sweep = sweep.fields["freq"]
    assert isinstance(freq_sweep, SweepValue)
    # Without resetf_w the edges are reset_f ± 50 (kept md-linked).
    assert isinstance(freq_sweep.start, EvalValue)
    assert isinstance(freq_sweep.stop, EvalValue)
    assert freq_sweep.start.expr == "reset_f - 50.0"
    assert freq_sweep.stop.expr == "reset_f + 50.0"


def test_reset_freq_locks_tested_reset_freq() -> None:
    from zcu_tools.gui.app.main.adapter import (
        CfgSectionValue,
        DirectValue,
        ModuleRefValue,
    )

    schema = SingleToneFreqAdapter().make_default_cfg(_make_ctx(_make_ml()))
    modules = schema.value.fields["modules"]
    assert isinstance(modules, CfgSectionValue)
    tested_reset = modules.fields["tested_reset"]
    assert isinstance(tested_reset, ModuleRefValue)
    pulse_cfg = tested_reset.value.fields["pulse_cfg"]
    assert isinstance(pulse_cfg, CfgSectionValue)
    # The sweep axis owns the frequency, so it is locked to 0.0 (notebook freq=0).
    assert pulse_cfg.fields["freq"] == DirectValue(0.0)


def test_reset_freq_writeback_proposes_reset_f_and_resetf_w() -> None:
    from matplotlib.figure import Figure
    from zcu_tools.experiment.v2_gui.adapters.twotone.reset.single_tone.freq import (
        SingleToneFreqAnalyzeResult,
    )

    adapter = SingleToneFreqAdapter()

    analyze = SingleToneFreqAnalyzeResult(freq=1523.4, fwhm=3.2, figure=Figure())
    items = adapter.get_writeback_items(
        WritebackRequest(
            run_result=cast(FreqResult, MagicMock(spec=FreqResult)),
            analyze_result=analyze,
            ctx=cast(Any, MagicMock()),
        )
    )

    assert all(isinstance(it, MetaDictWriteback) for it in items)
    by_name = {cast(MetaDictWriteback, it).target_name: it for it in items}
    assert set(by_name) == {"reset_f", "resetf_w"}
    assert cast(MetaDictWriteback, by_name["reset_f"]).proposed_value == 1523.4
    assert cast(MetaDictWriteback, by_name["resetf_w"]).proposed_value == 3.2


def test_reset_length_sets_tested_reset_freq_from_md() -> None:
    from zcu_tools.gui.app.main.adapter import (
        CfgSectionValue,
        EvalValue,
        ModuleRefValue,
    )

    ctx = _make_ctx(_make_ml())
    ctx.md.reset_f = 1500.0
    schema = SingleToneLengthAdapter().make_default_cfg(ctx)
    modules = schema.value.fields["modules"]
    assert isinstance(modules, CfgSectionValue)
    tested_reset = modules.fields["tested_reset"]
    assert isinstance(tested_reset, ModuleRefValue)
    pulse_cfg = tested_reset.value.fields["pulse_cfg"]
    assert isinstance(pulse_cfg, CfgSectionValue)
    # Length sweep holds freq fixed at the calibrated reset_f (md-linked).
    assert isinstance(pulse_cfg.fields["freq"], EvalValue)
    assert cast(EvalValue, pulse_cfg.fields["freq"]).expr == "reset_f"


def test_reset_length_has_no_writeback() -> None:
    from matplotlib.figure import Figure
    from zcu_tools.experiment.v2_gui.adapters.twotone.reset.single_tone.length import (
        SingleToneLengthAnalyzeResult,
    )

    adapter = SingleToneLengthAdapter()
    items = adapter.get_writeback_items(
        WritebackRequest(
            run_result=cast(LengthResult, MagicMock(spec=LengthResult)),
            analyze_result=SingleToneLengthAnalyzeResult(figure=Figure()),
            ctx=cast(Any, MagicMock()),
        )
    )
    assert list(items) == []


@pytest.mark.parametrize(
    "adapter",
    [SingleToneFreqAdapter(), SingleToneLengthAdapter()],
)
def test_reset_run_without_soc_fast_fails(adapter: Any) -> None:
    ml = _make_ml()
    schema = adapter.make_default_cfg(_make_ctx(ml))

    with pytest.raises(RuntimeError, match="soc is required"):
        adapter.run(_make_req(ml), schema)
