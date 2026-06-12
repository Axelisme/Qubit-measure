from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from zcu_tools.experiment.v2.twotone.reset.dual_tone.freq import (
    FreqCfg as DualFreqCfg,
)
from zcu_tools.experiment.v2.twotone.reset.dual_tone.freq import (
    FreqResult as DualFreqResult,
)
from zcu_tools.experiment.v2.twotone.reset.dual_tone.length import (
    LengthCfg as DualLengthCfg,
)
from zcu_tools.experiment.v2.twotone.reset.dual_tone.length import (
    LengthResult as DualLengthResult,
)
from zcu_tools.experiment.v2.twotone.reset.dual_tone.power import (
    PowerCfg as DualPowerCfg,
)
from zcu_tools.experiment.v2.twotone.reset.dual_tone.power import (
    PowerResult as DualPowerResult,
)
from zcu_tools.experiment.v2.twotone.reset.single_tone.freq import (
    FreqCfg,
    FreqResult,
)
from zcu_tools.experiment.v2.twotone.reset.single_tone.length import (
    LengthCfg,
    LengthResult,
)
from zcu_tools.experiment.v2_gui.adapters.twotone import (
    DualToneFreqAdapter,
    DualToneLengthAdapter,
    DualTonePowerAdapter,
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


# --- dual-tone (two-pulse reset) adapters --------------------------------


@pytest.mark.parametrize(
    ("adapter", "cfg_model", "sweep_axes"),
    [
        (DualToneFreqAdapter(), DualFreqCfg, ("freq1", "freq2")),
        (DualTonePowerAdapter(), DualPowerCfg, ("gain1", "gain2")),
        (DualToneLengthAdapter(), DualLengthCfg, ("length",)),
    ],
)
def test_dual_reset_build_exp_cfg_round_trip(
    adapter: Any, cfg_model: type, sweep_axes: tuple[str, ...]
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
    for axis in sweep_axes:
        assert isinstance(sweep[axis], SweepCfg)

    adapter.build_exp_cfg(raw, _make_req(ml))
    ml.make_cfg.assert_called_once_with(raw, cfg_model)


def test_dual_reset_freq_centres_each_axis_and_locks_freqs() -> None:
    from zcu_tools.gui.app.main.adapter import (
        CfgSectionValue,
        DirectValue,
        EvalValue,
        ModuleRefValue,
        SweepValue,
    )

    ctx = _make_ctx(_make_ml())
    ctx.md.reset_f1 = 1500.0
    ctx.md.reset_f2 = 2500.0
    schema = DualToneFreqAdapter().make_default_cfg(ctx)

    sweep = schema.value.fields["sweep"]
    assert isinstance(sweep, CfgSectionValue)
    f1 = sweep.fields["freq1"]
    f2 = sweep.fields["freq2"]
    assert isinstance(f1, SweepValue) and isinstance(f2, SweepValue)
    # Each axis centres on its own md key (kept md-linked without a width key).
    assert isinstance(f1.start, EvalValue) and f1.start.expr == "reset_f1 - 50.0"
    assert isinstance(f2.stop, EvalValue) and f2.stop.expr == "reset_f2 + 50.0"

    # Both tested-reset tone frequencies are owned by the sweep → locked to 0.0.
    modules = schema.value.fields["modules"]
    assert isinstance(modules, CfgSectionValue)
    tested = modules.fields["tested_reset"]
    assert isinstance(tested, ModuleRefValue)
    for key in ("pulse1_cfg", "pulse2_cfg"):
        pulse = tested.value.fields[key]
        assert isinstance(pulse, CfgSectionValue)
        assert pulse.fields["freq"] == DirectValue(0.0)


def test_dual_reset_freq_gains_md_link() -> None:
    from zcu_tools.gui.app.main.adapter import (
        CfgSectionValue,
        EvalValue,
        ModuleRefValue,
    )

    ctx = _make_ctx(_make_ml())
    ctx.md.reset_gain1 = 0.7
    ctx.md.reset_gain2 = 0.8
    schema = DualToneFreqAdapter().make_default_cfg(ctx)
    modules = schema.value.fields["modules"]
    assert isinstance(modules, CfgSectionValue)
    tested = modules.fields["tested_reset"]
    assert isinstance(tested, ModuleRefValue)
    p1 = tested.value.fields["pulse1_cfg"]
    p2 = tested.value.fields["pulse2_cfg"]
    assert isinstance(p1, CfgSectionValue) and isinstance(p2, CfgSectionValue)
    # Gains are not swept here → held md-linked to the calibrated values (D2(a)).
    assert isinstance(p1.fields["gain"], EvalValue)
    assert cast(EvalValue, p1.fields["gain"]).expr == "reset_gain1"
    assert cast(EvalValue, p2.fields["gain"]).expr == "reset_gain2"


def test_dual_reset_freq_writeback_proposes_reset_f1_and_f2() -> None:
    from matplotlib.figure import Figure
    from zcu_tools.experiment.v2_gui.adapters.twotone.reset.dual_tone.freq import (
        DualToneFreqAnalyzeResult,
    )

    adapter = DualToneFreqAdapter()
    analyze = DualToneFreqAnalyzeResult(freq1=1438.0, freq2=2615.0, figure=Figure())
    items = adapter.get_writeback_items(
        WritebackRequest(
            run_result=cast(DualFreqResult, MagicMock(spec=DualFreqResult)),
            analyze_result=analyze,
            ctx=cast(Any, MagicMock()),
        )
    )
    assert all(isinstance(it, MetaDictWriteback) for it in items)
    by_name = {cast(MetaDictWriteback, it).target_name: it for it in items}
    assert set(by_name) == {"reset_f1", "reset_f2"}
    assert cast(MetaDictWriteback, by_name["reset_f1"]).proposed_value == 1438.0
    assert cast(MetaDictWriteback, by_name["reset_f2"]).proposed_value == 2615.0


def test_dual_reset_freq_run_passes_method_hard() -> None:
    ml = _make_ml()
    adapter = DualToneFreqAdapter()
    schema = adapter.make_default_cfg(_make_ctx(ml))

    captured: dict[str, Any] = {}

    class _FakeExp:
        def run(self, soc: Any, soccfg: Any, cfg: Any, *, method: str = "soft") -> str:
            captured["method"] = method
            return "result"

    adapter.exp_cls = _FakeExp  # type: ignore[assignment]
    req = RunRequest(md=MagicMock(), ml=ml, soc=MagicMock(), soccfg=MagicMock())
    assert adapter.run(req, schema) == "result"
    # The dual-tone freq map runs as a 2D hard sweep (notebook: method="hard").
    assert captured["method"] == "hard"


def test_dual_reset_power_locks_gains_and_freqs_md_link() -> None:
    from zcu_tools.gui.app.main.adapter import (
        CfgSectionValue,
        DirectValue,
        EvalValue,
        ModuleRefValue,
    )

    ctx = _make_ctx(_make_ml())
    ctx.md.reset_f1 = 1438.0
    ctx.md.reset_f2 = 2615.0
    schema = DualTonePowerAdapter().make_default_cfg(ctx)
    modules = schema.value.fields["modules"]
    assert isinstance(modules, CfgSectionValue)
    tested = modules.fields["tested_reset"]
    assert isinstance(tested, ModuleRefValue)
    p1 = tested.value.fields["pulse1_cfg"]
    p2 = tested.value.fields["pulse2_cfg"]
    assert isinstance(p1, CfgSectionValue) and isinstance(p2, CfgSectionValue)
    # Gains are owned by the sweep → locked to 0.0.
    assert p1.fields["gain"] == DirectValue(0.0)
    assert p2.fields["gain"] == DirectValue(0.0)
    # Frequencies held md-linked to the calibrated values (D2(a)).
    assert isinstance(p1.fields["freq"], EvalValue)
    assert cast(EvalValue, p1.fields["freq"]).expr == "reset_f1"
    assert cast(EvalValue, p2.fields["freq"]).expr == "reset_f2"


def test_dual_reset_power_writeback_proposes_reset_gains() -> None:
    from matplotlib.figure import Figure
    from zcu_tools.experiment.v2_gui.adapters.twotone.reset.dual_tone.power import (
        DualTonePowerAnalyzeResult,
    )

    adapter = DualTonePowerAdapter()
    analyze = DualTonePowerAnalyzeResult(gain1=0.5, gain2=0.7, figure=Figure())
    items = adapter.get_writeback_items(
        WritebackRequest(
            run_result=cast(DualPowerResult, MagicMock(spec=DualPowerResult)),
            analyze_result=analyze,
            ctx=cast(Any, MagicMock()),
        )
    )
    assert all(isinstance(it, MetaDictWriteback) for it in items)
    by_name = {cast(MetaDictWriteback, it).target_name: it for it in items}
    assert set(by_name) == {"reset_gain1", "reset_gain2"}
    assert cast(MetaDictWriteback, by_name["reset_gain1"]).proposed_value == 0.5
    assert cast(MetaDictWriteback, by_name["reset_gain2"]).proposed_value == 0.7


def test_dual_reset_length_carries_calibrated_reset_and_no_writeback() -> None:
    from matplotlib.figure import Figure
    from zcu_tools.experiment.v2_gui.adapters.twotone.reset.dual_tone.length import (
        DualToneLengthAnalyzeResult,
    )
    from zcu_tools.gui.app.main.adapter import (
        CfgSectionValue,
        EvalValue,
        ModuleRefValue,
    )

    ctx = _make_ctx(_make_ml())
    ctx.md.reset_f1 = 1.0
    ctx.md.reset_f2 = 2.0
    ctx.md.reset_gain1 = 0.5
    ctx.md.reset_gain2 = 0.6
    schema = DualToneLengthAdapter().make_default_cfg(ctx)
    modules = schema.value.fields["modules"]
    assert isinstance(modules, CfgSectionValue)
    tested = modules.fields["tested_reset"]
    assert isinstance(tested, ModuleRefValue)
    p1 = tested.value.fields["pulse1_cfg"]
    p2 = tested.value.fields["pulse2_cfg"]
    assert isinstance(p1, CfgSectionValue) and isinstance(p2, CfgSectionValue)
    # All four calibrated knobs md-link so the cfg snapshot carries the fully
    # calibrated reset for the final reset_120 registration (D2(a)).
    assert cast(EvalValue, p1.fields["freq"]).expr == "reset_f1"
    assert cast(EvalValue, p2.fields["freq"]).expr == "reset_f2"
    assert cast(EvalValue, p1.fields["gain"]).expr == "reset_gain1"
    assert cast(EvalValue, p2.fields["gain"]).expr == "reset_gain2"

    # D5: no scalar fit → no writeback.
    items = DualToneLengthAdapter().get_writeback_items(
        WritebackRequest(
            run_result=cast(DualLengthResult, MagicMock(spec=DualLengthResult)),
            analyze_result=DualToneLengthAnalyzeResult(figure=Figure()),
            ctx=cast(Any, MagicMock()),
        )
    )
    assert list(items) == []


@pytest.mark.parametrize(
    "adapter",
    [DualTonePowerAdapter(), DualToneLengthAdapter()],
)
def test_dual_reset_run_without_soc_fast_fails(adapter: Any) -> None:
    ml = _make_ml()
    schema = adapter.make_default_cfg(_make_ctx(ml))

    with pytest.raises(RuntimeError, match="soc is required"):
        adapter.run(_make_req(ml), schema)


def test_dual_reset_freq_run_without_soc_fast_fails() -> None:
    # The freq adapter overrides run (to inject method="hard"); it must keep the
    # same soc-handle guard as the base run.
    ml = _make_ml()
    adapter = DualToneFreqAdapter()
    schema = adapter.make_default_cfg(_make_ctx(ml))

    with pytest.raises(RuntimeError, match="soc is required"):
        adapter.run(_make_req(ml), schema)
