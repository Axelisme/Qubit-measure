from __future__ import annotations

from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from zcu_tools.experiment.v2.twotone.reset.bath.freq import (
    FreqGainCfg as BathFreqGainCfg,
)
from zcu_tools.experiment.v2.twotone.reset.bath.freq import (
    FreqGainResult as BathFreqGainResult,
)
from zcu_tools.experiment.v2.twotone.reset.bath.length import (
    LengthCfg as BathLengthCfg,
)
from zcu_tools.experiment.v2.twotone.reset.bath.length import (
    LengthResult as BathLengthResult,
)
from zcu_tools.experiment.v2.twotone.reset.bath.phase import (
    PhaseCfg as BathPhaseCfg,
)
from zcu_tools.experiment.v2.twotone.reset.bath.phase import (
    PhaseResult as BathPhaseResult,
)
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
from zcu_tools.experiment.v2.twotone.reset.rabi_check import RabiCheckCfg
from zcu_tools.experiment.v2.twotone.reset.single_tone.freq import (
    FreqCfg,
    FreqResult,
)
from zcu_tools.experiment.v2.twotone.reset.single_tone.length import (
    LengthCfg,
    LengthResult,
)
from zcu_tools.experiment.v2_gui.adapters.twotone import (
    BathFreqGainAdapter,
    BathLengthAdapter,
    BathPhaseAdapter,
    DualToneFreqAdapter,
    DualToneLengthAdapter,
    DualTonePowerAdapter,
    RabiCheckAdapter,
    SingleToneFreqAdapter,
    SingleToneLengthAdapter,
)
from zcu_tools.gui.app.main.adapter import (
    CfgSchema,
    MetaDictWriteback,
    ModuleWriteback,
    RunRequest,
    WritebackRequest,
)
from zcu_tools.meta_tool import MetaDict
from zcu_tools.program.v2 import PulseCfg, SweepCfg
from zcu_tools.program.v2.modules import (
    BathResetCfg,
    PulseResetCfg,
    TwoPulseResetCfg,
)
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg


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


def _make_pulse(freq: float = 100.0, gain: float = 0.5) -> PulseCfg:
    """Minimal PulseCfg with a const waveform, for building reset snapshots."""
    return PulseCfg(
        type="pulse",
        waveform=ConstWaveformCfg(style="const", length=10.0),
        ch=0,
        nqz=1,
        freq=freq,
        gain=gain,
    )


def _snapshot_with_tested_reset(tested_reset: Any) -> Any:
    """A cfg_snapshot mock exposing modules.tested_reset (the calibrated reset).

    The reset adapters read only modules.tested_reset off the snapshot, so a mock
    carrying the real reset cfg is enough to drive module writeback. Returned
    untyped (Any) so it can seed the frozen result dataclasses' cfg_snapshot.
    """
    modules = MagicMock()
    modules.tested_reset = tested_reset
    cfg = MagicMock()
    cfg.modules = modules
    return cfg


def _ctx_with_md(**md_values: float) -> Any:
    """A WritebackRequest ctx whose md is a real MetaDict seeded with md_values.

    The gated module writeback reads ``ctx.md`` (via md_has_key/md_get_float), so
    module-writeback tests need a real MetaDict — a bare MagicMock md would make
    every key "present" and defeat the gate.
    """
    ctx = MagicMock()
    md = MetaDict()
    for key, value in md_values.items():
        setattr(md, key, value)
    ctx.md = md
    return ctx


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


def _single_freq_items(*, reset_f: float | None, with_snapshot: bool) -> list[Any]:
    import numpy as np
    from matplotlib.figure import Figure
    from zcu_tools.experiment.v2_gui.adapters.twotone.reset.single_tone.freq import (
        SingleToneFreqAnalyzeResult,
    )

    snapshot: Any = None
    if with_snapshot:
        snapshot = _snapshot_with_tested_reset(
            PulseResetCfg(pulse_cfg=_make_pulse(freq=0.0))
        )
    run_result = FreqResult(
        freqs=np.array([0.0, 1.0]),
        signals=np.array([0.0, 1.0], dtype=complex),
        cfg_snapshot=snapshot,
    )
    md_kwargs = {} if reset_f is None else {"reset_f": reset_f}
    return list(
        SingleToneFreqAdapter().get_writeback_items(
            WritebackRequest(
                run_result=run_result,
                analyze_result=SingleToneFreqAnalyzeResult(
                    freq=1523.4, fwhm=3.2, figure=Figure()
                ),
                ctx=_ctx_with_md(**md_kwargs),
            )
        )
    )


def test_reset_freq_writeback_proposes_reset_f_and_resetf_w() -> None:
    # md scalar items are always present; without the reset_f gate met there is no
    # module item.
    items = _single_freq_items(reset_f=None, with_snapshot=True)
    md_items = [it for it in items if isinstance(it, MetaDictWriteback)]
    by_name = {it.target_name: it for it in md_items}
    assert set(by_name) == {"reset_f", "resetf_w"}
    assert by_name["reset_f"].proposed_value == 1523.4
    assert by_name["resetf_w"].proposed_value == 3.2
    assert not any(isinstance(it, ModuleWriteback) for it in items)


def test_reset_freq_gated_reset_10_when_md_present() -> None:
    from zcu_tools.gui.app.main.adapter import CfgSectionValue, DirectValue

    # md has reset_f and a snapshot exists → reset_10 module proposed, freq from md.
    items = _single_freq_items(reset_f=1500.0, with_snapshot=True)
    mod_items = [it for it in items if isinstance(it, ModuleWriteback)]
    assert len(mod_items) == 1
    item = mod_items[0]
    assert item.target_name == "reset_10"
    assert item.description == "Reset with one pulse from 1 to 0"
    assert item.edit_schema is not None
    pulse_cfg = item.edit_schema.value.fields["pulse_cfg"]
    assert isinstance(pulse_cfg, CfgSectionValue)
    assert pulse_cfg.fields["freq"] == DirectValue(1500.0)


def test_reset_freq_no_reset_10_without_snapshot() -> None:
    # md has reset_f but no cfg_snapshot → only the md items remain.
    items = _single_freq_items(reset_f=1500.0, with_snapshot=False)
    assert all(isinstance(it, MetaDictWriteback) for it in items)


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


def _single_length_items(*, reset_f: float | None, with_snapshot: bool) -> list[Any]:
    import numpy as np
    from matplotlib.figure import Figure
    from zcu_tools.experiment.v2_gui.adapters.twotone.reset.single_tone.length import (
        SingleToneLengthAnalyzeResult,
    )

    snapshot: Any = None
    if with_snapshot:
        # Swept freq is lock_literal 0.0 in the snapshot; md must overwrite it.
        snapshot = _snapshot_with_tested_reset(
            PulseResetCfg(pulse_cfg=_make_pulse(freq=0.0))
        )
    run_result = LengthResult(
        lengths=np.array([0.1, 0.2]),
        signals=np.array([0.0, 1.0], dtype=complex),
        cfg_snapshot=snapshot,
    )
    md_kwargs = {} if reset_f is None else {"reset_f": reset_f}
    return list(
        SingleToneLengthAdapter().get_writeback_items(
            WritebackRequest(
                run_result=run_result,
                analyze_result=SingleToneLengthAnalyzeResult(figure=Figure()),
                ctx=_ctx_with_md(**md_kwargs),
            )
        )
    )


def test_reset_length_gated_reset_10_when_md_present() -> None:
    from zcu_tools.gui.app.main.adapter import CfgSchema, CfgSectionValue, DirectValue

    # Gated per-experiment: md has reset_f + a snapshot → reset_10 proposed, with
    # the calibrated sideband freq taken from md (overwriting the swept 0.0).
    items = _single_length_items(reset_f=1500.0, with_snapshot=True)
    mod_items = [it for it in items if isinstance(it, ModuleWriteback)]
    assert len(mod_items) == 1
    item = mod_items[0]
    assert item.target_name == "reset_10"
    assert item.description == "Reset with one pulse from 1 to 0"
    assert isinstance(item.edit_schema, CfgSchema)
    pulse_cfg = item.edit_schema.value.fields["pulse_cfg"]
    assert isinstance(pulse_cfg, CfgSectionValue)
    assert pulse_cfg.fields["freq"] == DirectValue(1500.0)


def test_reset_length_no_module_without_md() -> None:
    # md lacks reset_f → gate not met → no module item (and no md item here).
    items = _single_length_items(reset_f=None, with_snapshot=True)
    assert list(items) == []


def test_reset_length_no_module_without_snapshot() -> None:
    # md has reset_f but no cfg_snapshot → gate not met → no module item.
    items = _single_length_items(reset_f=1500.0, with_snapshot=False)
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


# The four md keys the gated reset_120 proposal needs (freq1/2 + gain1/2).
_DUAL_RESET_MD = {
    "reset_f1": 1438.0,
    "reset_f2": 2615.0,
    "reset_gain1": 0.5,
    "reset_gain2": 0.6,
}


def _dual_two_pulse_snapshot() -> Any:
    # Swept fields are lock_literal 0.0 in the snapshot; md overwrites all four.
    return _snapshot_with_tested_reset(
        TwoPulseResetCfg(
            pulse1_cfg=_make_pulse(freq=0.0, gain=0.0),
            pulse2_cfg=_make_pulse(freq=0.0, gain=0.0),
        )
    )


def _assert_reset_120(items: list[Any]) -> None:
    from zcu_tools.gui.app.main.adapter import CfgSchema, CfgSectionValue, DirectValue

    mod_items = [it for it in items if isinstance(it, ModuleWriteback)]
    assert len(mod_items) == 1
    item = mod_items[0]
    assert item.target_name == "reset_120"
    assert item.description == "Reset with two pulse from 1 to 2 to 0"
    assert isinstance(item.edit_schema, CfgSchema)
    p1 = item.edit_schema.value.fields["pulse1_cfg"]
    p2 = item.edit_schema.value.fields["pulse2_cfg"]
    assert isinstance(p1, CfgSectionValue) and isinstance(p2, CfgSectionValue)
    # All four calibrated fields taken from md (overwriting the swept 0.0).
    assert p1.fields["freq"] == DirectValue(1438.0)
    assert p2.fields["freq"] == DirectValue(2615.0)
    assert p1.fields["gain"] == DirectValue(0.5)
    assert p2.fields["gain"] == DirectValue(0.6)


def test_dual_reset_freq_writeback_proposes_reset_f1_and_f2() -> None:
    import numpy as np
    from matplotlib.figure import Figure
    from zcu_tools.experiment.v2_gui.adapters.twotone.reset.dual_tone.freq import (
        DualToneFreqAnalyzeResult,
    )

    # md scalar items always present; without all four gate keys no module item.
    run_result = DualFreqResult(
        freqs1=np.array([0.0, 1.0]),
        freqs2=np.array([0.0, 1.0]),
        signals=np.array([0.0, 1.0], dtype=complex),
        cfg_snapshot=_dual_two_pulse_snapshot(),
    )
    analyze = DualToneFreqAnalyzeResult(freq1=1438.0, freq2=2615.0, figure=Figure())
    items = list(
        DualToneFreqAdapter().get_writeback_items(
            WritebackRequest(
                run_result=run_result, analyze_result=analyze, ctx=_ctx_with_md()
            )
        )
    )
    md_items = [it for it in items if isinstance(it, MetaDictWriteback)]
    by_name = {it.target_name: it for it in md_items}
    assert set(by_name) == {"reset_f1", "reset_f2"}
    assert by_name["reset_f1"].proposed_value == 1438.0
    assert by_name["reset_f2"].proposed_value == 2615.0
    assert not any(isinstance(it, ModuleWriteback) for it in items)


def test_dual_reset_per_experiment_each_step_emits_reset_120() -> None:
    """freq / power / length all propose reset_120 once md is complete — the
    gated, order-independent, repeatable behavior (not a single 'last step')."""
    import numpy as np
    from matplotlib.figure import Figure
    from zcu_tools.experiment.v2_gui.adapters.twotone.reset.dual_tone.freq import (
        DualToneFreqAnalyzeResult,
    )
    from zcu_tools.experiment.v2_gui.adapters.twotone.reset.dual_tone.length import (
        DualToneLengthAnalyzeResult,
    )
    from zcu_tools.experiment.v2_gui.adapters.twotone.reset.dual_tone.power import (
        DualTonePowerAnalyzeResult,
    )

    freq_items = list(
        DualToneFreqAdapter().get_writeback_items(
            WritebackRequest(
                run_result=DualFreqResult(
                    freqs1=np.array([0.0, 1.0]),
                    freqs2=np.array([0.0, 1.0]),
                    signals=np.array([0.0, 1.0], dtype=complex),
                    cfg_snapshot=_dual_two_pulse_snapshot(),
                ),
                analyze_result=DualToneFreqAnalyzeResult(
                    freq1=1438.0, freq2=2615.0, figure=Figure()
                ),
                ctx=_ctx_with_md(**_DUAL_RESET_MD),
            )
        )
    )
    power_items = list(
        DualTonePowerAdapter().get_writeback_items(
            WritebackRequest(
                run_result=DualPowerResult(
                    gains1=np.array([0.0, 1.0]),
                    gains2=np.array([0.0, 1.0]),
                    signals=np.array([0.0, 1.0], dtype=complex),
                    cfg_snapshot=_dual_two_pulse_snapshot(),
                ),
                analyze_result=DualTonePowerAnalyzeResult(
                    gain1=0.5, gain2=0.7, figure=Figure()
                ),
                ctx=_ctx_with_md(**_DUAL_RESET_MD),
            )
        )
    )
    length_items = list(
        DualToneLengthAdapter().get_writeback_items(
            WritebackRequest(
                run_result=DualLengthResult(
                    lengths=np.array([0.1, 0.2]),
                    signals=np.array([0.0, 1.0], dtype=complex),
                    cfg_snapshot=_dual_two_pulse_snapshot(),
                ),
                analyze_result=DualToneLengthAnalyzeResult(figure=Figure()),
                ctx=_ctx_with_md(**_DUAL_RESET_MD),
            )
        )
    )
    # Each step's own cfg snapshot is the template; md overwrites all four fields.
    _assert_reset_120(freq_items)
    _assert_reset_120(power_items)
    _assert_reset_120(length_items)


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
    import numpy as np
    from matplotlib.figure import Figure
    from zcu_tools.experiment.v2_gui.adapters.twotone.reset.dual_tone.power import (
        DualTonePowerAnalyzeResult,
    )

    # md scalar items always present; without all four gate keys no module item.
    run_result = DualPowerResult(
        gains1=np.array([0.0, 1.0]),
        gains2=np.array([0.0, 1.0]),
        signals=np.array([0.0, 1.0], dtype=complex),
        cfg_snapshot=_dual_two_pulse_snapshot(),
    )
    analyze = DualTonePowerAnalyzeResult(gain1=0.5, gain2=0.7, figure=Figure())
    items = list(
        DualTonePowerAdapter().get_writeback_items(
            WritebackRequest(
                run_result=run_result, analyze_result=analyze, ctx=_ctx_with_md()
            )
        )
    )
    md_items = [it for it in items if isinstance(it, MetaDictWriteback)]
    by_name = {it.target_name: it for it in md_items}
    assert set(by_name) == {"reset_gain1", "reset_gain2"}
    assert by_name["reset_gain1"].proposed_value == 0.5
    assert by_name["reset_gain2"].proposed_value == 0.7
    assert not any(isinstance(it, ModuleWriteback) for it in items)


def test_dual_reset_length_carries_calibrated_reset() -> None:
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


def _dual_length_items(*, md: dict[str, float], with_snapshot: bool) -> list[Any]:
    import numpy as np
    from matplotlib.figure import Figure
    from zcu_tools.experiment.v2_gui.adapters.twotone.reset.dual_tone.length import (
        DualToneLengthAnalyzeResult,
    )

    run_result = DualLengthResult(
        lengths=np.array([0.1, 0.2]),
        signals=np.array([0.0, 1.0], dtype=complex),
        cfg_snapshot=_dual_two_pulse_snapshot() if with_snapshot else None,
    )
    return list(
        DualToneLengthAdapter().get_writeback_items(
            WritebackRequest(
                run_result=run_result,
                analyze_result=DualToneLengthAnalyzeResult(figure=Figure()),
                ctx=_ctx_with_md(**md),
            )
        )
    )


def test_dual_reset_length_gated_reset_120_when_md_complete() -> None:
    # All four md keys present + snapshot → reset_120 proposed, fields from md.
    _assert_reset_120(_dual_length_items(md=_DUAL_RESET_MD, with_snapshot=True))


def test_dual_reset_length_no_module_with_partial_md() -> None:
    # Only some of the four gate keys → gate not met → no module item.
    partial = {"reset_f1": 1438.0, "reset_f2": 2615.0}
    assert list(_dual_length_items(md=partial, with_snapshot=True)) == []


def test_dual_reset_length_no_module_without_snapshot() -> None:
    # md complete but no cfg_snapshot → gate not met → no module item.
    assert list(_dual_length_items(md=_DUAL_RESET_MD, with_snapshot=False)) == []


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


# --- bath-reset adapters -------------------------------------------------


@pytest.mark.parametrize(
    ("adapter", "cfg_model", "sweep_axes"),
    [
        (BathFreqGainAdapter(), BathFreqGainCfg, ("freq", "gain")),
        (BathLengthAdapter(), BathLengthCfg, ("length",)),
        (BathPhaseAdapter(), BathPhaseCfg, ("phase",)),
    ],
)
def test_bath_reset_build_exp_cfg_round_trip(
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


def test_bath_freq_gain_centres_freq_and_locks_cavity_and_pi2() -> None:
    from zcu_tools.gui.app.main.adapter import (
        CfgSectionValue,
        DirectValue,
        EvalValue,
        ModuleRefValue,
        SweepValue,
    )

    ctx = _make_ctx(_make_ml())
    ctx.md.r_f = 6000.0
    ctx.md.rabi_f = 10.0
    schema = BathFreqGainAdapter().make_default_cfg(ctx)

    # Freq axis centres on r_f - rabi_f (md-linked); gain axis is a literal span.
    sweep = schema.value.fields["sweep"]
    assert isinstance(sweep, CfgSectionValue)
    freq = sweep.fields["freq"]
    assert isinstance(freq, SweepValue)
    assert isinstance(freq.start, EvalValue) and freq.start.expr == "r_f - 1.2 * rabi_f"
    assert isinstance(freq.stop, EvalValue) and freq.stop.expr == "r_f - 0.8 * rabi_f"

    modules = schema.value.fields["modules"]
    assert isinstance(modules, CfgSectionValue)
    tested = modules.fields["tested_reset"]
    assert isinstance(tested, ModuleRefValue)
    cavity = tested.value.fields["cavity_tone_cfg"]
    assert isinstance(cavity, CfgSectionValue)
    # The sweep owns cavity freq + gain → locked to 0.0 (notebook: not used).
    assert cavity.fields["freq"] == DirectValue(0.0)
    assert cavity.fields["gain"] == DirectValue(0.0)
    # The domain adds the 4-point phase axis onto pi2_cfg.phase → locked to the
    # notebook base offset (90 deg).
    pi2 = tested.value.fields["pi2_cfg"]
    assert isinstance(pi2, CfgSectionValue)
    assert pi2.fields["phase"] == DirectValue(90.0)


# The four md keys both gated bath variants need (cavity freq/gain + the two
# pi/2 phase keys). reset_bath gates on max_phase, reset_bath_e on min_phase.
_BATH_RESET_MD = {
    "bathreset_freq": 5500.0,
    "bathreset_gain": 0.7,
    "bathreset_max_phase": 12.5,
    "bathreset_min_phase": 192.5,
}


def _bath_snapshot() -> Any:
    # Swept cavity freq/gain + pi/2 phase are lock_literal in the snapshot; md
    # overwrites them.
    return _snapshot_with_tested_reset(
        BathResetCfg(
            cavity_tone_cfg=_make_pulse(freq=0.0, gain=0.0),
            qubit_tone_cfg=_make_pulse(freq=3000.0, gain=0.1),
            pi2_cfg=_make_pulse(freq=3000.0, gain=0.2),
        )
    )


def _assert_reset_bath_pair(items: list[Any]) -> None:
    from zcu_tools.gui.app.main.adapter import CfgSchema, CfgSectionValue, DirectValue

    mod_items = [it for it in items if isinstance(it, ModuleWriteback)]
    by_name = {it.target_name: it for it in mod_items}
    assert set(by_name) == {"reset_bath", "reset_bath_e"}
    assert (
        by_name["reset_bath"].description
        == "Reset to Ground with cavity-assisted bath reset"
    )
    assert (
        by_name["reset_bath_e"].description
        == "Reset to Excited with cavity-assisted bath reset"
    )
    for target, phase in [("reset_bath", 12.5), ("reset_bath_e", 192.5)]:
        edit_schema = by_name[target].edit_schema
        assert isinstance(edit_schema, CfgSchema)
        pi2 = edit_schema.value.fields["pi2_cfg"]
        assert isinstance(pi2, CfgSectionValue)
        assert pi2.fields["phase"] == DirectValue(phase)
        # Cavity freq/gain taken from md (overwriting the swept 0.0).
        cavity = edit_schema.value.fields["cavity_tone_cfg"]
        assert isinstance(cavity, CfgSectionValue)
        assert cavity.fields["freq"] == DirectValue(5500.0)
        assert cavity.fields["gain"] == DirectValue(0.7)


def test_bath_freq_gain_writeback_proposes_bathreset_gain_and_freq() -> None:
    import numpy as np
    from matplotlib.figure import Figure
    from zcu_tools.experiment.v2_gui.adapters.twotone.reset.bath.freq_gain import (
        BathFreqGainAnalyzeResult,
    )

    # md scalar items always present; without the bath gate keys no module item.
    run_result = BathFreqGainResult(
        gains=np.array([0.0, 1.0]),
        freqs=np.array([0.0, 1.0]),
        signals=np.array([0.0, 1.0], dtype=complex),
        cfg_snapshot=_bath_snapshot(),
    )
    analyze = BathFreqGainAnalyzeResult(gain=0.72, freq=5483.0, figure=Figure())
    items = list(
        BathFreqGainAdapter().get_writeback_items(
            WritebackRequest(
                run_result=run_result, analyze_result=analyze, ctx=_ctx_with_md()
            )
        )
    )
    md_items = [it for it in items if isinstance(it, MetaDictWriteback)]
    by_name = {it.target_name: it for it in md_items}
    assert set(by_name) == {"bathreset_gain", "bathreset_freq"}
    assert by_name["bathreset_gain"].proposed_value == 0.72
    assert by_name["bathreset_freq"].proposed_value == 5483.0
    assert not any(isinstance(it, ModuleWriteback) for it in items)


def test_bath_per_experiment_each_step_emits_reset_bath_pair() -> None:
    """freq_gain / length / phase all propose reset_bath + reset_bath_e once the
    bath md is complete — gated, order-independent, repeatable."""
    import numpy as np
    from matplotlib.figure import Figure
    from zcu_tools.experiment.v2_gui.adapters.twotone.reset.bath.freq_gain import (
        BathFreqGainAnalyzeResult,
    )
    from zcu_tools.experiment.v2_gui.adapters.twotone.reset.bath.length import (
        BathLengthAnalyzeResult,
    )

    freq_items = list(
        BathFreqGainAdapter().get_writeback_items(
            WritebackRequest(
                run_result=BathFreqGainResult(
                    gains=np.array([0.0, 1.0]),
                    freqs=np.array([0.0, 1.0]),
                    signals=np.array([0.0, 1.0], dtype=complex),
                    cfg_snapshot=_bath_snapshot(),
                ),
                analyze_result=BathFreqGainAnalyzeResult(
                    gain=0.72, freq=5483.0, figure=Figure()
                ),
                ctx=_ctx_with_md(**_BATH_RESET_MD),
            )
        )
    )
    length_items = list(
        BathLengthAdapter().get_writeback_items(
            WritebackRequest(
                run_result=BathLengthResult(
                    lengths=np.array([0.1, 0.2]),
                    signals=np.array([0.0, 1.0], dtype=complex),
                    cfg_snapshot=_bath_snapshot(),
                ),
                analyze_result=BathLengthAnalyzeResult(figure=Figure()),
                ctx=_ctx_with_md(**_BATH_RESET_MD),
            )
        )
    )
    phase_items = _bath_phase_items(with_snapshot=True, md=_BATH_RESET_MD)
    _assert_reset_bath_pair(freq_items)
    _assert_reset_bath_pair(length_items)
    _assert_reset_bath_pair(phase_items)


def test_bath_freq_gain_save_not_supported() -> None:
    from zcu_tools.gui.app.main.adapter import SaveDataRequest

    # D3: the 2D bath freq-gain experiment writes four phase-resolved files, which
    # the single-path GUI save pipeline cannot represent — save must fast-fail
    # rather than report a path that never exists.
    req = SaveDataRequest(
        run_result=cast(BathFreqGainResult, MagicMock(spec=BathFreqGainResult)),
        data_path="/tmp/whatever.hdf5",
        md=MagicMock(),
        ml=MagicMock(),
        chip_name="C",
        qub_name="Q1",
        res_name="R",
        active_label="flux0",
        comment="",
    )
    with pytest.raises(NotImplementedError, match="not supported"):
        BathFreqGainAdapter().save(req)


def test_bath_length_holds_cavity_md_link_and_no_writeback() -> None:
    from matplotlib.figure import Figure
    from zcu_tools.experiment.v2_gui.adapters.twotone.reset.bath.length import (
        BathLengthAnalyzeResult,
    )
    from zcu_tools.gui.app.main.adapter import (
        CfgSectionValue,
        DirectValue,
        EvalValue,
        ModuleRefValue,
    )

    ctx = _make_ctx(_make_ml())
    ctx.md.bathreset_freq = 5500.0
    ctx.md.bathreset_gain = 0.7
    schema = BathLengthAdapter().make_default_cfg(ctx)
    modules = schema.value.fields["modules"]
    assert isinstance(modules, CfgSectionValue)
    tested = modules.fields["tested_reset"]
    assert isinstance(tested, ModuleRefValue)
    cavity = tested.value.fields["cavity_tone_cfg"]
    assert isinstance(cavity, CfgSectionValue)
    # Cavity freq/gain held md-linked so the cfg snapshot carries the calibrated
    # reset forward (D2(a)).
    assert isinstance(cavity.fields["freq"], EvalValue)
    assert cast(EvalValue, cavity.fields["freq"]).expr == "bathreset_freq"
    assert isinstance(cavity.fields["gain"], EvalValue)
    assert cast(EvalValue, cavity.fields["gain"]).expr == "bathreset_gain"
    # The domain replaces pi2_cfg.phase with its own sweep → locked off the form.
    pi2 = tested.value.fields["pi2_cfg"]
    assert isinstance(pi2, CfgSectionValue)
    assert pi2.fields["phase"] == DirectValue(90.0)

    # D5: no scalar fit → no md writeback; with empty md the gated module proposal
    # is also withheld, so nothing is proposed.
    import numpy as np

    run_result = BathLengthResult(
        lengths=np.array([0.1, 0.2]),
        signals=np.array([0.0, 1.0], dtype=complex),
        cfg_snapshot=_bath_snapshot(),
    )
    items = BathLengthAdapter().get_writeback_items(
        WritebackRequest(
            run_result=run_result,
            analyze_result=BathLengthAnalyzeResult(figure=Figure()),
            ctx=_ctx_with_md(),
        )
    )
    assert list(items) == []


def test_bath_phase_locks_pi2_phase_and_holds_cavity() -> None:
    from zcu_tools.gui.app.main.adapter import (
        CfgSectionValue,
        DirectValue,
        EvalValue,
        ModuleRefValue,
        SweepValue,
    )

    ctx = _make_ctx(_make_ml())
    ctx.md.bathreset_freq = 5500.0
    ctx.md.bathreset_gain = 0.7
    schema = BathPhaseAdapter().make_default_cfg(ctx)

    sweep = schema.value.fields["sweep"]
    assert isinstance(sweep, CfgSectionValue)
    phase = sweep.fields["phase"]
    assert isinstance(phase, SweepValue)
    assert phase.start == -360.0 and phase.stop == 360.0

    modules = schema.value.fields["modules"]
    assert isinstance(modules, CfgSectionValue)
    tested = modules.fields["tested_reset"]
    assert isinstance(tested, ModuleRefValue)
    # The sweep owns pi2_cfg.phase → locked to 0.0.
    pi2 = tested.value.fields["pi2_cfg"]
    assert isinstance(pi2, CfgSectionValue)
    assert pi2.fields["phase"] == DirectValue(0.0)
    # Cavity still carries the calibrated freq/gain forward (D2(a)).
    cavity = tested.value.fields["cavity_tone_cfg"]
    assert isinstance(cavity, CfgSectionValue)
    assert isinstance(cavity.fields["freq"], EvalValue)
    assert cast(EvalValue, cavity.fields["freq"]).expr == "bathreset_freq"


def _bath_phase_items(
    *,
    with_snapshot: bool,
    max_phase: float = 12.5,
    min_phase: float = 192.5,
    md: dict[str, float] | None = None,
) -> list[Any]:
    import numpy as np
    from matplotlib.figure import Figure
    from zcu_tools.experiment.v2_gui.adapters.twotone.reset.bath.phase import (
        BathPhaseAnalyzeResult,
    )

    run_result = BathPhaseResult(
        phases=np.array([-360.0, 360.0]),
        signals=np.array([0.0, 1.0], dtype=complex),
        cfg_snapshot=_bath_snapshot() if with_snapshot else None,
    )

    analyze = BathPhaseAnalyzeResult(
        max_phase=max_phase, min_phase=min_phase, figure=Figure()
    )
    return list(
        BathPhaseAdapter().get_writeback_items(
            WritebackRequest(
                run_result=run_result,
                analyze_result=analyze,
                ctx=_ctx_with_md(**(md or {})),
            )
        )
    )


def test_bath_phase_writeback_proposes_max_and_min_phase() -> None:
    # The two md scalar items are always present (independent of cfg_snapshot).
    items = _bath_phase_items(with_snapshot=False)
    md_items = [it for it in items if isinstance(it, MetaDictWriteback)]
    by_name = {it.target_name: it for it in md_items}
    assert set(by_name) == {"bathreset_max_phase", "bathreset_min_phase"}
    assert by_name["bathreset_max_phase"].proposed_value == 12.5
    assert by_name["bathreset_min_phase"].proposed_value == 192.5


def test_bath_phase_registers_reset_bath_and_reset_bath_e_modules() -> None:
    # Gated per-experiment: md carries cavity freq/gain + both pi/2 phases → the
    # phase step proposes reset_bath (max/ground) and reset_bath_e (min/excited),
    # each with its pi/2 phase taken from the matching md key.
    items = _bath_phase_items(with_snapshot=True, md=_BATH_RESET_MD)
    _assert_reset_bath_pair(items)


def test_bath_phase_only_one_variant_when_partial_phase_md() -> None:
    # Each variant gates independently: with only the max-phase key present,
    # reset_bath is proposed but reset_bath_e is withheld.
    partial = {
        "bathreset_freq": 5500.0,
        "bathreset_gain": 0.7,
        "bathreset_max_phase": 12.5,
    }
    items = _bath_phase_items(with_snapshot=True, md=partial)
    mod_items = [it for it in items if isinstance(it, ModuleWriteback)]
    assert {it.target_name for it in mod_items} == {"reset_bath"}


def test_bath_phase_no_module_without_snapshot() -> None:
    # Without a cfg_snapshot only the two md items remain (no module writeback),
    # even when the bath md is complete.
    items = _bath_phase_items(with_snapshot=False, md=_BATH_RESET_MD)
    assert all(isinstance(it, MetaDictWriteback) for it in items)
    assert len(items) == 2


@pytest.mark.parametrize(
    "adapter",
    [BathFreqGainAdapter(), BathLengthAdapter(), BathPhaseAdapter()],
)
def test_bath_reset_run_without_soc_fast_fails(adapter: Any) -> None:
    ml = _make_ml()
    schema = adapter.make_default_cfg(_make_ctx(ml))

    with pytest.raises(RuntimeError, match="soc is required"):
        adapter.run(_make_req(ml), schema)


# --- rabi check (shared across all three reset types) --------------------


def test_rabi_check_round_trip() -> None:
    """make_default_cfg → validate → to_raw_dict → ml.make_cfg is wired correctly."""
    ml = _make_ml()
    adapter = RabiCheckAdapter()
    raw = _lower(adapter.make_default_cfg(_make_ctx(ml)), _make_req(ml))

    assert "modules" in raw
    assert "sweep" in raw
    modules = cast(dict[str, Any], raw["modules"])
    # Required modules are present.
    assert "rabi_pulse" in modules
    assert "tested_reset" in modules
    assert "pi_pulse" in modules
    assert "readout" in modules
    # Optional upstream reset is disabled by default (no library entry).
    assert "reset" not in modules

    sweep = cast(dict[str, Any], raw["sweep"])
    assert isinstance(sweep["gain"], SweepCfg)

    adapter.build_exp_cfg(raw, _make_req(ml))
    ml.make_cfg.assert_called_once_with(raw, RabiCheckCfg)


def test_rabi_check_capabilities_analysis_none() -> None:
    from zcu_tools.gui.app.main.adapter import AnalysisMode

    assert RabiCheckAdapter.capabilities.analysis is AnalysisMode.NONE
    assert RabiCheckAdapter.capabilities.requires_soc is True


def test_rabi_check_tested_reset_is_4_shape() -> None:
    """tested_reset accepts all four reset shapes (none/pulse/two_pulse/bath)."""
    from zcu_tools.gui.app.main.specs.reset import (
        make_bath_reset_spec,
        make_none_reset_spec,
        make_pulse_reset_spec,
        make_two_pulse_reset_spec,
    )

    spec = RabiCheckAdapter.cfg_spec()
    modules_spec = spec.fields["modules"]
    # The spec tree owns a "modules" CfgSectionSpec containing the tested_reset slot.
    tested_reset_spec = modules_spec.fields["tested_reset"]  # type: ignore[union-attr]

    allowed_types = {s.fields["type"] for s in tested_reset_spec.allowed}  # type: ignore[union-attr]
    for shape_spec in (
        make_none_reset_spec(),
        make_pulse_reset_spec(),
        make_two_pulse_reset_spec(),
        make_bath_reset_spec(),
    ):
        type_spec = shape_spec.fields["type"]
        assert type_spec in allowed_types, (
            f"Shape {type_spec} missing from tested_reset allowed list"
        )


def test_rabi_check_tested_reset_can_be_pulse_reset() -> None:
    """Round-trip with a pulse-reset tested_reset validates correctly."""
    from zcu_tools.gui.app.main.adapter import (
        CfgSectionValue,
        ModuleRefValue,
    )

    adapter = RabiCheckAdapter()
    schema = adapter.make_default_cfg(_make_ctx(_make_ml()))
    modules = schema.value.fields["modules"]
    assert isinstance(modules, CfgSectionValue)
    tested_reset = modules.fields["tested_reset"]
    # Default shape is pulse reset (the reset role's blank).
    assert isinstance(tested_reset, ModuleRefValue)
    reset_type = tested_reset.value.fields.get("type")
    assert reset_type is not None


def test_rabi_check_sweep_gain_range() -> None:
    """Default gain sweep spans 0.0 → 1.0 over 51 points."""
    from zcu_tools.gui.app.main.adapter import CfgSectionValue, SweepValue

    schema = RabiCheckAdapter().make_default_cfg(_make_ctx(_make_ml()))
    sweep = schema.value.fields["sweep"]
    assert isinstance(sweep, CfgSectionValue)
    gain = sweep.fields["gain"]
    assert isinstance(gain, SweepValue)
    assert gain.start == 0.0
    assert gain.stop == 1.0
    assert gain.expts == 51


def test_rabi_check_no_writeback() -> None:
    """NONE analysis → get_writeback_items returns empty (base default)."""
    adapter = RabiCheckAdapter()
    # WritebackRequest with dummy values; writeback is never called for NONE
    # mode by the framework, but the base must still return [] if invoked.
    items = adapter.get_writeback_items(
        WritebackRequest(
            run_result=MagicMock(),
            analyze_result=None,  # type: ignore[arg-type]
            ctx=cast(Any, MagicMock()),
        )
    )
    assert list(items) == []


def test_rabi_check_run_without_soc_fast_fails() -> None:
    ml = _make_ml()
    adapter = RabiCheckAdapter()
    schema = adapter.make_default_cfg(_make_ctx(ml))

    with pytest.raises(RuntimeError, match="soc is required"):
        adapter.run(_make_req(ml), schema)
