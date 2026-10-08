"""IRB pairing, physical recovery, persistence and ratio-estimator regressions."""

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.stop_signal import StopSignal
from zcu_tools.plotting.plots import NonPresentingHost, Plots
from zcu_tools.program.v2 import ModularProgramV2
from zcu_tools.program.v2.mocksoc import make_mock_soc
from zcu_tools.program.v2.modules.pulse import PulseCfg
from zcu_tools.program.v2.modules.readout import DirectReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg

from zcu_lab.v2.twotone.irb.analysis import analyze_irb
from zcu_lab.v2.twotone.irb.core import IRB_Exp, IRBCfg
from zcu_lab.v2.twotone.rb.program import RBModuleCfg, RBSweepCfg
from zcu_lab.v2.twotone.rb.sequence import GATE_EFFECT_MAP, TargetGate, make_seed_tables


@pytest.mark.parametrize("target", ["X90", "X180", "Y90", "Y180"])
def test_target_preserved_and_recovery_correct_at_every_depth(
    target: TargetGate,
) -> None:
    depths = np.arange(201, dtype=np.int64)
    names = ("Id", "X90", "X180", "-X90", "Y90", "Y180", "-Y90")
    for seed in (42, 71, 20261006):
        ref = make_seed_tables(seed, depths)
        gates, lengths, recovery0, recovery1 = make_seed_tables(seed, depths, target)
        # A real target pulse per depth, even when adjacent rotations could cancel.
        np.testing.assert_array_equal(np.array(lengths) - np.array(ref[1]), depths)
        for length, first, second in zip(lengths, recovery0, recovery1, strict=True):
            state = 4
            for gate in (*gates[:length], first, second):
                state = GATE_EFFECT_MAP[names[gate]][state]
            assert state == 4
        assert make_seed_tables(seed, depths, target) == (
            gates,
            lengths,
            recovery0,
            recovery1,
        )


def _cfg() -> IRBCfg:
    pulse = PulseCfg(
        ch=0, nqz=1, freq=1000.0, gain=0.2, waveform=ConstWaveformCfg(length=0.1)
    )
    return IRBCfg(
        modules=RBModuleCfg(
            X90_pulse=pulse,
            X180_pulse=pulse.with_updates(gain=0.4),
            readout=DirectReadoutCfg(ro_ch=0, ro_length=1.0, ro_freq=1000.0),
        ),
        sweep=RBSweepCfg(depth=[0, 1, 3, 20]),
        seed=42,
        n_seeds=3,
        reps=2,
        rounds=2,
        relax_delay=1.0,
    )


def test_mock_acquisition_and_fixed_four_axis_roundtrip(tmp_path: Path) -> None:
    cfg = _cfg()
    soc, soccfg = make_mock_soc()
    plots = Plots(NonPresentingHost())
    stop = StopSignal()
    try:
        with patch.object(
            ModularProgramV2,
            "acquire",
            autospec=True,
            side_effect=ModularProgramV2.acquire,
        ) as acquire:
            result = IRB_Exp().run(
                cfg,
                context=RunContext(
                    soc=soc, soccfg=soccfg, plots=plots, devices={}, cancel_signal=stop
                ),
            )
        stop.raise_if_error()
    finally:
        plots.finish(present=False)
        plots.release()
    assert result.signals.shape == (2, 3, 2, 4)
    programs = [call.args[0] for call in acquire.call_args_list]
    assert len(programs) == 12
    assert all(program.cfg_model.rounds == 1 for program in programs)
    assert len({id(program) for program in programs}) == 6
    assert [id(p) for p in programs[6:]] == [
        id(programs[i]) for i in (1, 0, 3, 2, 5, 4)
    ]
    assert np.isfinite(result.signals).all()
    assert len(np.unique(result.sub_seeds)) == 3
    np.testing.assert_array_equal(result.arms, [0, 1])
    path = tmp_path / "paired.hdf5"
    # Preserve partial data too, without converting absent arms to zero.
    result.signals[1, 2, 1] = np.nan
    IRB_Exp().save(RunRecord(cfg=cfg, result=result), path)
    loaded = IRB_Exp().load(path)
    np.testing.assert_array_equal(loaded.result.signals, result.signals)
    np.testing.assert_array_equal(loaded.result.sub_seeds, result.sub_seeds)
    assert loaded.cfg is not None
    assert loaded.cfg.target_gate == "X90"
    assert loaded.cfg.rounds == 2


def test_cancel_after_first_arm_preserves_unpaired_raw_iq() -> None:
    cfg = _cfg()
    soc, soccfg = make_mock_soc()
    plots = Plots(NonPresentingHost())
    stop = StopSignal()
    original = ModularProgramV2.acquire

    def acquire_then_stop(*args, **kwargs):
        result = original(*args, **kwargs)
        stop.set()
        return result

    try:
        with patch.object(
            ModularProgramV2, "acquire", autospec=True, side_effect=acquire_then_stop
        ):
            result = IRB_Exp().run(
                cfg,
                context=RunContext(
                    soc=soc, soccfg=soccfg, plots=plots, devices={}, cancel_signal=stop
                ),
            )
        stop.raise_if_error()
    finally:
        plots.finish(present=False)
        plots.release()
    assert np.isfinite(result.signals[0, 0, 0]).all()
    assert np.isnan(result.signals[:, :, 1]).all()
    assert np.isnan(result.signals[1:]).all()


def _signals(
    p_ref: float = 0.98, p_gate: float = 0.99
) -> tuple[np.ndarray, np.ndarray]:
    depths = np.arange(0, 151, 5, dtype=np.int64)
    rng = np.random.default_rng(7)
    signals = np.empty((2, 12, 2, len(depths)), dtype=np.complex128)
    for seed in range(12):
        amplitude = 0.6 + rng.normal(0, 0.015)
        for arm, p in enumerate((p_ref, p_ref * p_gate)):
            curve = 0.3 + amplitude * p**depths
            signals[:, seed, arm] = (
                curve + rng.normal(0, 0.0002, (2, len(depths)))
            ) * np.exp(0.7j) + 0.2j
    return depths, signals


def test_ratio_fit_bootstrap_and_incomplete_pair_exclusion() -> None:
    depths, signals = _signals()
    # An unpaired outlier must not bias either fitted arm.
    signals[1, 0, 0] = 1e6
    signals[1, 0, 1] = np.nan
    result = analyze_irb(depths, signals, bootstrap_samples=100)
    assert result.p_reference == pytest.approx(0.98, abs=1e-4)
    assert result.p_interleaved == pytest.approx(0.9702, abs=1e-4)
    assert result.gate_fidelity == pytest.approx(0.995, abs=1e-4)
    assert result.fidelity_ci_low < result.fidelity_ci_high
    assert result.n_paired_seeds == 12
    assert result.n_paired_rounds == 23
    again = analyze_irb(depths, signals, bootstrap_samples=100)
    assert (result.fidelity_ci_low, result.fidelity_ci_high) == (
        again.fidelity_ci_low,
        again.fidelity_ci_high,
    )


def test_negative_error_is_reported_without_clipping() -> None:
    depths, signals = _signals(p_gate=1.005)
    result = analyze_irb(depths, signals, bootstrap_samples=100)
    assert result.gate_error < 0
    assert result.gate_fidelity > 1
    assert "not clipped" in result.warning


def test_unpaired_or_flat_data_cannot_produce_fidelity() -> None:
    depths, signals = _signals()
    with pytest.raises(ValueError, match="contrast"):
        analyze_irb(depths, np.ones_like(signals), bootstrap_samples=100)
    signals[:, :, 1] = np.nan
    with pytest.raises(ValueError, match="two seeds"):
        analyze_irb(depths, signals, bootstrap_samples=100)
