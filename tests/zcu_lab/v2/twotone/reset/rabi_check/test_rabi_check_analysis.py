from __future__ import annotations

import numpy as np
import pytest
from qick.asm_v2 import QickParam
from zcu_tools.experiment.records import RunRecord
from zcu_tools.plotting.plots import NonPresentingHost, Plots
from zcu_tools.program.v2 import (
    Branch,
    DirectReadoutCfg,
    ModularProgramV2,
    ProgramV2Cfg,
    Pulse,
    PulseCfg,
    SweepCfg,
)
from zcu_tools.program.v2.mocksoc import make_mock_soccfg
from zcu_tools.program.v2.modules.registry import PulseRegistry
from zcu_tools.program.v2.modules.reset import NoneResetCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg

from zcu_lab.v2.twotone.reset.rabi_check.core import (
    RabiCheckCfg,
    RabiCheckExp,
    RabiCheckModuleCfg,
    RabiCheckResult,
    _rabi_check_sequence,
)
from zcu_lab.v2.twotone.reset.rabi_check.fit import fit_reset_rabi


def test_reset_check_uses_same_swept_rabi_pulse_twice() -> None:
    modules = RabiCheckModuleCfg(
        rabi_pulse=PulseCfg(
            waveform=ConstWaveformCfg(length=0.1),
            ch=0,
            nqz=1,
            freq=4000.0,
            gain=0.5,
        ),
        tested_reset=NoneResetCfg(),
        readout=DirectReadoutCfg(ro_ch=0, ro_length=1.0, ro_freq=6000.0),
    )
    sweep = SweepCfg(start=0.1, stop=0.7, step=0.1, expts=7)

    sequence = _rabi_check_sequence(modules, sweep)

    assert "pi_pulse" not in RabiCheckModuleCfg.model_fields
    first = sequence[1]
    branch = sequence[2]
    assert isinstance(first, Pulse)
    assert isinstance(branch, Branch)
    assert len(branch.branches) == 3
    assert [len(case) for case in branch.branches] == [0, 1, 2]
    second = branch.branches[2][1]
    assert isinstance(second, Pulse)
    assert first.name != second.name
    assert first.cfg is not None and second.cfg is not None
    assert isinstance(first.cfg.gain, QickParam)
    assert isinstance(second.cfg.gain, QickParam)
    assert first.cfg.gain.start == second.cfg.gain.start == sweep.start
    assert first.cfg.gain.spans == second.cfg.gain.spans == {"gain": 0.6}
    registry = PulseRegistry()
    assert registry.calc_name(first.cfg) == registry.calc_name(second.cfg)

    program = ModularProgramV2(
        make_mock_soccfg(n_gens=1, n_readouts=1),
        ProgramV2Cfg(),
        modules=sequence,
        sweep=[("reset_sel", 3), ("gain", sweep)],
    )
    assert program.pulse_registry.count == 1


def test_analyze_fits_amplitudes_on_reference_iq_axis() -> None:
    gains = np.linspace(-0.2, 1.1, 101)
    angle = 2 * np.pi * 2.3 * gains
    values = np.array(
        [
            1.0 + 0.8 * np.cos(angle + 0.2),
            0.3 + 0.05 * np.cos(angle - 0.1),
            0.7 + 0.6 * np.cos(angle + 0.5) + 0.12 * np.sin(2 * angle),
        ]
    )
    # Large perpendicular branch offsets must not redefine the readout axis.
    signals = (values + 1j * np.array([0.0, 4.0, -3.0])[:, None]) * np.exp(0.7j)
    result = RabiCheckResult(gains=gains, signals=signals)
    source = RunRecord[RabiCheckCfg, RabiCheckResult](cfg=None, result=result)
    plots = Plots(NonPresentingHost())
    try:
        fit = RabiCheckExp().analyze(source, None, plots=plots)
        figure = plots["fit"]
        assert fit.frequency == pytest.approx(2.3, abs=1e-6)
        assert fit.before.amplitude == pytest.approx(0.8)
        assert fit.after.amplitude == pytest.approx(0.6)
        assert fit.contrast_ratio == pytest.approx(0.75)
        assert fit.before.phase_deg == pytest.approx(np.degrees(0.2), abs=1e-4)
        assert fit.after.phase_deg == pytest.approx(np.degrees(0.5), abs=1e-4)
        assert fit.phase_difference_deg == pytest.approx(np.degrees(0.3), abs=1e-4)
        assert fit.reset.amplitude == pytest.approx(0.05)
        assert fit.reset.offset == pytest.approx(0.3)
        assert fit.after.second_harmonic_amplitude == pytest.approx(0.12)
        assert (
            max(branch.residual_rms for branch in (fit.before, fit.reset, fit.after))
            < 1e-6
        )
        assert len(figure.axes) == 2
        for index, line in enumerate(figure.axes[0].lines[::2]):
            np.testing.assert_allclose(
                np.asarray(line.get_ydata(), dtype=np.float64),
                values[index],
                atol=1e-12,
            )
        np.testing.assert_array_equal(result.signals, signals)
    finally:
        plots.finish(present=False)
        plots.release()


def test_dephased_population_memory_retains_second_harmonic() -> None:
    gains = np.linspace(0, 1, 101)
    z = np.cos(2 * np.pi * 2 * gains)
    fit = fit_reset_rabi(gains, np.array([z, 0.7 + 0.2 * z, (0.7 + 0.2 * z) * z]))
    assert fit.reset.amplitude == pytest.approx(0.2)
    assert fit.after.amplitude == pytest.approx(0.7)
    assert fit.after.second_harmonic_amplitude == pytest.approx(0.1)


def test_zero_after_contrast_has_no_phase_and_flat_reset_is_valid() -> None:
    gains = np.linspace(0, 1, 101)
    before = np.cos(2 * np.pi * 2 * gains)
    fit = fit_reset_rabi(
        gains, np.array([before, np.ones_like(gains), np.ones_like(gains)])
    )
    assert fit.after.amplitude < 1e-12
    assert fit.after.phase_deg is None
    assert fit.phase_difference_deg is None
    assert fit.reset.amplitude < 1e-12


def test_noisy_partial_descending_sweep_preserves_contrast_and_residuals() -> None:
    gains = np.linspace(1, -0.1, 121)
    angle = 2 * np.pi * 2.4 * gains
    signals = np.array(
        [np.cos(angle), 0.3 + 0.04 * np.cos(angle), 0.7 * np.cos(angle + 0.3)]
    )
    signals += np.random.default_rng(42).normal(0, 0.01, signals.shape)
    signals[0, 20] = np.nan
    signals[1, :10] = np.nan
    signals[2, 40:42] = np.nan
    fit = fit_reset_rabi(gains, signals)
    assert fit.frequency == pytest.approx(2.4, abs=0.003)
    assert fit.contrast_ratio == pytest.approx(0.7, abs=0.01)
    assert fit.phase_difference_deg == pytest.approx(np.degrees(0.3), abs=1)
    assert 0.007 < fit.after.residual_rms < 0.013


@pytest.mark.parametrize(
    "case, message",
    [
        ("flat", "no resolvable"),
        ("short", "six finite gains"),
        ("duplicate", "unique"),
        ("infinite", "infinities"),
        ("missing_after", "six finite after"),
        ("missing_reset", "Too few finite"),
        ("undersampled", "second harmonic"),
    ],
)
def test_invalid_or_unresolved_fits_fail_explicitly(case: str, message: str) -> None:
    gains = np.linspace(0, 1, 51)
    signals = np.tile(np.cos(2 * np.pi * 2 * gains), (3, 1))
    if case == "flat":
        signals[0] = 1
    elif case == "short":
        gains, signals = gains[:3], signals[:, :3]
    elif case == "duplicate":
        gains[1] = gains[0]
    elif case == "infinite":
        signals[1, 0] = np.inf
    elif case == "missing_after":
        signals[2] = np.nan
    elif case == "missing_reset":
        signals[1] = np.nan
    elif case == "undersampled":
        signals = np.tile(np.cos(2 * np.pi * 15 * gains), (3, 1))
    with pytest.raises(ValueError, match=message):
        fit_reset_rabi(gains, signals)
