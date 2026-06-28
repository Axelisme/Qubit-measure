"""Cross-experiment inject -> recover integration tests for the SimEngine.

This is the value proof of mocksim: inject physical parameters via
:class:`SimParams`, run a *real* experiment class (its ``run`` + ``analyze``,
not a hand-built program) on a sim mock soc, and assert the analyze fit recovers
the injected physics within tolerance.  Unlike ``test_engine.py`` (which builds
``ModularProgramV2`` directly and only checks feature *shape*), every test here
drives the full ``experiment/v2`` path end to end and checks recovered *values*.

Phase-1 covers freq / amp_rabi / len_rabi / T1 / T2-Ramsey / T2-echo recovery;
Phase 2 adds the dephasing-model proof: with ``T2 != T2_star`` the echo recovers
the homogeneous ``T2`` (and is insensitive to the inhomogeneous rate Gamma) while
Ramsey recovers ``T2_star``, and Ramsey decays faster than echo — exactly the
Lorentzian quasi-static detune model the engine averages over.

Operating point (R-3: fixed reduced flux = 1.0)
-----------------------------------------------
The engine pins the operating point at reduced flux ``Phi/Phi0 = 1.0`` (R-3); it
no longer derives flux from the cfg ``dev`` map.  At EJ/EC/EL = 3.0/0.9/0.5 the
fluxonium 0->1 frequency there is ~4086 MHz.  The mock soccfg's gen f_dds is
12288 MHz, so this f01 sits well below f_dds and the analyzer's absolute frequency
axis reports it *un-folded* (folding is a ``f mod f_dds`` effect; see sim/README).
So ``test_freq_recovers_f_qubit`` asserts the *absolute* recovered f_qubit
directly; the other tests still rely only on *relative* structure (detuning, decay
times, gain scaling, fringe frequency) which is folding-invariant regardless.

The engine drives the qubit at the f_qubit it computes from the same SimParams
(via FluxoniumPredictor at flux 1.0), and the readout sits near ``rf_g`` to
maximise |g>/|e> contrast — i.e. each test plays an experimenter who has already
located the qubit and resonator, exactly as the real path requires.

len_rabi note: the mock soccfg's const/flat_top pulse-length *register* grid is
too coarse for a hard length sweep to compile, so len_rabi is driven with a
gauss pulse (the soft-sweep path that recompiles per length).  A gauss envelope's
rotation angle is area-weighted, so the absolute const formula
``pi_len == pi_gain_len/gain`` does not hold; instead the gain *scaling* law is
asserted (Rabi freq proportional to gain), which is the injection-faithful
invariant for any envelope.

dephasing note: the engine averages the deterministic per-point signal over a
Lorentzian quasi-static detune ensemble (HWHM Gamma = ``1/T2_star - 1/T2``).  A
Ramsey free evolution accumulates the un-refocused ensemble phase -> an extra
``exp(-Gamma*t)`` decay, so a Ramsey fit recovers ``T2_star``; an echo pi pulse
refocuses every static detune, so an echo fit recovers the homogeneous ``T2`` and
is insensitive to Gamma.  This is sequence-agnostic: the engine never identifies
the pulse sequence, the refocusing emerges from the pi flip plus the ensemble
average alone.
"""

from __future__ import annotations

import matplotlib

# Headless backend: these tests build figures via the experiments' analyze() but
# never display them (the autouse _close_matplotlib_figures fixture cleans up).
matplotlib.use("Agg")

import numpy as np
import pytest
from zcu_tools.experiment.v2.lookback import (
    LookbackCfg,
    LookbackExp,
    LookbackModuleCfg,
)
from zcu_tools.experiment.v2.singleshot.ge import GE_Cfg, GE_Exp, GEModuleCfg
from zcu_tools.experiment.v2.twotone.freq import FreqCfg, FreqExp, FreqSweepCfg
from zcu_tools.experiment.v2.twotone.rabi.amp_rabi import (
    AmpRabiCfg,
    AmpRabiExp,
    AmpRabiSweepCfg,
)
from zcu_tools.experiment.v2.twotone.rabi.len_rabi import (
    LenRabiCfg,
    LenRabiExp,
    LenRabiSweepCfg,
)
from zcu_tools.experiment.v2.twotone.time_domain.t1 import (
    T1Cfg,
    T1Exp,
    T1ModuleCfg,
    T1SweepCfg,
)
from zcu_tools.experiment.v2.twotone.time_domain.t2echo import (
    T2EchoCfg,
    T2EchoExp,
    T2EchoModuleCfg,
    T2EchoSweepCfg,
)
from zcu_tools.experiment.v2.twotone.time_domain.t2ramsey import (
    T2RamseyCfg,
    T2RamseyExp,
    T2RamseyModuleCfg,
    T2RamseySweepCfg,
)
from zcu_tools.program.v2 import SweepCfg
from zcu_tools.program.v2.mocksoc import make_mock_soc
from zcu_tools.program.v2.modules.pulse import PulseCfg
from zcu_tools.program.v2.modules.readout import DirectReadoutCfg, PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg, GaussWaveformCfg
from zcu_tools.program.v2.sim import SimParams
from zcu_tools.program.v2.sim.engine import _FULL_SCALE
from zcu_tools.program.v2.sim.readout import resonator_freqs, s21
from zcu_tools.program.v2.twotone import TwoToneModuleCfg
from zcu_tools.simulate.fluxonium.predict import FluxoniumPredictor

# Fixed operating point: reduced flux = 1.0 (R-3, matches the engine constant).
# T1/T2 are a few µs so decay/dephasing are resolvable over modest sweeps; snr is
# generous and the seed is fixed so the fits are reproducible.
_OPERATING_FLUX = 1.0

_SIM = SimParams(
    EJ=3.0,
    EC=0.9,
    EL=0.5,
    flux_period=1.0,
    flux_half=0.0,
    flux_bias=0.1,
    T1=20.0,
    T2=10.0,
    T2_star=10.0,  # T2_star == T2 => gamma=0 (pure homogeneous; preserves existing physics)
    bare_rf=7.2,
    g=0.08,
    Ql=5000.0,
    Qi=50000.0,
    snr=300.0,
    pi_gain_len=0.4,
    seed=12345,
)


def _predictor() -> FluxoniumPredictor:
    return FluxoniumPredictor(
        params=(_SIM.EJ, _SIM.EC, _SIM.EL),
        flux_half=_SIM.flux_half,
        flux_period=_SIM.flux_period,
        flux_bias=_SIM.flux_bias,
    )


def _f_qubit_mhz() -> float:
    """The qubit 0->1 frequency (MHz) the engine sees at the fixed operating flux.

    The engine pins reduced flux = 1.0 (R-3) and feeds ``predict_freq`` a *device
    value*, so map the fixed flux back through the predictor's affine alignment
    (``flux_to_value``) exactly as the engine does — this is the true f_qubit the
    engine drives at, ~4086 MHz, which sits below f_dds (12288 MHz) so the analyzer
    reports it un-folded.
    """

    predictor = _predictor()
    return float(predictor.predict_freq(predictor.flux_to_value(_OPERATING_FLUX)))


def _rf_g_mhz() -> float:
    """Ground-state dressed resonator frequency (MHz) at the fixed operating flux.

    Reading out near rf_g maximises |g>/|e> contrast so the time-domain decays
    and the Rabi oscillation are visible in the readout magnitude.
    """

    rf_g, _rf_e = resonator_freqs(_SIM, _OPERATING_FLUX)
    return rf_g * 1e3


def _readout() -> DirectReadoutCfg:
    return DirectReadoutCfg(ro_ch=0, ro_length=1.0, ro_freq=_rf_g_mhz())


def _twotone_modules(qub_pulse: PulseCfg) -> TwoToneModuleCfg:
    return TwoToneModuleCfg(
        reset=None, init_pulse=None, qub_pulse=qub_pulse, readout=_readout()
    )


def _sim_dephasing(*, T2: float, T2_star: float) -> SimParams:
    """``_SIM`` clone with a chosen homogeneous/inhomogeneous coherence split.

    ``T2 != T2_star`` makes the inhomogeneous rate Gamma = ``1/T2_star - 1/T2``
    nonzero, which is what separates the echo (recovers ``T2``) from the Ramsey
    (recovers ``T2_star``) recovery.  All other physics (operating point, readout,
    snr, seed) is inherited so the only difference vs the Phase-1 tests is the
    dephasing split.  ``0 < T2_star <= T2 <= 2*T1`` is enforced by SimParams.
    """

    return _SIM.model_copy(update={"T2": T2, "T2_star": T2_star})


# --------------------------------------------------------------- twotone freq


def test_freq_recovers_f_qubit() -> None:
    """twotone freq fit recovers the injected absolute f_qubit.

    Injected: EJ/EC/EL + flux 1.0 -> a definite (true) f_qubit ~4086 MHz via
    FluxoniumPredictor.  The engine drives at that true frequency, which sits below
    the mock gen f_dds (12288 MHz), so the analyzer's ``sweep2array`` frequency axis
    reports it *un-folded*.  The Lorentzian fit must therefore land on the true
    absolute f_qubit to within a few sweep steps (step = 5 MHz).
    """

    soc, soccfg = make_mock_soc(sim=_SIM)
    f_qubit = _f_qubit_mhz()

    qub_pulse = PulseCfg(
        ch=0,
        nqz=1,
        gain=0.03,  # weak drive -> a narrow, well-localised qubit peak
        freq=f_qubit,
        phase=0.0,
        waveform=ConstWaveformCfg(length=2.0),
    )
    cfg = FreqCfg(
        reps=200,
        rounds=2,
        modules=_twotone_modules(qub_pulse),
        sweep=FreqSweepCfg(
            freq=SweepCfg(
                start=f_qubit - 200.0, stop=f_qubit + 200.0, expts=81, step=400.0 / 80
            )
        ),
    )

    exp = FreqExp()
    result = exp.run(soc, soccfg, cfg)
    fit_freq, _freq_err, _fwhm, _fwhm_err, _fig = exp.analyze(result, model_type="lor")

    # f_qubit < f_dds so the analyzer axis is un-folded: the recovered peak must
    # land on the true injected f_qubit to within a few sweep steps (step = 5 MHz).
    assert fit_freq == pytest.approx(f_qubit, abs=10.0)


# --------------------------------------------------------------- amp_rabi


def test_amp_rabi_recovers_pi_gain() -> None:
    """amp_rabi fit recovers the pi gain set by pi_gain_len at fixed length L.

    Injected: pi_gain_len (the gain*length product for a pi rotation).  With a
    fixed const length L, an exact pi rotation needs gain == pi_gain_len / L, so
    the fitted pi gain must equal that ratio.
    """

    soc, soccfg = make_mock_soc(sim=_SIM)
    f_qubit = _f_qubit_mhz()

    length = 0.5
    expected_pi_gain = _SIM.pi_gain_len / length  # 0.4 / 0.5 == 0.8

    qub_pulse = PulseCfg(
        ch=0,
        nqz=1,
        gain=0.0,  # swept below
        freq=f_qubit,
        phase=0.0,
        waveform=ConstWaveformCfg(length=length),
    )
    cfg = AmpRabiCfg(
        reps=120,
        rounds=2,
        modules=_twotone_modules(qub_pulse),
        sweep=AmpRabiSweepCfg(
            gain=SweepCfg(start=0.0, stop=1.6, expts=60, step=1.6 / 59)
        ),
    )

    exp = AmpRabiExp()
    result = exp.run(soc, soccfg, cfg)
    pi_gain, _pi_gain_err, _pi2_gain, _pi2_gain_err, _fig = exp.analyze(result)

    # Recovered pi gain == pi_gain_len / length.
    assert pi_gain == pytest.approx(expected_pi_gain, rel=0.05)


# --------------------------------------------------------------- len_rabi


def test_len_rabi_recovers_gain_scaling() -> None:
    """len_rabi Rabi frequency scales linearly with the drive gain.

    The mock soccfg's const length register is too coarse for a hard length
    sweep to compile, so this uses a gauss pulse (soft-sweep path).  A gauss
    envelope is area-weighted, so the absolute const formula pi_len ==
    pi_gain_len/gain does not apply; the injection-faithful invariant that *does*
    hold for any envelope is that the Rabi frequency is proportional to gain
    (Omega ∝ gain).  Doubling the gain must double the fitted Rabi frequency and
    halve the fitted pi length.
    """

    f_qubit = _f_qubit_mhz()

    def _run(gain: float) -> tuple[float, float]:
        soc, soccfg = make_mock_soc(sim=_SIM)
        qub_pulse = PulseCfg(
            ch=0,
            nqz=1,
            gain=gain,
            freq=f_qubit,
            phase=0.0,
            waveform=GaussWaveformCfg(length=0.2, sigma=0.05),
        )
        cfg = LenRabiCfg(
            reps=150,
            rounds=2,
            modules=_twotone_modules(qub_pulse),
            sweep=LenRabiSweepCfg(
                length=SweepCfg(start=0.05, stop=3.0, expts=25, step=(3.0 - 0.05) / 24)
            ),
        )
        exp = LenRabiExp()
        result = exp.run(soc, soccfg, cfg)
        pi_len, _pi_len_err, _pi2_len, _pi2_len_err, rabi_freq, _rabi_f_err, _fig = (
            exp.analyze(result, decay=False)
        )
        return pi_len, rabi_freq

    pi_len_lo, freq_lo = _run(0.4)
    pi_len_hi, freq_hi = _run(0.8)

    # Doubling gain doubles the Rabi frequency (Omega ∝ gain) ...
    assert freq_hi / freq_lo == pytest.approx(2.0, rel=0.05)
    # ... and halves the pi length (pi_len ∝ 1/gain).
    assert pi_len_hi / pi_len_lo == pytest.approx(0.5, rel=0.05)


# --------------------------------------------------------------- T1


def test_t1_recovers_t1() -> None:
    """T1 fit recovers the injected SimParams.T1.

    Injected: sim.T1 = 20 µs.  A pi pulse excites the qubit, a swept delay lets
    it relax, and T1Exp.analyze fits the exponential decay; the fitted T1 must
    equal the injected value.
    """

    soc, soccfg = make_mock_soc(sim=_SIM)
    f_qubit = _f_qubit_mhz()

    # gain * length == pi_gain_len (1.0 * 0.4) is an exact pi rotation.
    pi_pulse = PulseCfg(
        ch=0,
        nqz=1,
        gain=1.0,
        freq=f_qubit,
        phase=0.0,
        waveform=ConstWaveformCfg(length=0.4),
    )
    cfg = T1Cfg(
        reps=120,
        rounds=2,
        modules=T1ModuleCfg(reset=None, pi_pulse=pi_pulse, readout=_readout()),
        sweep=T1SweepCfg(
            length=SweepCfg(start=0.0, stop=80.0, expts=30, step=80.0 / 29)
        ),
    )

    exp = T1Exp()
    result = exp.run(soc, soccfg, cfg)
    t1, _t1err, _fig = exp.analyze(result)

    # Recovered T1 == injected sim.T1 (20 µs).
    assert t1 == pytest.approx(_SIM.T1, rel=0.05)


# --------------------------------------------------------------- T2 Ramsey / echo runners
#
# Phase 2 injects an asymmetric coherence split (T2 != T2_star) so the two
# experiments recover *different* times: Ramsey -> T2_star, echo -> T2.  The run
# logic is factored into helpers so the value-recover tests and the cross-checks
# (Gamma-insensitivity, Ramsey-faster-than-echo) share one definition.

# Asymmetric split: 0 < T2_star (8) <= T2 (15) <= 2*T1 (40).  Gamma = 1/8 - 1/15
# = 0.0583 /µs, large enough to separate Ramsey from echo at this snr/seed.
_T2_INJECT = 15.0
_T2_STAR_INJECT = 8.0


def _run_ramsey(sim: SimParams, detune: float = 2.0) -> tuple[float, float, float]:
    """Run T2 Ramsey end to end; return (fitted_decay, fitted_detune, true_detune)."""

    soc, soccfg = make_mock_soc(sim=sim)
    f_qubit = _f_qubit_mhz()
    pi2_pulse = PulseCfg(
        ch=0,
        nqz=1,
        gain=1.0,
        freq=f_qubit,
        phase=0.0,
        # gain * length == pi_gain_len / 2 (1.0 * 0.2) is a pi/2 rotation.
        waveform=ConstWaveformCfg(length=0.2),
    )
    cfg = T2RamseyCfg(
        reps=120,
        rounds=2,
        modules=T2RamseyModuleCfg(reset=None, pi2_pulse=pi2_pulse, readout=_readout()),
        sweep=T2RamseySweepCfg(
            length=SweepCfg(start=0.0, stop=12.0, expts=100, step=12.0 / 99)
        ),
    )
    exp = T2RamseyExp()
    # true_detune is the detune after length rounding; the fringe fit recovers it.
    result = exp.run(soc, soccfg, cfg, detune=detune)
    true_detune = result.true_activate_detune
    t2r, _t2rerr, fit_detune, _detune_err, _fig = exp.analyze(result)
    return t2r, fit_detune, true_detune


def _run_echo(sim: SimParams) -> float:
    """Run T2 echo (on resonance) end to end; return the fitted homogeneous T2."""

    soc, soccfg = make_mock_soc(sim=sim)
    f_qubit = _f_qubit_mhz()
    pi2_pulse = PulseCfg(
        ch=0,
        nqz=1,
        gain=1.0,
        freq=f_qubit,
        phase=0.0,
        waveform=ConstWaveformCfg(length=0.2),  # pi/2
    )
    pi_pulse = PulseCfg(
        ch=0,
        nqz=1,
        gain=1.0,
        freq=f_qubit,
        phase=0.0,
        waveform=ConstWaveformCfg(length=0.4),  # pi
    )
    # reps is high (2000) because the engine now draws a per-shot Bernoulli(P_e)
    # so the accumulated readout carries genuine shot noise ~ sqrt(P_e(1-P_e)/reps).
    # The echo decay is slow (T2 ~ 15 µs) with little contrast in its tail, so the
    # exp fit is sensitive to that shot noise; reps=2000 averages it down enough to
    # recover T2 (snr — the Gaussian readout noise — does not help here, only reps
    # suppresses the Bernoulli shot noise).
    cfg = T2EchoCfg(
        reps=2000,
        rounds=2,
        modules=T2EchoModuleCfg(
            reset=None, pi2_pulse=pi2_pulse, pi_pulse=pi_pulse, readout=_readout()
        ),
        sweep=T2EchoSweepCfg(
            length=SweepCfg(start=0.0, stop=30.0, expts=40, step=30.0 / 39)
        ),
    )
    exp = T2EchoExp()
    # Echo runs on resonance (detune=0); the pi pulse refocuses the static detune
    # regardless, so the engine's ensemble average leaves only the homogeneous T2.
    result, _true_detune = exp.run(soc, soccfg, cfg, detune=0.0)
    t2e, _t2eerr, _detune, _detune_err, _fig = exp.analyze(result, fit_method="decay")
    return t2e


def test_t2ramsey_recovers_t2_star_and_detuning() -> None:
    """T2 Ramsey fit recovers the injected T2_star (not T2) and the detuning.

    Injected: T2 = 15 µs, T2_star = 8 µs (Gamma > 0).  Ramsey does not refocus the
    Lorentzian static detune, so its envelope decays at the *inhomogeneous* rate
    -> the fitted decay must equal sim.T2_star (the Lorentzian FID is a pure
    exponential, so the analyzer's exp fit lands on it).  The fringe frequency
    must equal the configured (rounding-corrected) detuning.
    """

    sim = _sim_dephasing(T2=_T2_INJECT, T2_star=_T2_STAR_INJECT)
    t2r, fit_detune, true_detune = _run_ramsey(sim, detune=2.0)

    # Recovered decay == injected sim.T2_star (8 µs), NOT T2 (15 µs).
    assert t2r == pytest.approx(sim.T2_star, rel=0.15)
    # Recovered fringe frequency == the configured (rounding-corrected) detuning.
    assert fit_detune == pytest.approx(true_detune, rel=0.05)


# --------------------------------------------------------------- T2 Echo


def test_t2echo_recovers_homogeneous_t2() -> None:
    """T2 echo recovers the injected homogeneous T2 (not T2_star).

    Injected: T2 = 15 µs, T2_star = 8 µs.  The echo pi pulse refocuses every
    static detune in the Lorentzian ensemble, so the surviving decay is the
    homogeneous T2; the fit must land on sim.T2, well above sim.T2_star.  End to
    end this proves the echo recovery through the genuine run->analyze pipeline.
    """

    sim = _sim_dephasing(T2=_T2_INJECT, T2_star=_T2_STAR_INJECT)
    t2e = _run_echo(sim)

    # Recovered decay == injected sim.T2 (15 µs), NOT T2_star (8 µs).
    assert t2e == pytest.approx(sim.T2, rel=0.1)


def test_t2echo_recovery_insensitive_to_gamma() -> None:
    """Echo recovers the same T2 across two different T2_star (= two Gamma).

    Fixing the homogeneous T2 and sweeping T2_star changes only the inhomogeneous
    rate Gamma, which the echo pi pulse refocuses away.  The end-to-end echo
    recovery must therefore barely move between the two splits — the load-bearing
    proof that the model's refocusing is real and not an artifact of one Gamma.
    """

    t2e_a = _run_echo(_sim_dephasing(T2=_T2_INJECT, T2_star=8.0))
    t2e_b = _run_echo(_sim_dephasing(T2=_T2_INJECT, T2_star=5.0))

    # Both recover T2 = 15 µs ...
    assert t2e_a == pytest.approx(_T2_INJECT, rel=0.1)
    assert t2e_b == pytest.approx(_T2_INJECT, rel=0.1)
    # ... and the two recoveries agree with each other despite different Gamma.
    assert t2e_a == pytest.approx(t2e_b, rel=0.05)


def test_ramsey_decays_faster_than_echo() -> None:
    """Same params: Ramsey recovers a shorter coherence time than echo.

    With T2 > T2_star the un-refocused Ramsey decay (-> T2_star) is faster than the
    refocused echo decay (-> T2), so the recovered Ramsey time must be strictly
    below the recovered echo time on the identical operating point.
    """

    sim = _sim_dephasing(T2=_T2_INJECT, T2_star=_T2_STAR_INJECT)
    t2r, _fit_detune, _true_detune = _run_ramsey(sim, detune=2.0)
    t2e = _run_echo(sim)

    assert t2r < t2e


# --------------------------------------------------------------- lookback (D2)


def test_lookback_recovers_timefly_as_trig_offset() -> None:
    """Lookback run + analyze recover the injected timeFly as the trig_offset.

    Injected: ``sim.timeFly`` (the readout time of flight).  The decimated model A
    places the readout envelope at program-time ``timeFly``, so the trace is ~0
    before it and rises into the readout window after — exactly the rising edge
    ``LookbackExp.analyze`` locates.  The recovered offset must therefore land on
    ``timeFly`` (here 0.5 µs) within a tolerance set by the decimated sample
    spacing (~3.3 ns) plus the analyze ratio threshold.
    """

    soc, soccfg = make_mock_soc(sim=_SIM)

    ro_length = 2.0
    ro_pulse = PulseCfg(
        ch=0,
        nqz=1,
        gain=1.0,
        freq=_rf_g_mhz(),
        phase=0.0,
        waveform=ConstWaveformCfg(length=ro_length),
    )
    readout = PulseReadoutCfg(
        pulse_cfg=ro_pulse,
        ro_cfg=DirectReadoutCfg(
            ro_ch=0, ro_length=ro_length, ro_freq=_rf_g_mhz(), trig_offset=0.0
        ),
    )
    cfg = LookbackCfg(
        reps=1,
        rounds=2,
        modules=LookbackModuleCfg(reset=None, init_pulse=None, readout=readout),
    )

    exp = LookbackExp()
    result = exp.run(soc, soccfg, cfg)
    offset, _fig = exp.analyze(result, plot_fit=True)

    # The rising edge sits at program-time == timeFly; analyze returns the last
    # sub-threshold time before the magnitude peak, i.e. just before timeFly.
    assert offset == pytest.approx(_SIM.timeFly, abs=0.1)


# --------------------------------------------------------------- singleshot GE


def _ge_readout() -> PulseReadoutCfg:
    """A PulseReadout at rf_g (GE_Exp needs a PulseReadout, not a DirectReadout)."""

    ro = _rf_g_mhz()
    return PulseReadoutCfg(
        pulse_cfg=PulseCfg(
            ch=0,
            nqz=1,
            gain=0.1,
            freq=ro,
            phase=0.0,
            waveform=ConstWaveformCfg(length=1.0),
        ),
        ro_cfg=DirectReadoutCfg(ro_ch=0, ro_length=1.0, ro_freq=ro, trig_offset=0.0),
    )


def _run_ge(
    snr: float, shots: int = 5000
) -> tuple[float, np.ndarray, complex, complex]:
    """Run GE_Exp end to end on a low-snr sim soc; return the recovered analysis.

    Returns ``(fidelity, populations, g_center, e_center)`` from
    ``GE_Exp.analyze(backend='pca')``.  ``snr`` is lowered (the DEFAULT snr=300
    fully separates the blobs so the fidelity is trivially ~1); a small snr makes
    the |g>/|e> blobs overlap so the discrimination fidelity is meaningful.
    """

    sim = _SIM.model_copy(update={"snr": snr})
    soc, soccfg = make_mock_soc(sim=sim)
    f_qubit = _f_qubit_mhz()

    # pi probe (gain*length == pi_gain_len) prepares |e> on the with-probe scan.
    probe = PulseCfg(
        ch=0,
        nqz=1,
        gain=1.0,
        freq=f_qubit,
        phase=0.0,
        waveform=ConstWaveformCfg(length=_SIM.pi_gain_len),
    )
    cfg = GE_Cfg(
        reps=1,
        rounds=1,
        shots=shots,
        modules=GEModuleCfg(
            reset=None, init_pulse=None, probe_pulse=probe, readout=_ge_readout()
        ),
    )
    exp = GE_Exp()
    result = exp.run(soc, soccfg, cfg)
    fid, pops, fit, _fig = exp.analyze(result, backend="pca")
    return fid, pops, fit["g_center"], fit["e_center"]


def _expected_ge_centers() -> tuple[complex, complex]:
    """The |g>/|e> blob centres in GE analyze (avgiq) units.

    GE_Exp divides the raw per-shot acc_buf by the compiled readout length.  The
    test readout uses a full-window const PulseReadout with gain 0.1, so the
    integrated raw centre normalizes to ``_FULL_SCALE * 0.1 * S21(rf_g; rf_X)``.
    """

    rf_g, rf_e = resonator_freqs(_SIM, _OPERATING_FLUX)
    freqs = np.array([_rf_g_mhz() * 1e-3], dtype=np.float64)  # probe at rf_g
    readout_gain = 0.1
    g = _FULL_SCALE * readout_gain * complex(s21(_SIM, freqs, rf_g)[0])
    e = _FULL_SCALE * readout_gain * complex(s21(_SIM, freqs, rf_e)[0])
    return g, e


def test_ge_recovers_centers_population_and_fidelity() -> None:
    """GE_Exp run + analyze recover the injected blob centres, populations, fidelity.

    A low snr (the blobs overlap) drives a genuine PCA + histogram discrimination
    through the real GE_Exp.run -> analyze path:

      - the recovered g_center / e_center land on the gain-scaled deterministic
        blob centres (the two per-shot Bernoulli outcomes the engine produces),
      - the populations track the preparation: the no-probe scan is mostly |g>,
        the pi-probe scan is mostly |e>,
      - the fidelity is a real (non-trivial) discrimination value in (0.5, 1.0)
        and improves with snr (less blob overlap).
    """

    g_expected, e_expected = _expected_ge_centers()
    blob_dist = abs(g_expected - e_expected)

    fid_lo, pops_lo, g_lo, e_lo = _run_ge(snr=5.0)
    fid_hi, pops_hi, g_hi, e_hi = _run_ge(snr=10.0)

    # Recovered centres land on the deterministic blob centres (assignment of the
    # fit's g/e to the physical g/e may flip, so match as an unordered pair).
    def _matches(g_fit: complex, e_fit: complex) -> bool:
        direct = abs(g_fit - g_expected) + abs(e_fit - e_expected)
        flipped = abs(g_fit - e_expected) + abs(e_fit - g_expected)
        return min(direct, flipped) < 0.3 * blob_dist

    assert _matches(g_hi, e_hi)

    # Populations track the preparation: pops[0] is the no-probe (ground) scan,
    # pops[1] the pi-probe (excited) scan; each row is [P(measured g), P(measured e)].
    # The no-probe row is mostly ground, the pi-probe row mostly excited.
    assert pops_hi[0, 0] > pops_hi[0, 1]  # no-probe: more ground than excited
    assert pops_hi[1, 1] > pops_hi[1, 0]  # pi-probe: more excited than ground

    # Fidelity is a real discrimination value in (0.5, 1.0) and improves with snr.
    assert 0.5 < fid_lo < 1.0
    assert 0.5 < fid_hi <= 1.0
    assert fid_hi > fid_lo
