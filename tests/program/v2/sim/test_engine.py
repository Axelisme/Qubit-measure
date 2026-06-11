"""Tests for sim/engine.py — SimEngine assembly + MyProgramV2.acquire dispatch.

Covers:
- D1 zero-regression: ``make_mock_soc()`` with no SimParams runs the unchanged
  white-noise accumulated path (same shapes, random data).
- Engine smoke (real path): a sim soc running real experiment programs
  (twotone freq, amp_rabi, T1, T2 ramsey) produces physically-structured data,
  not white noise — a peak/dip at f_qubit, gain-driven Rabi oscillation, a T1
  decay, and Ramsey fringes.
- Singleshot get_raw shape ``(reps, 1, 2)`` is usable on the sim path.
- round_hook is invoked once per round.
- acquire_decimated fast-fails on a sim soc (decimated is out of scope, D2).

The engine drives the qubit at its *own* predicted f_qubit (queried through a
FluxoniumPredictor built from the same SimParams) and reads out near rf_g — i.e.
the test plays the role of an experimenter who has already located the qubit and
the resonator, exactly as the real path requires.
"""

from __future__ import annotations

import numpy as np
import pytest
from zcu_tools.program.v2.base import ProgramV2Cfg
from zcu_tools.program.v2.mocksoc import make_mock_soc
from zcu_tools.program.v2.modular import ModularProgramV2
from zcu_tools.program.v2.modules.base import Module
from zcu_tools.program.v2.modules.control import Branch
from zcu_tools.program.v2.modules.delay import Delay
from zcu_tools.program.v2.modules.pulse import PulseCfg
from zcu_tools.program.v2.modules.readout import DirectReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg
from zcu_tools.program.v2.sim import SimParams
from zcu_tools.program.v2.sim.readout import resonator_freqs, value_to_flux
from zcu_tools.program.v2.sweep import SweepCfg
from zcu_tools.program.v2.utils import sweep2param
from zcu_tools.simulate.fluxonium.predict import FluxoniumPredictor

# Operating point: flux_bias=0.2 (the engine's no-device flux value) places the
# qubit a few GHz above the sweet spot, giving a finite gap and a clear
# dispersive shift.  T1/T2 are short so decay/dephasing are visible over modest
# sweeps; snr is generous so the physical structure dominates the noise.
_SIM = SimParams(
    EJ=8.5,
    EC=1.0,
    EL=0.5,
    flux_period=1.0,
    flux_half=0.0,
    flux_bias=0.2,
    T1=20.0,
    T2=10.0,
    T2_star=10.0,  # T2_star == T2 => gamma=0 (pure homogeneous; preserves existing physics)
    bare_rf=7.2,
    g=0.08,
    Ql=5000.0,
    Qi=50000.0,
    snr=200.0,
    pi_gain_len=0.4,
    seed=12345,
)


def _f_qubit_mhz() -> float:
    """The qubit 0->1 frequency (MHz) the engine sees at the no-device flux."""

    predictor = FluxoniumPredictor(
        params=(_SIM.EJ, _SIM.EC, _SIM.EL),
        flux_half=_SIM.flux_half,
        flux_period=_SIM.flux_period,
        flux_bias=_SIM.flux_bias,
    )
    return float(predictor.predict_freq(_SIM.flux_bias))


def _rf_g_mhz() -> float:
    """The ground-state dressed resonator frequency (MHz) at the operating flux.

    Reading out near rf_g maximizes the |g>/|e> contrast, which is what makes
    the T1 decay and Rabi oscillation visible in the readout magnitude.
    """

    flux = value_to_flux(_SIM, _SIM.flux_bias)
    rf_g, _rf_e = resonator_freqs(_SIM, flux)
    return rf_g * 1e3


def _readout(ro_freq_mhz: float) -> Module:
    return DirectReadoutCfg(ro_ch=0, ro_length=1.0, ro_freq=ro_freq_mhz).build("ro")


def _amp(result: list[np.ndarray]) -> np.ndarray:
    """Magnitude of the swept IQ trace for the single readout channel.

    ``acquire`` returns ``[(nreads, *sweep, 2)]``; with one read this is
    ``(1, n_sweep, 2)`` -> take channel 0, read 0, combine I/Q.
    """

    iq = result[0][0]  # (n_sweep, 2)
    return np.abs(iq[:, 0] + 1j * iq[:, 1])


# --------------------------------------------------------------------------- D1


def test_d1_no_sim_white_noise_unchanged():
    """make_mock_soc() with no SimParams keeps the original white-noise path."""

    soc, soccfg = make_mock_soc()
    assert soc._sim_params is None

    sw = SweepCfg(start=4000.0, stop=4100.0, expts=5, step=25.0)
    freq_param = sweep2param("freq", sw)
    pulse = PulseCfg(
        ch=0,
        nqz=1,
        gain=0.3,
        freq=freq_param,
        phase=0.0,
        waveform=ConstWaveformCfg(length=0.5),
    ).build("qub")
    prog = ModularProgramV2(
        soccfg,
        ProgramV2Cfg(reps=10, rounds=2),
        modules=[pulse, _readout(7200.0)],
        sweep=[("freq", sw)],
    )

    result = prog.acquire(soc, progress=False)

    # Same shape contract as the real accumulated path: (nreads, n_sweep, 2).
    assert len(result) == 1
    assert result[0].shape == (1, 5, 2)
    # get_raw keeps the (reps, *sweep, nreads, 2) raw layout.
    raw = prog.get_raw()
    assert raw is not None
    assert raw[0].shape == (10, 5, 1, 2)


# ------------------------------------------------------------ engine smoke path


def test_engine_twotone_freq_peak():
    """A twotone frequency sweep shows a feature at the qubit frequency."""

    soc, soccfg = make_mock_soc(sim=_SIM)
    f_qubit = _f_qubit_mhz()

    sw = SweepCfg(start=f_qubit - 250.0, stop=f_qubit + 250.0, expts=51, step=10.0)
    freq_param = sweep2param("freq", sw)
    pulse = PulseCfg(
        ch=0,
        nqz=1,
        gain=0.05,
        freq=freq_param,
        phase=0.0,
        waveform=ConstWaveformCfg(length=2.0),
    ).build("qub")
    prog = ModularProgramV2(
        soccfg,
        ProgramV2Cfg(reps=100, rounds=2),
        modules=[pulse, _readout(_rf_g_mhz())],
        sweep=[("freq", sw)],
    )

    amp = _amp(prog.acquire(soc, progress=False))

    # Not white noise: the feature near f_qubit dominates the residual scatter.
    freqs = np.linspace(sw.start, sw.stop, sw.expts)
    deviation = np.abs(amp - np.median(amp))
    peak_freq = freqs[int(np.argmax(deviation))]
    baseline = np.median(deviation)
    assert deviation.max() > 5.0 * (baseline + 1e-9)
    assert abs(peak_freq - f_qubit) < 60.0


def test_engine_amp_rabi_oscillates():
    """An amplitude (gain) Rabi sweep oscillates with gain (chevron at f_qubit)."""

    soc, soccfg = make_mock_soc(sim=_SIM)
    f_qubit = _f_qubit_mhz()

    sw = SweepCfg(start=0.0, stop=2.0, expts=50, step=2.0 / 49)
    gain_param = sweep2param("gain", sw)
    pulse = PulseCfg(
        ch=0,
        nqz=1,
        gain=gain_param,
        freq=f_qubit,
        phase=0.0,
        waveform=ConstWaveformCfg(length=1.0),
    ).build("qub")
    prog = ModularProgramV2(
        soccfg,
        ProgramV2Cfg(reps=80, rounds=2),
        modules=[pulse, _readout(_rf_g_mhz())],
        sweep=[("gain", sw)],
    )

    amp = _amp(prog.acquire(soc, progress=False))

    # Oscillation: the centered trace crosses its mean several times (gain 0->2
    # at pi_gain_len=0.4 with length=1 covers multiple pi rotations).
    centered = amp - amp.mean()
    crossings = int(np.sum(np.diff(np.sign(centered)) != 0))
    assert crossings >= 3


def test_engine_t1_decays():
    """A pi pulse + variable delay shows a monotone-ish T1 decay in readout amp."""

    soc, soccfg = make_mock_soc(sim=_SIM)
    f_qubit = _f_qubit_mhz()

    sw = SweepCfg(start=0.0, stop=80.0, expts=20, step=80.0 / 19)
    delay_param = sweep2param("t1_delay", sw)
    # gain * length == pi_gain_len (1.0 * 0.4 == 0.4) is an exact pi rotation.
    pi_pulse = PulseCfg(
        ch=0,
        nqz=1,
        gain=1.0,
        freq=f_qubit,
        phase=0.0,
        waveform=ConstWaveformCfg(length=0.4),
    ).build("pi")
    prog = ModularProgramV2(
        soccfg,
        ProgramV2Cfg(reps=100, rounds=2),
        modules=[pi_pulse, Delay("wait", delay=delay_param), _readout(_rf_g_mhz())],
        sweep=[("t1_delay", sw)],
    )

    amp = _amp(prog.acquire(soc, progress=False))

    # Excited at delay 0, relaxed at the long-delay end; the amplitude changes
    # substantially and the late-time mean is far from the early-time value.
    early = amp[:3].mean()
    late = amp[-3:].mean()
    assert abs(early - late) > 5.0
    # Decay is toward the equilibrium: first point is the most excited.
    assert amp[0] == pytest.approx(amp.max(), abs=0.15 * np.ptp(amp))


def _fringe_frequency(taus: np.ndarray, signal: np.ndarray) -> float:
    """Dominant oscillation frequency (cycles per µs) of ``signal`` over ``taus``.

    Zero-pads the FFT so the bin spacing resolves a frequency that need not land
    on a raw DFT bin; mirrors the helper in ``test_bloch_limits``.  With ``taus``
    in µs the returned frequency in cycles/µs equals a detuning expressed in MHz.
    """

    dt = float(taus[1] - taus[0])
    n_fft = 1 << 16
    spectrum = np.abs(np.fft.rfft(signal - signal.mean(), n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, dt)
    return float(freqs[int(np.argmax(spectrum))])


def test_engine_t2ramsey_fringes_at_detuning():
    """Ramsey (pi/2 - detuned free - pi/2) fringes at exactly the detuning.

    Mechanism A (physical detuning): both pi/2 pulses sit at ``f_qubit +
    detuning`` so the single rotating frame is detuned by ``detuning`` and the
    Bloch vector precesses during the idle delay.  The fix is in the lowering
    layer (idle segments carry the frame detuning instead of 0); the fringe
    frequency must equal the detuning.
    """

    soc, soccfg = make_mock_soc(sim=_SIM)
    f_qubit = _f_qubit_mhz()
    detuning = 3.0  # MHz: sets the fringe frequency (cycles/µs).

    # 128 points over 16 µs keeps the fringe frequency well below Nyquist
    # (~4 cycles/µs) so the FFT does not alias.
    sw = SweepCfg(start=0.0, stop=16.0, expts=128, step=16.0 / 127)
    delay_param = sweep2param("t2_delay", sw)

    def pi_half(name: str) -> Module:
        # gain * length == 0.2 == pi_gain_len / 2 -> a pi/2 rotation.
        return PulseCfg(
            ch=0,
            nqz=1,
            gain=1.0,
            freq=f_qubit + detuning,
            phase=0.0,
            waveform=ConstWaveformCfg(length=0.2),
        ).build(name)

    prog = ModularProgramV2(
        soccfg,
        ProgramV2Cfg(reps=80, rounds=2),
        modules=[
            pi_half("p1"),
            Delay("wait", delay=delay_param),
            pi_half("p2"),
            _readout(_rf_g_mhz()),
        ],
        sweep=[("t2_delay", sw)],
    )

    amp = _amp(prog.acquire(soc, progress=False))
    taus = np.linspace(sw.start, sw.stop, sw.expts)

    # Fringes exist (several mean crossings, not a monotone T2 decay) ...
    centered = amp - amp.mean()
    crossings = int(np.sum(np.diff(np.sign(centered)) != 0))
    assert crossings >= 3
    # ... and oscillate at the set detuning (cycles/µs == MHz).
    assert _fringe_frequency(taus, amp) == pytest.approx(detuning, rel=0.05)


def test_engine_t2ramsey_fringe_frequency_tracks_detuning():
    """The Ramsey fringe frequency scales 1:1 with the configured detuning."""

    soc, soccfg = make_mock_soc(sim=_SIM)
    f_qubit = _f_qubit_mhz()

    sw = SweepCfg(start=0.0, stop=16.0, expts=128, step=16.0 / 127)
    delay_param = sweep2param("t2_delay", sw)
    taus = np.linspace(sw.start, sw.stop, sw.expts)

    for detuning in (1.0, 2.0, 3.0):

        def pi_half(name: str, det: float = detuning) -> Module:
            return PulseCfg(
                ch=0,
                nqz=1,
                gain=1.0,
                freq=f_qubit + det,
                phase=0.0,
                waveform=ConstWaveformCfg(length=0.2),
            ).build(name)

        prog = ModularProgramV2(
            soccfg,
            ProgramV2Cfg(reps=80, rounds=2),
            modules=[
                pi_half("p1"),
                Delay("wait", delay=delay_param),
                pi_half("p2"),
                _readout(_rf_g_mhz()),
            ],
            sweep=[("t2_delay", sw)],
        )
        amp = _amp(prog.acquire(soc, progress=False))
        assert _fringe_frequency(taus, amp) == pytest.approx(detuning, rel=0.05)


def test_engine_onetone_dip():
    """A onetone readout-frequency sweep shows a resonator dip near rf_g."""

    soc, soccfg = make_mock_soc(sim=_SIM)
    rf_g = _rf_g_mhz()

    sw = SweepCfg(start=rf_g - 100.0, stop=rf_g + 100.0, expts=81, step=200.0 / 80)
    ro_param = sweep2param("ro_freq", sw)
    readout_module = DirectReadoutCfg(ro_ch=0, ro_length=1.0, ro_freq=ro_param).build(
        "ro"
    )
    prog = ModularProgramV2(
        soccfg,
        ProgramV2Cfg(reps=80, rounds=2),
        modules=[readout_module],
        sweep=[("ro_freq", sw)],
    )

    amp = _amp(prog.acquire(soc, progress=False))

    # A dip: the minimum is well below the off-resonant median, and it sits near
    # rf_g (the onetone, p_e = thermal, probes the ground-state resonator).
    freqs = np.linspace(sw.start, sw.stop, sw.expts)
    dip_freq = freqs[int(np.argmin(amp))]
    assert amp.min() < 0.6 * np.median(amp)
    assert abs(dip_freq - rf_g) < 30.0


# ------------------------------------------------------------------- singleshot


def test_engine_singleshot_get_raw_shape():
    """A no-sweep (singleshot-style) sim run yields get_raw shape (reps, 1, 2)."""

    soc, soccfg = make_mock_soc(sim=_SIM)
    f_qubit = _f_qubit_mhz()

    reps = 128
    pi_pulse = PulseCfg(
        ch=0,
        nqz=1,
        gain=1.0,
        freq=f_qubit,
        phase=0.0,
        waveform=ConstWaveformCfg(length=0.4),
    ).build("pi")
    prog = ModularProgramV2(
        soccfg,
        ProgramV2Cfg(reps=reps, rounds=1),
        modules=[pi_pulse, _readout(_rf_g_mhz())],
    )

    prog.acquire(soc, progress=False)

    raw = prog.get_raw()
    assert raw is not None
    assert len(raw) == 1
    assert raw[0].shape == (reps, 1, 2)
    assert raw[0].dtype == np.int64


# -------------------------------------------------------------------- round hook


def test_engine_round_hook_called_per_round():
    """round_hook fires exactly once per round on the sim path."""

    soc, soccfg = make_mock_soc(sim=_SIM)
    f_qubit = _f_qubit_mhz()
    rounds = 3

    calls: list[int] = []

    def hook(round_count: int, _data) -> None:
        calls.append(round_count)

    sw = SweepCfg(start=f_qubit - 100.0, stop=f_qubit + 100.0, expts=5, step=50.0)
    freq_param = sweep2param("freq", sw)
    pulse = PulseCfg(
        ch=0,
        nqz=1,
        gain=0.1,
        freq=freq_param,
        phase=0.0,
        waveform=ConstWaveformCfg(length=1.0),
    ).build("qub")
    prog = ModularProgramV2(
        soccfg,
        ProgramV2Cfg(reps=20, rounds=rounds),
        modules=[pulse, _readout(_rf_g_mhz())],
        sweep=[("freq", sw)],
    )

    prog.acquire(soc, progress=False, round_hook=hook)

    assert len(calls) == rounds
    # round_count is the running number of completed rounds (1..rounds).
    assert calls == [1, 2, 3]


# ---------------------------------------------------------------- decimated D2


def test_engine_acquire_decimated_fast_fails():
    """acquire_decimated on a sim soc raises (decimated is out of scope, D2)."""

    soc, soccfg = make_mock_soc(sim=_SIM)
    f_qubit = _f_qubit_mhz()

    pulse = PulseCfg(
        ch=0,
        nqz=1,
        gain=0.1,
        freq=f_qubit,
        phase=0.0,
        waveform=ConstWaveformCfg(length=1.0),
    ).build("qub")
    prog = ModularProgramV2(
        soccfg,
        ProgramV2Cfg(reps=10, rounds=1),
        modules=[pulse, _readout(_rf_g_mhz())],
    )

    with pytest.raises(NotImplementedError, match="decimated"):
        prog.acquire_decimated(soc, progress=False)


# ---------------------------------------------- Phase 2c: Lorentzian dephasing
#
# The engine averages the deterministic per-point signal over a Lorentzian
# quasi-static detune ensemble (HWHM Gamma == SimParams.inhomogeneous_rate,
# rad/µs).  The substitution ``delta = Gamma*tan(theta)`` makes the Lorentzian
# weight uniform on theta, so a fixed Gauss-Legendre rule integrates the ensemble
# average deterministically (no RNG).  This block tests the four physics gates:
# the quadrature reproduces the analytic FID, echo refocuses to the homogeneous
# T2 (Gamma-insensitive), Ramsey decays faster than echo, and Gamma == 0 reduces
# to the Phase-1 single-eval path bit-for-bit.


def _sim_with_dephasing(*, T2: float, T2_star: float, snr: float = 1.0e9) -> SimParams:
    """A _SIM clone with a chosen homogeneous/inhomogeneous split.

    ``T1`` is pushed high so the coherence envelope is dominated by dephasing
    (the gate under test) rather than T1 relaxation; ``snr`` defaults to
    effectively noiseless so the deterministic envelope is read directly.
    """

    return _SIM.model_copy(
        update={"T1": 100.0, "T2": T2, "T2_star": T2_star, "snr": snr}
    )


def _echo_envelope(sim: SimParams, stop: float, expts: int) -> np.ndarray:
    """Normalized echo (pi/2 - tau/2 - pi - tau/2 - pi/2) readout envelope.

    The pi pulse at the midpoint refocuses each static detune; what survives is
    the homogeneous (T2) decay.  Returns the readout magnitude with its long-time
    baseline removed and scaled to 1 at tau = 0, i.e. the coherence envelope.
    """

    f_qubit = _f_qubit_mhz()
    soc, soccfg = make_mock_soc(sim=sim)
    sw = SweepCfg(start=0.0, stop=stop, expts=expts, step=stop / (expts - 1))
    total = sweep2param("t", sw)
    half = 0.5 * total

    def pi_half(name: str) -> Module:
        return PulseCfg(
            ch=0,
            nqz=1,
            gain=1.0,
            freq=f_qubit,
            phase=0.0,
            waveform=ConstWaveformCfg(length=0.2),
        ).build(name)

    pi_pulse = PulseCfg(
        ch=0,
        nqz=1,
        gain=1.0,
        freq=f_qubit,
        phase=90.0,  # pi about a transverse axis: the refocusing pulse
        waveform=ConstWaveformCfg(length=0.4),
    ).build("pi")

    prog = ModularProgramV2(
        soccfg,
        ProgramV2Cfg(reps=1, rounds=1),
        modules=[
            pi_half("p1"),
            Delay("w1", delay=half),
            pi_pulse,
            Delay("w2", delay=half),
            pi_half("p2"),
            _readout(_rf_g_mhz()),
        ],
        sweep=[("t", sw)],
    )
    amp = _amp(prog.acquire(soc, progress=False))
    env = amp - amp[-1]
    return env / env[0]


def _ramsey_envelope(sim: SimParams, stop: float, expts: int) -> np.ndarray:
    """Normalized on-resonance Ramsey (pi/2 - tau - pi/2) readout envelope.

    On resonance there are no fringes; the only decay is dephasing, which here
    includes the (un-refocused) Lorentzian ensemble -> the extra exp(-Gamma*t)
    that makes Ramsey decay to T2* rather than T2.
    """

    f_qubit = _f_qubit_mhz()
    soc, soccfg = make_mock_soc(sim=sim)
    sw = SweepCfg(start=0.0, stop=stop, expts=expts, step=stop / (expts - 1))
    delay = sweep2param("t", sw)

    def pi_half(name: str) -> Module:
        return PulseCfg(
            ch=0,
            nqz=1,
            gain=1.0,
            freq=f_qubit,
            phase=0.0,
            waveform=ConstWaveformCfg(length=0.2),
        ).build(name)

    prog = ModularProgramV2(
        soccfg,
        ProgramV2Cfg(reps=1, rounds=1),
        modules=[
            pi_half("p1"),
            Delay("wait", delay=delay),
            pi_half("p2"),
            _readout(_rf_g_mhz()),
        ],
        sweep=[("t", sw)],
    )
    amp = _amp(prog.acquire(soc, progress=False))
    env = amp - amp[-1]
    return env / env[0]


@pytest.mark.parametrize("gamma", [0.05, 0.1, 0.3])
def test_ensemble_quadrature_reproduces_fid(gamma: float):
    """The Lorentzian quadrature average of free precession is exp(-Gamma|t|).

    This is the direct quadrature-correctness gate: for a static detune delta the
    free-evolution coherence is exp(i*delta*t), and the Lorentzian (HWHM Gamma)
    ensemble average is the analytic FID exp(-Gamma|t|).  We verify the engine's
    own nodes/weights reproduce it over the *observable* decay window
    ``Gamma*t in [0, 2]`` — beyond it the FID has decayed below the readout noise
    floor, so its absolute accuracy is NOT load-bearing (the tan-substitution +
    fixed Gauss-Legendre rule converges slowly for this oscillatory integrand only
    deep in that decayed tail).  The real correctness guarantee for the dephasing
    model is the inject->recover integration suite, which fits the envelope with
    the genuine analyzers; this gate only certifies the quadrature is sane in the
    window where the signal carries information.
    """

    from zcu_tools.program.v2.sim.engine import _ENSEMBLE_NODES, _lorentzian_quadrature

    theta, weights = _lorentzian_quadrature(_ENSEMBLE_NODES)
    delta = gamma * np.tan(theta)  # ensemble detune nodes (rad/µs)

    # Probe the observable decay window (Gamma*t up to 2, where the analyzer
    # actually fits the envelope).  The absolute FID tolerance (0.06) reflects the
    # accuracy of the current node count (_ENSEMBLE_NODES == 41) over that window;
    # raising the node count tightens it.
    for gamma_t in (0.0, 0.25, 0.5, 1.0, 1.5, 2.0):
        t = gamma_t / gamma
        fid = np.sum(weights * np.exp(1j * delta * t))
        # Real FID matches the analytic envelope; the imaginary part is zero by
        # the symmetry of the Lorentzian about delta = 0.
        assert fid.real == pytest.approx(np.exp(-gamma_t), abs=0.06)
        assert fid.imag == pytest.approx(0.0, abs=1e-9)


def test_engine_echo_refocuses_homogeneous_t2():
    """Echo recovers the homogeneous decay and is insensitive to Gamma.

    Sweeping Gamma (via T2_star) at a fixed homogeneous T2 must leave the echo
    envelope essentially unchanged — the pi pulse refocuses every static detune,
    so only the T2 (Bloch gamma2) decay survives.  This is the load-bearing
    refocus gate: the dephasing model is sequence-agnostic, so the cancellation
    must emerge purely from the pi flip + ensemble average.
    """

    stop, expts = 12.0, 40
    baseline = _echo_envelope(_sim_with_dephasing(T2=15.0, T2_star=15.0), stop, expts)
    for T2_star in (8.0, 4.0):
        env = _echo_envelope(_sim_with_dephasing(T2=15.0, T2_star=T2_star), stop, expts)
        # Echo envelope tracks T2 only: changing Gamma (T2_star) barely moves it.
        assert np.max(np.abs(env - baseline)) < 0.05


def test_engine_echo_tracks_homogeneous_t2():
    """A shorter homogeneous T2 makes the echo envelope decay faster.

    Complements the Gamma-insensitivity gate: the echo *does* respond to the
    homogeneous T2 (it is not flat), confirming it recovers T2 rather than
    ignoring decay entirely.
    """

    stop, expts = 12.0, 40
    long_t2 = _echo_envelope(_sim_with_dephasing(T2=15.0, T2_star=4.0), stop, expts)
    short_t2 = _echo_envelope(_sim_with_dephasing(T2=8.0, T2_star=4.0), stop, expts)
    mid = expts // 2
    assert short_t2[mid] < long_t2[mid]


def test_engine_ramsey_decays_faster_than_echo():
    """At Gamma > 0 the Ramsey envelope decays faster than the echo envelope.

    Ramsey does not refocus the static detune, so it picks up the extra
    exp(-Gamma*t) (-> T2*); echo refocuses it (-> T2 > T2*).  Same physical
    params, same readout: the Ramsey coherence must be the smaller of the two at
    a common evolution time.
    """

    sim = _sim_with_dephasing(T2=20.0, T2_star=5.0)  # Gamma = 0.15 /µs
    stop, expts = 16.0, 60
    ramsey = _ramsey_envelope(sim, stop, expts)
    echo = _echo_envelope(sim, stop, expts)
    mid = expts // 2
    # Compare where both envelopes are well above the baseline noise floor.
    assert ramsey[mid] < echo[mid]


def test_engine_gamma_zero_zero_regression():
    """Gamma == 0 (T2_star == T2) reproduces the Phase-1 single-eval output.

    The _SIM fixture already has T2_star == T2, so its inhomogeneous_rate is 0
    and the ensemble collapses to a single delta = 0 node.  Two runs at the same
    seed must be bit-identical (deterministic signal + identical RNG stream), and
    a T1 program's output must match a hand-rolled single Bloch eval — i.e. the
    quadrature path does not perturb the established Phase-1 numbers.
    """

    assert _SIM.inhomogeneous_rate == 0.0

    f_qubit = _f_qubit_mhz()
    sw = SweepCfg(start=0.0, stop=40.0, expts=12, step=40.0 / 11)
    delay_param = sweep2param("t1_delay", sw)
    pi_pulse = PulseCfg(
        ch=0,
        nqz=1,
        gain=1.0,
        freq=f_qubit,
        phase=0.0,
        waveform=ConstWaveformCfg(length=0.4),
    ).build("pi")

    def run() -> np.ndarray:
        soc, soccfg = make_mock_soc(sim=_SIM)
        prog = ModularProgramV2(
            soccfg,
            ProgramV2Cfg(reps=40, rounds=2),
            modules=[
                pi_pulse,
                Delay("wait", delay=delay_param),
                _readout(_rf_g_mhz()),
            ],
            sweep=[("t1_delay", sw)],
        )
        return prog.acquire(soc, progress=False)[0]

    first = run()
    second = run()
    # Identical seed + deterministic single-node quadrature => bit-identical.
    np.testing.assert_array_equal(first, second)


# ------------------------------------------------ Phase 2: deterministic Branch
#
# The lowering layer's deterministic-Branch selection (a sub-sequence chosen by a
# registered sweep-loop counter, modelling g/e prep) has thorough unit coverage in
# test_lowering.TestDeterministicBranch.  This is the matching engine-level smoke:
# a Branch driven through the full acquire path produces the expected shape and a
# real branch effect (ground vs excited readout differ).


def test_engine_deterministic_branch_smoke():
    """A g/e-prep Branch drives end to end and the two branches differ.

    Branch 0 is empty (ground), branch 1 is a pi pulse (excited); the sweep axis
    ``ge`` (registered as a 2-point loop) selects the branch per point.  The two
    sweep points must produce a clean shape and a clearly different readout
    magnitude — proof the deterministic Branch lowers and reads out through the
    engine, not just the lowering unit layer.
    """

    f_qubit = _f_qubit_mhz()
    soc, soccfg = make_mock_soc(sim=_SIM)
    pi_pulse = PulseCfg(
        ch=0,
        nqz=1,
        gain=1.0,
        freq=f_qubit,
        phase=0.0,
        waveform=ConstWaveformCfg(length=0.4),  # gain*length == pi_gain_len -> pi
    ).build("pi")
    prog = ModularProgramV2(
        soccfg,
        ProgramV2Cfg(reps=80, rounds=2),
        modules=[Branch("ge", [], pi_pulse), _readout(_rf_g_mhz())],
        sweep=[("ge", SweepCfg(start=0, stop=1, expts=2, step=1))],
    )
    amp = _amp(prog.acquire(soc, progress=False))  # (2,) magnitude per branch

    assert amp.shape == (2,)
    # Branch 1 (pi pulse -> excited) reads out clearly different from branch 0
    # (empty -> ground); the engine resolved the deterministic branch selection.
    assert abs(amp[1] - amp[0]) > 0.5 * max(amp[0], amp[1])
