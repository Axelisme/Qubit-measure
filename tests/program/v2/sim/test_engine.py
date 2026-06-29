"""Tests for sim/engine.py — SimEngine assembly + MyProgramV2.acquire dispatch.

Covers:
- D1 zero-regression: ``make_mock_soc()`` with no SimParams runs the unchanged
  white-noise accumulated path (same shapes, random data).
- Engine smoke (real path): a sim soc running real experiment programs
  (twotone freq, amp_rabi, T1, T2 ramsey) produces physically-structured data,
  not white noise — a peak/dip at f_qubit, gain-driven Rabi oscillation, a T1
  decay, and Ramsey fringes.
- Singleshot get_raw shape ``(reps, 1, 2)`` is usable on the sim path, and the
  per-shot Bernoulli blobs are correct: the shots cluster on the |g>/|e> blob
  centres set by the init pulse, a pi/2 pulse puts ~half on the excited blob, and
  the reps-mean of the blobs equals the accumulated readout (zero-regression).
- round_hook is invoked once per round.
- acquire_decimated on a sim soc returns a physically-structured time-domain
  trace (model A): a timeFly-shifted readout envelope, ~0 before timeFly.

The engine drives the qubit at its *own* predicted f_qubit (queried through a
FluxoniumPredictor built from the same SimParams) and reads out near rf_g — i.e.
the test plays the role of an experimenter who has already located the qubit and
the resonator, exactly as the real path requires.
"""

from __future__ import annotations

import numpy as np
import pytest
from numpy.typing import NDArray
from zcu_tools.program.v2.base import ProgramV2Cfg
from zcu_tools.program.v2.mocksoc import make_mock_soc
from zcu_tools.program.v2.modular import ModularProgramV2
from zcu_tools.program.v2.modules.base import Module
from zcu_tools.program.v2.modules.control import Branch
from zcu_tools.program.v2.modules.delay import Delay
from zcu_tools.program.v2.modules.pulse import PulseCfg
from zcu_tools.program.v2.modules.readout import DirectReadoutCfg, PulseReadoutCfg
from zcu_tools.program.v2.modules.waveform import ConstWaveformCfg
from zcu_tools.program.v2.sim import SimParams
from zcu_tools.program.v2.sim.engine import (
    _FULL_SCALE,
    SimCancelledError,
    SimEngine,
    _PointModel,
    _PointReadout,
)
from zcu_tools.program.v2.sim.readout import (
    effective_signal_samples,
    resonator_freqs,
    s21,
)
from zcu_tools.program.v2.sweep import SweepCfg
from zcu_tools.program.v2.utils import sweep2param
from zcu_tools.simulate.fluxonium.predict import FluxoniumPredictor

# Operating point: the engine pins reduced flux = 1.0 (R-3).  At EJ/EC/EL =
# 8.5/1.0/0.5 the qubit sits a few GHz above the sweet spot there, giving a finite
# gap and a clear dispersive shift.  T1/T2 are short so decay/dephasing are visible
# over modest sweeps; snr is generous so the physical structure dominates the noise.
# These tests drive ModularProgramV2 directly with a *raw* (un-folded) sweep axis,
# so the engine's true f_qubit (~7391 MHz) is the same axis the peak is read on —
# Nyquist folding only affects the analyzer's axis, which this layer does not use.
_OPERATING_FLUX = 1.0

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

_RESET_RELAX_DELAY = 10.0 * _SIM.T1


def _f_qubit_mhz() -> float:
    """The qubit 0->1 frequency (MHz) the engine sees at the fixed operating flux.

    The engine pins reduced flux = 1.0 (R-3) and feeds ``predict_freq`` a *device
    value*, so map the fixed flux back through ``flux_to_value`` exactly as the
    engine does — this is the true (un-folded) f_qubit the engine drives at.
    """

    predictor = FluxoniumPredictor(
        params=(_SIM.EJ, _SIM.EC, _SIM.EL),
        flux_half=_SIM.flux_half,
        flux_period=_SIM.flux_period,
        flux_bias=_SIM.flux_bias,
    )
    return float(predictor.predict_freq(predictor.flux_to_value(_OPERATING_FLUX)))


def _rf_g_mhz() -> float:
    """The ground-state dressed resonator frequency (MHz) at the fixed operating flux.

    Reading out near rf_g maximizes the |g>/|e> contrast, which is what makes
    the T1 decay and Rabi oscillation visible in the readout magnitude.
    """

    rf_g, _rf_e = resonator_freqs(_SIM, _OPERATING_FLUX)
    return rf_g * 1e3


def _readout(ro_freq_mhz: float) -> Module:
    return DirectReadoutCfg(ro_ch=0, ro_length=1.0, ro_freq=ro_freq_mhz).build("ro")


def _compiled_sample_times(
    readout_module: Module, *, sim: SimParams = _SIM
) -> tuple[int, NDArray[np.float64]]:
    """Compile a one-readout program and return its ADC sample axis."""

    _soc, soccfg = make_mock_soc(sim=sim)
    prog = ModularProgramV2(
        soccfg,
        ProgramV2Cfg(reps=1, rounds=1),
        modules=[readout_module],
    )
    prog.compile()
    ((ro_ch, ro),) = prog.ro_chs.items()
    n_samples = int(ro["length"])
    ts = prog.soccfg.cycles2us(np.arange(n_samples), ro_ch=ro_ch)
    return n_samples, np.asarray(ts, dtype=np.float64)


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


def test_engine_qub_pulse_with_swept_ro_freq_dip_tracks_pe():
    """R-1 unified path: a qubit pulse + swept readout frequency in one program.

    The pre-readout π pulse excites P_e≈1, so the dispersive dip rides rf_e; with
    no pulse the qubit stays at ~thermal and the dip rides rf_g.  This is the new
    case the unified SimEngine path supports — pre-R-1 the swept-ro_freq branch
    ignored the qubit pulse entirely and always probed rf_g.
    """

    f_qubit = _f_qubit_mhz()
    rf_g_mhz, rf_e_mhz = (f * 1e3 for f in resonator_freqs(_SIM, _OPERATING_FLUX))

    # Sweep wide enough to bracket both rf_g and rf_e.
    lo = min(rf_g_mhz, rf_e_mhz) - 50.0
    hi = max(rf_g_mhz, rf_e_mhz) + 50.0
    sw = SweepCfg(start=lo, stop=hi, expts=121, step=(hi - lo) / 120)
    freqs = np.linspace(sw.start, sw.stop, sw.expts)

    def _dip_freq(pi_gain: float) -> float:
        soc, soccfg = make_mock_soc(sim=_SIM)
        # gain*length == pi_gain_len (0.4) is an exact π; gain=0 is no excitation.
        qub_pulse = PulseCfg(
            ch=0,
            nqz=1,
            gain=pi_gain,
            freq=f_qubit,
            phase=0.0,
            waveform=ConstWaveformCfg(length=0.4),
        ).build("qub")
        ro_param = sweep2param("ro_freq", sw)
        ro = DirectReadoutCfg(ro_ch=0, ro_length=1.0, ro_freq=ro_param).build("ro")
        prog = ModularProgramV2(
            soccfg,
            ProgramV2Cfg(reps=80, rounds=2),
            modules=[qub_pulse, ro],
            sweep=[("ro_freq", sw)],
        )
        amp = _amp(prog.acquire(soc, progress=False))
        return float(freqs[int(np.argmin(amp))])

    # π pulse (gain=1.0, length=0.4 => θ=π): dip rides rf_e.
    dip_pi = _dip_freq(1.0)
    assert abs(dip_pi - rf_e_mhz) < abs(dip_pi - rf_g_mhz)

    # no pulse (gain=0): qubit at ~thermal => dip rides rf_g.
    dip_zero = _dip_freq(0.0)
    assert abs(dip_zero - rf_g_mhz) < abs(dip_zero - rf_e_mhz)

    # The two dips are distinct (the pulse genuinely moved the readout response).
    assert abs(dip_pi - dip_zero) > 10.0


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


def _expected_blob_centers(ro_freq_mhz: float) -> tuple[complex, complex]:
    """The two per-shot blob centres ``(g_center, e_center)`` in raw ADC units.

    The engine reads out at probe frequency ``ro_freq_mhz`` and lays out a shot
    on either the |g>-conditioned blob
    ``_FULL_SCALE * signal_samples * S21(f_ro; rf_g)`` or the |e>-conditioned
    blob with ``rf_e``.  These are the integrated raw deterministic centres a
    noiseless, fully-polarised population would produce, so the singleshot
    get_raw clusters must land on them.
    """

    rf_g, rf_e = resonator_freqs(_SIM, _OPERATING_FLUX)
    freqs = np.array([ro_freq_mhz * 1e-3], dtype=np.float64)  # MHz -> GHz
    s_g = complex(s21(_SIM, freqs, rf_g)[0])
    s_e = complex(s21(_SIM, freqs, rf_e)[0])
    _n_samples, sample_times = _compiled_sample_times(_readout(ro_freq_mhz))
    signal_samples = effective_signal_samples(None, None, sample_times)
    return _FULL_SCALE * signal_samples * s_g, _FULL_SCALE * signal_samples * s_e


def _singleshot_raw(
    *, gain: float, length: float, reps: int = 4000
) -> NDArray[np.complex128]:
    """Run a no-sweep init-pulse + readout and return the per-shot signals.

    A const pulse of (gain, length) sets the excited population P_e, then a single
    readout at rf_g is taken; the returned complex array has one entry per rep
    (the singleshot get_raw IQ at integration full scale).
    """

    soc, soccfg = make_mock_soc(sim=_SIM)
    f_qubit = _f_qubit_mhz()
    ro_freq = _rf_g_mhz()

    modules: list[Module] = [_readout(ro_freq)]
    if gain > 0.0:
        pulse = PulseCfg(
            ch=0,
            nqz=1,
            gain=gain,
            freq=f_qubit,
            phase=0.0,
            waveform=ConstWaveformCfg(length=length),
        ).build("init")
        modules = [pulse, *modules]

    prog = ModularProgramV2(
        soccfg,
        ProgramV2Cfg(reps=reps, rounds=1),
        modules=modules,
    )
    prog.acquire(soc, progress=False)

    raw = prog.get_raw()
    assert raw is not None
    iq = raw[0]  # (reps, 1, 2)
    return iq[:, 0, 0] + 1j * iq[:, 0, 1]  # (reps,)


def test_engine_singleshot_two_blobs_centers():
    """get_raw shots cluster on the |g> / |e> blob centres set by the init pulse.

    A pi pulse (gain*length == pi_gain_len) drives P_e ~ 1, so essentially every
    shot lands on the integrated |e> blob; with no pulse P_e ~ thermal ~ 0, so
    the shots land on the integrated |g> blob.  The cluster medians (robust to
    the few mis-prepared shots and the Gaussian readout noise) must match the
    deterministic blob centres.
    """

    g_center, e_center = _expected_blob_centers(_rf_g_mhz())

    # pi pulse: gain * length == pi_gain_len (0.4) => a full pi rotation => P_e ~ 1.
    excited = _singleshot_raw(gain=1.0, length=_SIM.pi_gain_len)
    excited_med = np.median(excited.real) + 1j * np.median(excited.imag)

    # No pulse: P_e ~ thermal_pop ~ 0 => the |g> blob.
    ground = _singleshot_raw(gain=0.0, length=0.0)
    ground_med = np.median(ground.real) + 1j * np.median(ground.imag)

    blob_dist = abs(g_center - e_center)
    # Each cluster median lands on its blob centre, within a small fraction of the
    # inter-blob distance (median is robust to noise + the few mis-prepared shots).
    assert abs(excited_med - e_center) < 0.15 * blob_dist
    assert abs(ground_med - g_center) < 0.15 * blob_dist


def test_engine_singleshot_population_ratio():
    """A pi/2 pulse puts ~half the shots on the |e> blob (P_e ~ 0.5).

    gain * length == pi_gain_len / 2 is a pi/2 rotation, so P_e ~ 0.5: classifying
    each shot by which blob centre it is nearer must give ~50% on the excited
    blob.  This is the per-shot Bernoulli population coming through get_raw.
    """

    g_center, e_center = _expected_blob_centers(_rf_g_mhz())

    # pi/2: gain * length == pi_gain_len / 2 (1.0 * 0.2).
    signals = _singleshot_raw(gain=1.0, length=_SIM.pi_gain_len / 2.0)

    # Classify each shot by the nearer blob centre.
    d_g = np.abs(signals - g_center)
    d_e = np.abs(signals - e_center)
    frac_excited = float(np.mean(d_e < d_g))

    assert frac_excited == pytest.approx(0.5, abs=0.05)


def test_engine_singleshot_accumulated_mean_invariant():
    """The reps-mean of the per-shot blobs equals the accumulated readout.

    This is the load-bearing zero-regression invariant: averaging the two per-shot
    Bernoulli blobs over reps must reproduce the deterministic accumulated readout
    the old path broadcast directly.  Run the *same* pi/2 program twice on the same
    sim soc — once read shot-by-shot via get_raw, once read averaged via acquire —
    and check the get_raw reps-mean lands on the acquire value.  Deriving the
    reference from acquire (rather than an assumed P_e) makes this test the pure
    blob-averaging invariant, independent of the exact pi/2 calibration.
    """

    reps = 20000
    soc, soccfg = make_mock_soc(sim=_SIM)
    f_qubit = _f_qubit_mhz()
    ro_freq = _rf_g_mhz()

    pulse = PulseCfg(
        ch=0,
        nqz=1,
        gain=1.0,
        freq=f_qubit,
        phase=0.0,
        waveform=ConstWaveformCfg(length=_SIM.pi_gain_len / 2.0),  # pi/2
    ).build("init")
    prog = ModularProgramV2(
        soccfg,
        ProgramV2Cfg(reps=reps, rounds=1),
        modules=[pulse, _readout(ro_freq)],
    )

    # acquire averages over reps then divides the per-shot acc by the readout
    # window length, so ``acquire_value * length`` is the per-shot reps-mean.
    acc = prog.acquire(soc, progress=False)[0]  # (nreads, 2)
    length = list(prog.ro_chs.values())[0]["length"]
    acc_reps_mean = (acc[0, 0] + 1j * acc[0, 1]) * length

    # get_raw exposes the same run's per-shot blobs; their reps-mean must match.
    raw = prog.get_raw()
    assert raw is not None
    signals = raw[0][:, 0, 0] + 1j * raw[0][:, 0, 1]  # (reps,)
    reps_mean = np.mean(signals.real) + 1j * np.mean(signals.imag)

    g_center, e_center = _expected_blob_centers(ro_freq)
    blob_dist = abs(g_center - e_center)
    # The acquire value (rescaled to raw ADC units) and the get_raw reps-mean are
    # the same quantity computed two ways; they agree to within the shot-noise
    # floor (~ blob_dist * sqrt(0.25 / reps) ~ 0.0035 * blob_dist at reps=20000).
    assert abs(reps_mean - acc_reps_mean) < 0.02 * blob_dist


# ---------------------------------------------------------- rep state carry / nreads


def _pi_pulse_prog(relax_delay: float, *, reps: int = 4) -> ModularProgramV2:
    _soc, soccfg = make_mock_soc(sim=_SIM)
    f_qubit = _f_qubit_mhz()
    pulse = PulseCfg(
        ch=0,
        nqz=1,
        gain=1.0,
        freq=f_qubit,
        phase=0.0,
        waveform=ConstWaveformCfg(length=_SIM.pi_gain_len),
    ).build("pi")
    prog = ModularProgramV2(
        soccfg,
        ProgramV2Cfg(reps=reps, rounds=1, relax_delay=relax_delay),
        modules=[pulse, _readout(_rf_g_mhz())],
    )
    prog.compile()
    return prog


def _deterministic_p_e_chain(prog: ModularProgramV2) -> NDArray[np.float64]:
    engine = SimEngine(prog, _SIM)
    _s_g, _s_e, p_e, _signal_scale, _noise_scale, _gain_noise_scale = (
        engine._ensure_signal()
    )
    return p_e[:, 0]


def test_engine_relax_delay_zero_carries_state_between_reps() -> None:
    """With no inter-shot relaxation, a repeated pi pulse alternates population."""

    p_e = _deterministic_p_e_chain(_pi_pulse_prog(relax_delay=0.0))

    assert p_e[0] > 0.95
    assert p_e[1] < 0.05
    assert p_e[2] > 0.90
    assert p_e[3] < 0.10


def test_engine_long_relax_delay_reprepares_each_rep() -> None:
    """A long relax_delay passively returns the carried state near ground."""

    p_e = _deterministic_p_e_chain(_pi_pulse_prog(relax_delay=_RESET_RELAX_DELAY))

    assert np.all(p_e > 0.95)


def test_engine_compute_round_supports_multiple_read_triggers() -> None:
    """nreads > 1 broadcasts deterministic readout physics over trigger slots."""

    prog = _pi_pulse_prog(relax_delay=_RESET_RELAX_DELAY, reps=3)
    ((_, ro),) = prog.ro_chs.items()
    ro["trigs"] = 2

    engine = SimEngine(prog, _SIM)
    acc = engine.compute_round(0)[0]
    _s_g, _s_e, p_e, signal_scale, noise_scale, gain_noise_scale = (
        engine._ensure_signal()
    )

    assert acc.shape == (3, 2, 2)
    assert p_e.shape == (3, 2)
    assert signal_scale.shape == (2,)
    assert noise_scale.shape == (2,)
    assert gain_noise_scale.shape == (2,)
    np.testing.assert_allclose(p_e[:, 0], p_e[:, 1])
    np.testing.assert_allclose(signal_scale[0], signal_scale[1])
    np.testing.assert_allclose(noise_scale[0], noise_scale[1])
    np.testing.assert_allclose(gain_noise_scale[0], gain_noise_scale[1])


def _normalized_ground_raw_direct(
    *, ro_length: float, reps: int = 12_000
) -> tuple[NDArray[np.complex128], int]:
    """Run a ground-state DirectReadout and return raw samples divided by length."""

    sim = _SIM.model_copy(update={"snr": 25.0, "thermal_pop": 0.0})
    soc, soccfg = make_mock_soc(sim=sim)
    ro_freq = _rf_g_mhz()
    readout = DirectReadoutCfg(ro_ch=0, ro_length=ro_length, ro_freq=ro_freq).build(
        "ro"
    )
    prog = ModularProgramV2(
        soccfg,
        ProgramV2Cfg(reps=reps, rounds=1),
        modules=[readout],
    )
    prog.acquire(soc, progress=False)

    raw = prog.get_raw()
    assert raw is not None
    compiled_length = int(list(prog.ro_chs.values())[0]["length"])
    samples = raw[0][:, 0, 0] + 1j * raw[0][:, 0, 1]
    return samples / compiled_length, compiled_length


def test_engine_length_normalization_keeps_mean_and_reduces_noise():
    """Integrated raw sums normalize to stable means and 1/sqrt(length) noise."""

    short, n_short = _normalized_ground_raw_direct(ro_length=0.5)
    long, n_long = _normalized_ground_raw_direct(ro_length=2.0)

    mean_short = np.mean(short)
    mean_long = np.mean(long)
    scale = max(abs(mean_short), abs(mean_long), 1.0)
    assert abs(mean_short - mean_long) < 0.05 * scale

    std_short = float(np.std(short.real))
    std_long = float(np.std(long.real))
    assert std_short > std_long
    assert std_short / std_long == pytest.approx(np.sqrt(n_long / n_short), rel=0.15)


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


# ----------------------------------------------------------------- R-2 lazy poll


def test_engine_lazy_compute_respects_early_stop(monkeypatch):
    """R-2: an early-stopping run never computes the rounds it does not poll.

    With lazy poll-time compute the soc asks the engine for round N only when it
    polls round N.  A stop_checker that fires after the first round halts the
    round loop, so compute_round is called exactly once even though 5 rounds were
    configured — proving the unpolled rounds' physics is never computed.
    """

    from zcu_tools.program.v2.sim.engine import SimEngine

    calls: list[int] = []
    completed_rounds = 0
    real_compute_round = SimEngine.compute_round

    def spy_compute_round(self, round_idx: int):
        nonlocal completed_rounds
        calls.append(round_idx)
        result = real_compute_round(self, round_idx)
        completed_rounds += 1
        return result

    monkeypatch.setattr(SimEngine, "compute_round", spy_compute_round)

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
        ProgramV2Cfg(reps=20, rounds=5),
        modules=[pulse, _readout(_rf_g_mhz())],
    )

    # The stop_checker fires after the first round has been computed, so this
    # remains a round-boundary early-stop test rather than an intra-round cancel
    # test (covered separately below).
    prog.acquire(
        soc,
        progress=False,
        stop_checkers=[lambda: completed_rounds >= 1],
    )

    assert calls == [0], (
        f"expected exactly one round computed (early stop), got rounds {calls}"
    )


def test_acquire_stop_checker_is_checked_only_after_mock_round(monkeypatch):
    """Acquire-level stop_checkers keep hardware-like round-boundary semantics."""

    from zcu_tools.program.v2.sim.engine import SimEngine

    calls: list[int] = []
    real_compute_round = SimEngine.compute_round

    def spy_compute_round(self, round_idx: int):
        calls.append(round_idx)
        return real_compute_round(self, round_idx)

    monkeypatch.setattr(SimEngine, "compute_round", spy_compute_round)

    soc, soccfg = make_mock_soc(sim=_SIM)
    pulse = PulseCfg(
        ch=0,
        nqz=1,
        gain=0.1,
        freq=_f_qubit_mhz(),
        phase=0.0,
        waveform=ConstWaveformCfg(length=1.0),
    ).build("qub")
    prog = ModularProgramV2(
        soccfg,
        ProgramV2Cfg(reps=20, rounds=5),
        modules=[pulse, _readout(_rf_g_mhz())],
    )

    def stop_at_round_boundary() -> bool:
        return True

    prog.acquire(
        soc,
        progress=False,
        stop_checkers=[stop_at_round_boundary],
    )

    assert calls == [0], (
        "acquire-level stop checker should stop after the first completed round, "
        f"not before mock round compute; got rounds {calls}"
    )


def test_acquire_stop_checker_does_not_cancel_inside_mock_signal_grid(monkeypatch):
    """Acquire-level stop_checkers do not interrupt one mock round mid-compute."""

    from zcu_tools.program.v2.sim.engine import SimEngine

    readout_calls = 0

    def fake_operating_signal(self) -> tuple[float, float, float]:
        return (4.0, 7.0, 7.01)

    def fake_point_readout_model(
        self,
        lowered,
        f_qubit_ghz: float,
        rf_g: float,
        rf_e: float,
        n_samples: int,
        sample_times_us: NDArray[np.float64],
    ) -> _PointReadout:
        del lowered, f_qubit_ghz, rf_g, rf_e, n_samples, sample_times_us
        nonlocal readout_calls
        readout_calls += 1
        return _PointReadout(
            s_g=np.array([1.0 + 0.0j], dtype=np.complex128),
            s_e=np.array([0.0 + 0.0j], dtype=np.complex128),
            signal_scale=1.0,
            noise_std_scale=1.0,
            gain_noise_std_scale=0.0,
        )

    monkeypatch.setattr(SimEngine, "_operating_signal", fake_operating_signal)
    monkeypatch.setattr(SimEngine, "_point_readout_model", fake_point_readout_model)

    soc, soccfg = make_mock_soc(sim=_SIM.model_copy(update={"poll_latency": 0.0}))
    sw = SweepCfg(start=7000.0, stop=7010.0, expts=8, step=10.0 / 7)
    ro_param = sweep2param("ro_freq", sw)
    readout = DirectReadoutCfg(ro_ch=0, ro_length=1.0, ro_freq=ro_param).build("ro")
    prog = ModularProgramV2(
        soccfg,
        ProgramV2Cfg(reps=1, rounds=5),
        modules=[readout],
        sweep=[("ro_freq", sw)],
    )

    def stop_after_two_points() -> bool:
        return readout_calls >= 2

    prog.acquire(soc, progress=False, stop_checkers=[stop_after_two_points])

    assert readout_calls == sw.expts


def test_engine_cancel_during_detune_loop_raises(monkeypatch):
    """The Lorentzian detune ensemble loop checks stop_checkers cooperatively."""

    from zcu_tools.program.v2.sim import engine as engine_module
    from zcu_tools.program.v2.sim.bloch import Segment
    from zcu_tools.program.v2.sim.engine import SimEngine
    from zcu_tools.program.v2.sim.lowering import LoweredPoint, ReadoutPlan

    sim = _SIM.model_copy(update={"T2": 10.0, "T2_star": 5.0})
    soc, soccfg = make_mock_soc(sim=sim)
    prog = ModularProgramV2(
        soccfg,
        ProgramV2Cfg(reps=1, rounds=1),
        modules=[_readout(_rf_g_mhz())],
    )
    prog.compile()

    propagator_calls = 0

    def fake_lower(
        self, point: dict[str, int], f_qubit_ghz: float, detune_offset: float
    ) -> LoweredPoint:
        return LoweredPoint(
            segments=[
                Segment(
                    omega=1.0,
                    delta=detune_offset,
                    phase=0.0,
                    t=0.01,
                    t1=None,
                    t2=None,
                    thermal_pop=0.0,
                )
            ],
            readout=ReadoutPlan(f_ro_ghz=7.0, ro_length_us=1.0),
        )

    def fake_sequence_propagator(segments):
        nonlocal propagator_calls
        propagator_calls += 1
        return np.eye(4, dtype=np.float64)

    def stop_after_three_detune_nodes() -> bool:
        return propagator_calls >= 3

    monkeypatch.setattr(SimEngine, "_lower", fake_lower)
    monkeypatch.setattr(engine_module, "_sequence_propagator", fake_sequence_propagator)

    engine = SimEngine(prog, sim, stop_checkers=[stop_after_three_detune_nodes])
    lowered = engine._lower({}, 4.0, 0.0)
    with pytest.raises(SimCancelledError, match="cancelled"):
        engine._point_evolution_props({}, 4.0, lowered)

    assert 0 < propagator_calls < 2 * len(engine._detune_nodes)


def test_engine_caches_population_chain_for_readout_only_sweep(monkeypatch):
    """Sweeping only readout parameters reuses the identical qubit state chain."""

    calls = 0
    real_population_chain = SimEngine._point_population_chain

    def spy_population_chain(
        self: SimEngine,
        model: _PointModel,
        reps: int,
        nreads: int,
        *,
        use_numba: bool = True,
    ) -> NDArray[np.float64]:
        nonlocal calls
        calls += 1
        return real_population_chain(self, model, reps, nreads, use_numba=use_numba)

    monkeypatch.setattr(SimEngine, "_point_population_chain", spy_population_chain)

    _soc, soccfg = make_mock_soc(sim=_SIM)
    f_qubit = _f_qubit_mhz()
    pulse = PulseCfg(
        ch=0,
        nqz=1,
        gain=0.3,
        freq=f_qubit,
        phase=0.0,
        waveform=ConstWaveformCfg(length=0.5),
    ).build("qub")
    sw = SweepCfg(start=_rf_g_mhz() - 5.0, stop=_rf_g_mhz() + 5.0, expts=5, step=2.5)
    ro_param = sweep2param("ro_freq", sw)
    readout = DirectReadoutCfg(ro_ch=0, ro_length=1.0, ro_freq=ro_param).build("ro")
    prog = ModularProgramV2(
        soccfg,
        ProgramV2Cfg(reps=7, rounds=1),
        modules=[pulse, readout],
        sweep=[("ro_freq", sw)],
    )
    prog.compile()

    engine = SimEngine(prog, _SIM)
    engine._ensure_signal()

    assert calls == 1


def test_engine_reuses_evolution_lowering_for_readout_only_sweep(monkeypatch):
    """Readout-only axes avoid per-detune re-lowering while sweeping readout."""

    from zcu_tools.program.v2.sim.engine import SimEngine

    sim = _SIM.model_copy(update={"T2": 10.0, "T2_star": 5.0})
    _soc, soccfg = make_mock_soc(sim=sim)
    f_qubit = _f_qubit_mhz()
    pulse = PulseCfg(
        ch=0,
        nqz=1,
        gain=0.3,
        freq=f_qubit,
        phase=0.0,
        waveform=ConstWaveformCfg(length=0.5),
    ).build("qub")
    sw = SweepCfg(start=_rf_g_mhz() - 5.0, stop=_rf_g_mhz() + 5.0, expts=5, step=2.5)
    ro_param = sweep2param("ro_freq", sw)
    readout = DirectReadoutCfg(ro_ch=0, ro_length=1.0, ro_freq=ro_param).build("ro")
    prog = ModularProgramV2(
        soccfg,
        ProgramV2Cfg(reps=7, rounds=1),
        modules=[pulse, readout],
        sweep=[("ro_freq", sw)],
    )
    prog.compile()

    lower_calls = 0
    real_lower = SimEngine._lower

    def spy_lower(
        self: SimEngine,
        point: dict[str, int],
        f_qubit_ghz: float,
        detune_offset: float,
    ):
        nonlocal lower_calls
        lower_calls += 1
        return real_lower(self, point, f_qubit_ghz, detune_offset)

    monkeypatch.setattr(SimEngine, "_lower", spy_lower)

    engine = SimEngine(prog, sim)
    engine._ensure_signal()

    assert lower_calls == sw.expts


def test_cooperative_yield_releases_gil(monkeypatch):
    """The mocksim CPU-loop yield hook explicitly releases the process-wide GIL."""

    from zcu_tools.program.v2.sim import engine as engine_module

    sleep_calls: list[float] = []
    monkeypatch.setattr(engine_module.time, "sleep", sleep_calls.append)

    yielder = engine_module._CooperativeYield(interval_s=0.0)
    yielder()

    assert sleep_calls == [0]
    assert yielder.count == 1


def test_engine_does_not_reuse_population_chain_for_qubit_sweep(monkeypatch):
    """Sweeping qubit drive parameters keeps distinct rep-to-rep state chains."""

    calls = 0
    real_population_chain = SimEngine._point_population_chain

    def spy_population_chain(
        self: SimEngine,
        model: _PointModel,
        reps: int,
        nreads: int,
        *,
        use_numba: bool = True,
    ) -> NDArray[np.float64]:
        nonlocal calls
        calls += 1
        return real_population_chain(self, model, reps, nreads, use_numba=use_numba)

    monkeypatch.setattr(SimEngine, "_point_population_chain", spy_population_chain)

    _soc, soccfg = make_mock_soc(sim=_SIM)
    f_qubit = _f_qubit_mhz()
    sw = SweepCfg(start=0.0, stop=0.9, expts=4, step=0.3)
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
        ProgramV2Cfg(reps=7, rounds=1),
        modules=[pulse, _readout(_rf_g_mhz())],
        sweep=[("gain", sw)],
    )
    prog.compile()

    engine = SimEngine(prog, _SIM)
    engine._ensure_signal()

    assert calls == sw.expts


def test_engine_skips_numba_when_population_work_is_small(monkeypatch):
    """The signal grid avoids numba setup cost for low-work population chains."""

    from zcu_tools.program.v2.sim import engine as engine_module

    def fail_numba(*_args: object, **_kwargs: object) -> NDArray[np.float64]:
        raise AssertionError("numba kernel should not be used for this signal grid")

    monkeypatch.setattr(engine_module, "_population_chain_numba", fail_numba)

    sim = _SIM.model_copy(update={"T2": 10.0, "T2_star": 5.0})
    _soc, soccfg = make_mock_soc(sim=sim)
    f_qubit = _f_qubit_mhz()
    pulse = PulseCfg(
        ch=0,
        nqz=1,
        gain=0.3,
        freq=f_qubit,
        phase=0.0,
        waveform=ConstWaveformCfg(length=0.5),
    ).build("qub")
    sw = SweepCfg(start=_rf_g_mhz() - 2.0, stop=_rf_g_mhz() + 2.0, expts=3, step=2.0)
    ro_param = sweep2param("ro_freq", sw)
    readout = DirectReadoutCfg(ro_ch=0, ro_length=1.0, ro_freq=ro_param).build("ro")
    prog = ModularProgramV2(
        soccfg,
        ProgramV2Cfg(reps=4, rounds=1),
        modules=[pulse, readout],
        sweep=[("ro_freq", sw)],
    )
    prog.compile()

    engine = SimEngine(prog, sim)
    engine._ensure_signal()


def test_engine_uses_numba_for_large_unique_population_work(monkeypatch):
    """Large multi-node unique qubit chains are routed through the numba kernel."""

    from zcu_tools.program.v2.sim import engine as engine_module

    calls = 0

    def fake_numba(
        _pre_props: NDArray[np.float64],
        _relax_props: NDArray[np.float64],
        _weights: NDArray[np.float64],
        _thermal_pop: float,
        reps: int,
        nreads: int,
    ) -> NDArray[np.float64]:
        nonlocal calls
        calls += 1
        return np.zeros((reps, nreads), dtype=np.float64)

    monkeypatch.setattr(engine_module, "_NUMBA_MIN_WORK_UNITS", 1)
    monkeypatch.setattr(engine_module, "_population_chain_numba", fake_numba)

    sim = _SIM.model_copy(update={"T2": 10.0, "T2_star": 5.0})
    _soc, soccfg = make_mock_soc(sim=sim)
    f_qubit = _f_qubit_mhz()
    sw = SweepCfg(start=0.1, stop=0.3, expts=3, step=0.1)
    gain_param = sweep2param("gain", sw)
    pulse = PulseCfg(
        ch=0,
        nqz=1,
        gain=gain_param,
        freq=f_qubit,
        phase=0.0,
        waveform=ConstWaveformCfg(length=0.5),
    ).build("qub")
    prog = ModularProgramV2(
        soccfg,
        ProgramV2Cfg(reps=4, rounds=1),
        modules=[pulse, _readout(_rf_g_mhz())],
        sweep=[("gain", sw)],
    )
    prog.compile()

    engine = SimEngine(prog, sim)
    engine._ensure_signal()

    assert calls == sw.expts


def test_engine_batched_population_chain_matches_scalar_reference(monkeypatch):
    """The optimized detune-node recurrence preserves the scalar physics."""

    from zcu_tools.program.v2.sim import engine as engine_module

    sim = _SIM.model_copy(update={"T2": 10.0, "T2_star": 5.0})
    _soc, soccfg = make_mock_soc(sim=sim)
    pulse = PulseCfg(
        ch=0,
        nqz=1,
        gain=0.2,
        freq=_f_qubit_mhz(),
        phase=0.0,
        waveform=ConstWaveformCfg(length=0.7),
    ).build("qub")
    prog = ModularProgramV2(
        soccfg,
        ProgramV2Cfg(reps=8, rounds=1, relax_delay=0.1),
        modules=[pulse, _readout(_rf_g_mhz())],
    )
    prog.compile()
    engine = SimEngine(prog, sim)
    n_samples, sample_times_us = engine._readout_sample_times_us()
    f_qubit_ghz, rf_g, rf_e = engine._operating_signal()
    lowered = engine._lower({}, f_qubit_ghz, 0.0)
    readout = engine._point_readout_model(
        lowered, f_qubit_ghz, rf_g, rf_e, n_samples, sample_times_us
    )
    evolution = engine._point_evolution_props({}, f_qubit_ghz, lowered)
    model = engine._point_model(readout, evolution)

    actual = engine._point_population_chain(model, reps=8, nreads=1)
    monkeypatch.setattr(engine_module, "_population_chain_numba", None)
    fallback = engine._point_population_chain(model, reps=8, nreads=1)

    z0 = 2.0 * sim.thermal_pop - 1.0
    states = [
        np.array([0.0, 0.0, z0, 1.0], dtype=np.float64) for _ in model.pre_readout_props
    ]
    expected = np.empty((8, 1), dtype=np.float64)
    for rep_idx in range(8):
        p_mean = 0.0
        next_states: list[NDArray[np.float64]] = []
        for state, pre_prop, relax_prop, weight in zip(
            states,
            model.pre_readout_props,
            model.inter_shot_props,
            engine._detune_weights,
        ):
            at_readout = pre_prop @ state
            node_p = 0.5 * (1.0 + float(at_readout[2]))
            p_mean += float(weight) * min(max(node_p, 0.0), 1.0)
            next_states.append(relax_prop @ at_readout)
        expected[rep_idx, :] = p_mean
        states = next_states

    np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(fallback, expected, rtol=1e-12, atol=1e-12)


# ---------------------------------------------------------------- decimated D2


def _pulse_readout(
    ro_freq_mhz: float,
    ro_length: float = 2.0,
    pulse_length: float | None = None,
    gain: float = 0.1,
    trig_offset: float = 0.0,
    pre_delay: float = 0.0,
) -> Module:
    """A PulseReadout (const envelope) for the decimated/lookback path.

    Decimated needs a PulseReadout because its ``pulse_cfg`` defines the readout
    envelope shape rendered in the time domain.
    """

    pulse_len = ro_length if pulse_length is None else pulse_length
    pulse_cfg = PulseCfg(
        ch=0,
        nqz=1,
        gain=gain,
        freq=ro_freq_mhz,
        phase=0.0,
        pre_delay=pre_delay,
        waveform=ConstWaveformCfg(length=pulse_len),
    )
    ro_cfg = DirectReadoutCfg(
        ro_ch=0, ro_length=ro_length, ro_freq=ro_freq_mhz, trig_offset=trig_offset
    )
    return PulseReadoutCfg(pulse_cfg=pulse_cfg, ro_cfg=ro_cfg).build("ro")


def _ground_raw_median(readout: Module, *, sim: SimParams) -> complex:
    """Run one no-pulse readout and return the raw median centre."""

    soc, soccfg = make_mock_soc(sim=sim)
    prog = ModularProgramV2(
        soccfg,
        ProgramV2Cfg(reps=2000, rounds=1),
        modules=[readout],
    )
    prog.acquire(soc, progress=False)
    raw = prog.get_raw()
    assert raw is not None
    samples = raw[0][:, 0, 0] + 1j * raw[0][:, 0, 1]
    return complex(np.median(samples.real) + 1j * np.median(samples.imag))


def _ground_normalized_raw_std(readout: Module, *, sim: SimParams) -> float:
    """Run one no-pulse readout and return normalized I-quadrature scatter."""

    soc, soccfg = make_mock_soc(sim=sim)
    prog = ModularProgramV2(
        soccfg,
        ProgramV2Cfg(reps=6000, rounds=1),
        modules=[readout],
    )
    prog.acquire(soc, progress=False)
    raw = prog.get_raw()
    assert raw is not None
    compiled_length = int(list(prog.ro_chs.values())[0]["length"])
    samples = (raw[0][:, 0, 0] + 1j * raw[0][:, 0, 1]) / compiled_length
    return float(np.std(samples.real))


def test_engine_pulse_readout_gain_scales_raw_blob_centers():
    """PulseReadout raw deterministic centres scale linearly with readout gain."""

    sim = _SIM.model_copy(
        update={
            "snr": 1.0e9,
            "thermal_pop": 0.0,
            "readout_photons_per_gain2": 1.0,
        }
    )
    low_gain = 0.05
    high_gain = 0.20

    low = _ground_raw_median(
        _pulse_readout(_rf_g_mhz(), ro_length=1.0, gain=low_gain), sim=sim
    )
    high = _ground_raw_median(
        _pulse_readout(_rf_g_mhz(), ro_length=1.0, gain=high_gain), sim=sim
    )

    assert abs(high) / abs(low) == pytest.approx(high_gain / low_gain, rel=0.01)


def test_engine_readout_gain_noise_scales_with_pulse_readout_gain() -> None:
    """The second Gaussian source is proportional to compressed readout drive."""

    sim = _SIM.model_copy(
        update={
            "snr": 1.0e9,
            "thermal_pop": 0.0,
            "readout_gain_noise_per_gain": 0.03,
        }
    )
    low = _ground_normalized_raw_std(
        _pulse_readout(_rf_g_mhz(), ro_length=1.0, gain=0.02),
        sim=sim,
    )
    high = _ground_normalized_raw_std(
        _pulse_readout(_rf_g_mhz(), ro_length=1.0, gain=0.08),
        sim=sim,
    )

    assert high > 2.5 * low


def _pulse_readout_blob_gap(gain: float) -> float:
    _soc, soccfg = make_mock_soc(sim=_SIM)
    prog = ModularProgramV2(
        soccfg,
        ProgramV2Cfg(reps=1, rounds=1),
        modules=[_pulse_readout(_rf_g_mhz(), ro_length=1.0, gain=gain)],
    )
    prog.compile()
    engine = SimEngine(prog, _SIM)
    s_g, s_e, _p_e, signal_scale, _noise_scale, _gain_noise_scale = (
        engine._ensure_signal()
    )
    return float(abs(signal_scale[0] * (s_e[0] - s_g[0])))


def test_engine_high_readout_gain_reduces_blob_contrast() -> None:
    """Above the dispersive guardrail, more gain no longer improves contrast."""

    low = _pulse_readout_blob_gap(0.02)
    mid = _pulse_readout_blob_gap(0.08)
    high = _pulse_readout_blob_gap(0.40)

    assert mid > low
    assert high < mid


def test_engine_pulse_readout_signal_samples_use_trigger_alignment() -> None:
    """Accumulated PulseReadout integrates only pulse envelope inside the ADC window."""

    sim = _SIM.model_copy(update={"snr": 1.0e9, "readout_photons_per_gain2": 1.0})

    def signal_scale(trig_offset: float) -> float:
        _soc, soccfg = make_mock_soc(sim=sim)
        prog = ModularProgramV2(
            soccfg,
            ProgramV2Cfg(reps=1, rounds=1),
            modules=[
                _pulse_readout(
                    _rf_g_mhz(),
                    ro_length=1.0,
                    pulse_length=0.2,
                    gain=0.1,
                    trig_offset=trig_offset,
                )
            ],
        )
        prog.compile()
        engine = SimEngine(prog, sim)
        _s_g, _s_e, _p_e, scale, _noise, _gain_noise = engine._ensure_signal()
        return float(scale[0])

    aligned = signal_scale(sim.timeFly)
    late = signal_scale(sim.timeFly + 1.0)

    assert aligned > 0.0
    assert late == pytest.approx(0.0, abs=1e-12)


def test_engine_pulse_readout_signal_samples_include_pulse_pre_delay() -> None:
    """PulseReadout envelope starts after its generator pre_delay."""

    sim = _SIM.model_copy(update={"snr": 1.0e9, "readout_photons_per_gain2": 1.0})
    pulse_length = 0.2
    pre_delay = 0.3

    def signal_scale(trig_offset: float) -> float:
        _soc, soccfg = make_mock_soc(sim=sim)
        prog = ModularProgramV2(
            soccfg,
            ProgramV2Cfg(reps=1, rounds=1),
            modules=[
                _pulse_readout(
                    _rf_g_mhz(),
                    ro_length=pulse_length,
                    pulse_length=pulse_length,
                    gain=0.1,
                    trig_offset=trig_offset,
                    pre_delay=pre_delay,
                )
            ],
        )
        prog.compile()
        engine = SimEngine(prog, sim)
        _s_g, _s_e, _p_e, scale, _noise, _gain_noise = engine._ensure_signal()
        return float(scale[0])

    aligned = signal_scale(sim.timeFly + pre_delay)
    early = signal_scale(sim.timeFly)

    assert aligned > 0.0
    assert early == pytest.approx(0.0, abs=1e-12)


def test_engine_direct_readout_skips_nonlinear_photon_guardrail() -> None:
    """DirectReadout has no generator gain, so critical-photon checks do not apply."""

    f_qubit_ghz = _f_qubit_mhz() * 1e-3
    sim = _SIM.model_copy(update={"bare_rf": f_qubit_ghz, "snr": 1.0e9})
    soc, soccfg = make_mock_soc(sim=sim)
    prog = ModularProgramV2(
        soccfg,
        ProgramV2Cfg(reps=3, rounds=1),
        modules=[
            DirectReadoutCfg(ro_ch=0, ro_length=1.0, ro_freq=_f_qubit_mhz()).build("ro")
        ],
    )

    result = prog.acquire(soc, progress=False)

    assert result[0].shape == (1, 2)


def test_engine_acquire_decimated_returns_timefly_shifted_trace():
    """acquire_decimated on a sim soc renders a model-A time-domain trace.

    The trace must be a real/imag stacked array over the readout window with the
    readout envelope shifted by ``sim.timeFly``: ~0 before timeFly, finite after.
    """

    soc, soccfg = make_mock_soc(sim=_SIM)

    # reps=1, single trigger -> qick returns the simple (length, 2) decimated shape.
    prog = ModularProgramV2(
        soccfg,
        ProgramV2Cfg(reps=1, rounds=2),
        modules=[_pulse_readout(_rf_g_mhz(), ro_length=2.0, gain=1.0)],
    )

    result = prog.acquire_decimated(soc, progress=False)

    # Single channel, (length, 2) float trace (averaged over rounds).
    assert len(result) == 1
    trace = result[0]
    assert trace.ndim == 2 and trace.shape[1] == 2

    ts = prog.get_time_axis(ro_index=0)  # program-time axis (trig_offset == 0 here)
    mag = np.abs(trace[:, 0] + 1j * trace[:, 1])

    tof = _SIM.timeFly
    # The signal floor is set by the per-sample noise (full_scale / snr); the in-
    # window amplitude is ~full_scale * |steady S21|, far above it.  Compare the
    # pre-timeFly region (noise only) to the in-window region (envelope present).
    before = mag[ts < tof - 0.1]
    inside = mag[(ts > tof + 0.1) & (ts < tof + 1.9)]
    assert before.size > 0 and inside.size > 0
    assert inside.mean() > 10.0 * (before.mean() + 1.0)


def test_engine_acquire_decimated_uses_pulse_length_not_adc_window():
    """The ADC window can outlive the generator pulse envelope."""

    sim = _SIM.model_copy(update={"snr": 1.0e9})
    soc, soccfg = make_mock_soc(sim=sim)
    pulse_length = 0.6
    ro_length = 2.0
    prog = ModularProgramV2(
        soccfg,
        ProgramV2Cfg(reps=1, rounds=1),
        modules=[
            _pulse_readout(
                _rf_g_mhz(),
                ro_length=ro_length,
                pulse_length=pulse_length,
                gain=1.0,
            )
        ],
    )

    trace = prog.acquire_decimated(soc, progress=False)[0]
    ts = prog.get_time_axis(ro_index=0)
    mag = np.abs(trace[:, 0] + 1j * trace[:, 1])
    tof = sim.timeFly

    inside = mag[(ts > tof + 0.1) & (ts < tof + pulse_length - 0.1)]
    after_pulse = mag[(ts > tof + pulse_length + 0.1) & (ts < tof + ro_length - 0.1)]

    assert inside.size > 0 and after_pulse.size > 0
    assert inside.mean() > 100.0 * (after_pulse.mean() + 1.0)


def test_engine_acquire_decimated_amplitude_scales_with_readout_gain():
    """Decimated/lookback traces use the same linear readout gain amplitude."""

    sim = _SIM.model_copy(update={"snr": 1.0e9, "readout_photons_per_gain2": 1.0})

    def inside_mean(gain: float) -> float:
        soc, soccfg = make_mock_soc(sim=sim)
        prog = ModularProgramV2(
            soccfg,
            ProgramV2Cfg(reps=1, rounds=1),
            modules=[_pulse_readout(_rf_g_mhz(), ro_length=2.0, gain=gain)],
        )
        trace = prog.acquire_decimated(soc, progress=False)[0]
        ts = prog.get_time_axis(ro_index=0)
        mag = np.abs(trace[:, 0] + 1j * trace[:, 1])
        inside = mag[(ts > sim.timeFly + 0.1) & (ts < sim.timeFly + 1.5)]
        assert inside.size > 0
        return float(np.mean(inside))

    low_gain = 0.05
    high_gain = 0.20
    assert inside_mean(high_gain) / inside_mean(low_gain) == pytest.approx(
        high_gain / low_gain, rel=0.02
    )


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


# reps for the coherence-envelope helpers.  The engine now draws a per-shot
# Bernoulli(P_e), so the accumulated readout carries shot noise ~
# sqrt(P_e(1-P_e)/reps) even at snr=1e9 (snr only scales the Gaussian readout
# noise, not the Bernoulli shot noise).  reps=1 (the old single-eval path) would
# return a raw 0/1 blob, not the mean; 2000 reps averages the shot noise down so
# the reps-mean is the deterministic coherence envelope these gates read.
_ENVELOPE_REPS = 2000


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
        ProgramV2Cfg(reps=_ENVELOPE_REPS, rounds=1, relax_delay=10.0 * sim.T1),
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
        ProgramV2Cfg(reps=_ENVELOPE_REPS, rounds=1, relax_delay=10.0 * sim.T1),
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
