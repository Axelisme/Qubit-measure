"""Synthetic signal generators for the dry-run Node bodies.

MockSoc.acquire returns white noise — not fittable. So the dry-run Node bodies
synthesise a physically-plausible *complex* IQ signal (the real fitting pipeline
does ``rotate2real`` first), shaped by the same base functions the fitters fit:

- ``lorentzian_dip``  → ``lorfunc``  → ``fit_qubit_freq`` (qubit_freq)
- ``exp_decay``       → ``expfunc``  → ``fit_decay`` (t1)
- ``rabi_oscillation``→ ``cosfunc``  → ``fit_rabi`` (lenrabi)
- ``decay_cos``       → ``decaycos`` → ``fit_decay_fringe`` (t2ramsey / t2echo)
- ``gaussian_peak_2d``→ argmax (ro_optimize, no fit — a 2D landscape)
- ``variance_curve``  → variance read (mist, no fit — a 1D disturbance curve)

with additive noise and an arbitrary readout phase. Fitting the synthetic signal
with the real ``fit_*`` (or taking the argmax / variance for the fit-less ones)
recovers the planted parameters, so the whole acquire→fit→produce path is
exercised end-to-end without hardware.

Determinism: a per-call ``seed`` (derived from the flux index) keeps a sweep
reproducible without using process-global RNG state.
"""

from __future__ import annotations

import time

import numpy as np
from numpy.typing import NDArray
from typing_extensions import Any, Callable

from zcu_tools.utils.fitting.base import cosfunc, decaycos, expfunc, lorfunc
from zcu_tools.utils.process import rotate2real

# The default per-flux-point delay (seconds) the controller seeds into a freshly
# placed Node's params, so the synthetic liveplot advances visibly instead of the
# whole sweep finishing in milliseconds. It is a *param default* (the user can
# tune it, the run reads it from params) — NOT a produce-time fallback, so a
# directly-constructed Node with no params (tests) sleeps zero and runs instantly.
# Phase B leaves it at 0 (the real acquire IS the wall-clock cost). One second per
# point makes each Node's liveplot clearly watchable as the sweep advances.
DEFAULT_ACQUIRE_DELAY = 1.0


def simulate_acquire_delay(seconds: float) -> None:
    """Sleep ``seconds`` to emulate a real acquire's duration (worker thread).

    Called inside a Node's ``produce`` on the run worker, NOT the Qt main
    thread, so it never freezes the UI — it only paces how fast the sweep fills
    rows, letting the main-thread liveplot redraw between points. A
    non-positive value is a no-op (so a Node with no delay param runs instantly).
    """
    if seconds > 0:
        time.sleep(seconds)


def resolve_acquire_delay(params: Any) -> float:
    """The acquire delay (seconds) from a Node's params, or ``0`` if unset/bad.

    Missing → 0 (no sleep): a directly-constructed Node (tests) runs instantly;
    a GUI-placed Node carries ``DEFAULT_ACQUIRE_DELAY`` in its params (seeded by
    the controller). The prototype's param field is free text, so an unparseable
    value degrades to 0 rather than failing the sweep.
    """
    try:
        value = params.get("acquire_delay")
    except AttributeError:
        return 0.0
    if value is None or value == "":
        return 0.0
    try:
        return max(0.0, float(value))
    except (ValueError, TypeError):
        return 0.0


# How many running-average rounds a synthetic acquire emulates. A real acquire
# averages many shots; each round's running-averaged trace is noisier early and
# settles as rounds accumulate. The default keeps the per-round redraw cadence
# watchable (delay / n_rounds per round); the user tunes it via the ``rounds``
# param. A directly-constructed Node (tests) gets 1 round (instant single pass).
DEFAULT_ROUNDS = 10


def resolve_rounds(params: Any) -> int:
    """The emulated round count from a Node's params, or ``1`` if unset/bad.

    Missing → 1 (a single pass, no accumulation): tests run instantly. A
    GUI-placed Node carries ``DEFAULT_ROUNDS`` (seeded by the controller). An
    unparseable value degrades to 1 rather than failing the sweep.
    """
    try:
        value = params.get("rounds")
    except AttributeError:
        return 1
    if value is None or value == "":
        return 1
    try:
        return max(1, int(value))
    except (ValueError, TypeError):
        return 1


def accumulate_rounds(
    make_round: Callable[[int], NDArray[Any]],
    n_rounds: int,
    on_round: Callable[[NDArray[Any], int], None],
    delay: float = 0.0,
) -> NDArray[Any]:
    """Emulate a multi-round acquire: running-average ``n_rounds`` noisy passes.

    Each round, ``make_round(round_idx)`` returns that round's raw signal (a
    fresh noise realisation of the same underlying physics — the Node uses
    ``seed=base+round_idx`` so each round differs). The running average over
    rounds 0..k is fed to ``on_round(running_average, round_idx)`` — where the
    Node overwrites its Result row and notifies the main thread — so the same
    row visibly settles round by round (noise ∝ 1/√k). ``delay`` (the acquire's
    total wall-clock) is split evenly across rounds so the redraws pace out.

    Returns the final running average (all ``n_rounds`` folded in) for the
    Node to fit. ``n_rounds`` is at least 1 (a single pass = no accumulation).
    """
    n = max(1, n_rounds)
    per_round_delay = delay / n if delay > 0 else 0.0
    running_sum = make_round(0)
    running_avg = running_sum
    on_round(running_avg, 0)
    simulate_acquire_delay(per_round_delay)
    for k in range(1, n):
        running_sum = running_sum + make_round(k)
        running_avg = running_sum / (k + 1)
        on_round(running_avg, k)
        simulate_acquire_delay(per_round_delay)
    return running_avg


def signal_to_real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    """PCA-rotate a complex IQ trace to the real axis and normalise to [0, 1].

    The shared pre-fit step for the 1D experiments (t1 / lenrabi / t2ramsey /
    t2echo / mist): ``rotate2real`` then min-max normalise. The fitters
    (``fit_decay`` / ``fit_rabi`` / ``fit_decay_fringe``) are baseline-agnostic,
    so no orientation flip is needed (unlike qubit_freq's dip). Min-max scaling
    is linear, so it does NOT change the SNR (signal and noise scale together) —
    a low-SNR row reads as noisier texture in the colormap, not a flatter one.
    """
    real = rotate2real(signals.astype(np.complex128)).real
    lo, hi = float(real.min()), float(real.max())
    return (real - lo) / (hi - lo + 1e-12)


def parse_linear_axis(
    spec: Any, default: tuple[float, float, int]
) -> NDArray[np.float64]:
    """Parse a free-text "start,stop,npts" sweep axis (or a 3-tuple) to a linspace.

    The prototype's sweep fields are free text; a malformed value degrades to
    ``default`` rather than failing the sweep (shared by t1 / lenrabi / t2ramsey
    / t2echo / mist, whose trailing axis is a simple linspace).
    """
    try:
        if isinstance(spec, str) and spec.strip():
            start, stop, npts = spec.split(",")
            lo, hi, n = float(start), float(stop), int(npts)
        elif isinstance(spec, (tuple, list)) and len(spec) == 3:
            lo, hi, n = float(spec[0]), float(spec[1]), int(spec[2])
        else:
            lo, hi, n = default
    except (ValueError, TypeError):
        lo, hi, n = default
    return np.linspace(lo, hi, max(2, n))


# --- flux-dependent drift + SNR (the prototype's adaptivity test substrate) ---
#
# Real fluxonium parameters vary with flux; a fixed plant can't exercise the
# closed-loop feedback (the predictor would have nothing to track). So each
# physical quantity drifts parabolically with flux (sweet-spot-like), and the
# signal SNR varies sinusoidally down to 0 at its troughs — those flux points are
# pure noise, so the fit-quality gate rejects them (no calibrate), testing the
# feedback's robustness to dead points. Module constants, not user params.
FLUX_DRIFT_CENTER = 0.5  # the parabola's vertex (sweet spot) in flux units
SNR_PERIOD = 0.35  # flux period of the SNR sinusoid
SNR_PHASE = 0.15  # flux phase offset (so troughs land mid-sweep, not at flux 0)


def flux_drift(
    flux: float, baseline: float, amplitude: float, center: float = FLUX_DRIFT_CENTER
) -> float:
    """A physical quantity's flux dependence: a parabola peaking at ``center``.

    ``baseline + amplitude * (flux - center)**2`` — a sweet-spot-like curve the
    predictor must track. Each quantity (qubit freq, t1, …) uses its own
    baseline / amplitude so they drift independently across the sweep.
    """
    return baseline + amplitude * (flux - center) ** 2


def flux_snr(flux: float) -> float:
    """The signal-to-noise coefficient at ``flux`` — sinusoidal in [0, 1].

    ``(1 + sin(2π·flux/period + phase)) / 2`` reaches 0 at its troughs, where the
    synthetic signal is all noise (SNR = 0) and the fit-quality gate discards the
    point. Multiplies the signal amplitude in the synth functions.
    """
    return 0.5 * (1.0 + np.sin(2.0 * np.pi * flux / SNR_PERIOD + SNR_PHASE))


def is_good_fit(
    real: NDArray[np.float64], fit_curve: NDArray[np.float64], threshold: float = 0.2
) -> bool:
    """Whether a fit is good enough to trust (the runner module's mean_err gate).

    Compares the mean absolute residual to the fit's peak-to-peak span: a good
    fit tracks the signal so the residual is a small fraction of the span. At an
    SNR-trough flux point the signal is pure noise — the fitted curve is nearly
    flat (tiny span) while the residual is large, so this returns False, and the
    Node omits that point's provides key (no downstream contamination) and skips
    calibration. Mirrors ``mean_err < threshold * ptp(fit)`` per experiment.
    """
    fit = np.asarray(fit_curve, dtype=np.float64)
    span = float(np.ptp(fit))
    if span <= 0 or not np.all(np.isfinite(fit)):
        return False
    residual = float(np.mean(np.abs(np.asarray(real, dtype=np.float64) - fit)))
    return residual < threshold * span


def _phase_and_noise(
    real: NDArray[np.float64],
    rng: np.random.RandomState,
    noise: float,
    phase_deg: float,
) -> NDArray[np.complex128]:
    """Turn a real curve into a complex IQ signal with an arbitrary readout
    phase and additive complex Gaussian noise (what acquire would return)."""
    phase = np.deg2rad(phase_deg)
    signal = real.astype(np.complex128) * np.exp(1j * phase)
    signal += noise * (rng.randn(*real.shape) + 1j * rng.randn(*real.shape))
    return signal


def lorentzian_dip(
    freqs: NDArray[np.float64],
    center: float,
    fwhm: float,
    *,
    depth: float = 0.3,
    baseline: float = 0.5,
    noise: float = 0.02,
    phase_deg: float = 37.0,
    snr: float = 1.0,
    seed: int = 0,
) -> NDArray[np.complex128]:
    """A qubit-frequency resonance: a Lorentzian dip at ``center`` of width
    ``fwhm`` (= 2·gamma), as a complex IQ signal. Fits with ``fit_qubit_freq``.
    ``snr`` (∈[0,1]) scales the dip depth — at 0 the signal is pure noise.
    """
    rng = np.random.RandomState(seed)
    gamma = fwhm / 2.0
    # lorfunc params: [y0, slope, yscale, x0, gamma]; negative yscale → a dip
    real = lorfunc(freqs, baseline, 0.0, -depth * snr, center, gamma)
    return _phase_and_noise(real, rng, noise, phase_deg)


def exp_decay(
    times: NDArray[np.float64],
    t1: float,
    *,
    amp: float = 0.8,
    baseline: float = 0.1,
    noise: float = 0.02,
    phase_deg: float = 37.0,
    snr: float = 1.0,
    seed: int = 0,
) -> NDArray[np.complex128]:
    """A T1 relaxation: an exponential decay with time constant ``t1`` (us), as a
    complex IQ signal. Fits with ``fit_decay`` to recover ``t1``. ``snr`` (∈[0,1])
    scales the decay amplitude — at 0 the signal is pure noise.
    """
    rng = np.random.RandomState(seed)
    # expfunc params: [y0, yscale, decay_time]
    real = expfunc(times, baseline, amp * snr, t1)
    return _phase_and_noise(real, rng, noise, phase_deg)


def rabi_oscillation(
    lengths: NDArray[np.float64],
    rabi_freq: float,
    *,
    amp: float = 0.45,
    baseline: float = 0.5,
    phase_deg_curve: float = 0.0,
    noise: float = 0.02,
    phase_deg: float = 37.0,
    snr: float = 1.0,
    seed: int = 0,
) -> NDArray[np.complex128]:
    """A length-Rabi oscillation: a cosine of frequency ``rabi_freq`` (1/us) vs
    pulse length, as a complex IQ signal. Fits with ``fit_rabi`` to recover the
    pi/pi-half lengths and the Rabi frequency. ``phase_deg_curve`` is the cosine's
    own phase (degrees), distinct from the readout ``phase_deg``; ``snr`` (∈[0,1])
    scales the oscillation amplitude — at 0 the signal is pure noise.
    """
    rng = np.random.RandomState(seed)
    # cosfunc params: [y0, yscale, freq (1/x), phase (deg)]
    real = cosfunc(lengths, baseline, amp * snr, rabi_freq, phase_deg_curve)
    return _phase_and_noise(real, rng, noise, phase_deg)


def decay_cos(
    times: NDArray[np.float64],
    t2: float,
    fringe_freq: float,
    *,
    amp: float = 0.45,
    baseline: float = 0.5,
    phase_deg_curve: float = 0.0,
    noise: float = 0.02,
    phase_deg: float = 37.0,
    snr: float = 1.0,
    seed: int = 0,
) -> NDArray[np.complex128]:
    """A Ramsey/echo fringe: a cosine of frequency ``fringe_freq`` (1/us) under an
    exponential envelope of time constant ``t2`` (us), as a complex IQ signal.
    Fits with ``fit_decay_fringe`` to recover ``t2`` and the detune. ``snr``
    (∈[0,1]) scales the fringe amplitude — at 0 the signal is pure noise.
    """
    rng = np.random.RandomState(seed)
    # decaycos params: [y0, yscale, freq, phase, decay_time]
    real = decaycos(times, baseline, amp * snr, fringe_freq, phase_deg_curve, t2)
    return _phase_and_noise(real, rng, noise, phase_deg)


def gaussian_peak_2d(
    freqs: NDArray[np.float64],
    gains: NDArray[np.float64],
    center_freq: float,
    center_gain: float,
    *,
    width_freq: float = 1.0,
    width_gain: float = 0.03,
    amp: float = 1.0,
    baseline: float = 0.05,
    noise: float = 0.01,
    seed: int = 0,
) -> NDArray[np.float64]:
    """A readout-optimisation landscape: a 2D Gaussian peak over freq × gain whose
    maximum marks the best readout point. Returned as a real ``(n_freq, n_gain)``
    SNR-like magnitude (ro_optimize has no fit — it takes the argmax), with
    additive non-negative noise. ``argmax`` recovers ``(center_freq, center_gain)``.
    """
    rng = np.random.RandomState(seed)
    ff, gg = np.meshgrid(freqs, gains, indexing="ij")
    real = baseline + amp * np.exp(
        -0.5 * (((ff - center_freq) / width_freq) ** 2)
        - 0.5 * (((gg - center_gain) / width_gain) ** 2)
    )
    real = real + noise * np.abs(rng.randn(*real.shape))  # magnitude → non-negative
    return real.astype(np.float64)


def variance_curve(
    gains: NDArray[np.float64],
    onset_gain: float,
    *,
    amp: float = 1.0,
    width: float = 0.05,
    baseline: float = 0.05,
    noise: float = 0.02,
    seed: int = 0,
) -> NDArray[np.float64]:
    """A MIST state-disturbance curve: a low-variance plateau that rises past an
    ``onset_gain`` (a logistic step), as a real ``(n_gain,)`` magnitude. MIST has
    no fit — it reads the variance directly; this gives a monotone curve whose
    onset is visible. Non-negative with additive noise.
    """
    rng = np.random.RandomState(seed)
    real = baseline + amp / (1.0 + np.exp(-(gains - onset_gain) / width))
    real = real + noise * np.abs(rng.randn(*real.shape))
    return real.astype(np.float64)
