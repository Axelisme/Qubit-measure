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
from typing_extensions import Any

from zcu_tools.utils.fitting.base import cosfunc, decaycos, expfunc, lorfunc
from zcu_tools.utils.process import rotate2real

# The default per-flux-point delay (seconds) the controller seeds into a freshly
# placed Node's params, so the synthetic liveplot advances visibly instead of the
# whole sweep finishing in milliseconds. It is a *param default* (the user can
# tune it, the run reads it from params) — NOT a produce-time fallback, so a
# directly-constructed Node with no params (tests) sleeps zero and runs instantly.
# Phase B leaves it at 0 (the real acquire IS the wall-clock cost).
DEFAULT_ACQUIRE_DELAY = 0.1


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


def signal_to_real(signals: NDArray[np.complex128]) -> NDArray[np.float64]:
    """PCA-rotate a complex IQ trace to the real axis and normalise to [0, 1].

    The shared pre-fit step for the 1D experiments (t1 / lenrabi / t2ramsey /
    t2echo / mist): ``rotate2real`` then min-max normalise. The fitters
    (``fit_decay`` / ``fit_rabi`` / ``fit_decay_fringe``) are baseline-agnostic,
    so no orientation flip is needed (unlike qubit_freq's dip).
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
    seed: int = 0,
) -> NDArray[np.complex128]:
    """A qubit-frequency resonance: a Lorentzian dip at ``center`` of width
    ``fwhm`` (= 2·gamma), as a complex IQ signal. Fits with ``fit_qubit_freq``.
    """
    rng = np.random.RandomState(seed)
    gamma = fwhm / 2.0
    # lorfunc params: [y0, slope, yscale, x0, gamma]; negative yscale → a dip
    real = lorfunc(freqs, baseline, 0.0, -depth, center, gamma)
    return _phase_and_noise(real, rng, noise, phase_deg)


def exp_decay(
    times: NDArray[np.float64],
    t1: float,
    *,
    amp: float = 0.8,
    baseline: float = 0.1,
    noise: float = 0.02,
    phase_deg: float = 37.0,
    seed: int = 0,
) -> NDArray[np.complex128]:
    """A T1 relaxation: an exponential decay with time constant ``t1`` (us), as a
    complex IQ signal. Fits with ``fit_decay`` to recover ``t1``.
    """
    rng = np.random.RandomState(seed)
    # expfunc params: [y0, yscale, decay_time]
    real = expfunc(times, baseline, amp, t1)
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
    seed: int = 0,
) -> NDArray[np.complex128]:
    """A length-Rabi oscillation: a cosine of frequency ``rabi_freq`` (1/us) vs
    pulse length, as a complex IQ signal. Fits with ``fit_rabi`` to recover the
    pi/pi-half lengths and the Rabi frequency. ``phase_deg_curve`` is the cosine's
    own phase (degrees), distinct from the readout ``phase_deg``.
    """
    rng = np.random.RandomState(seed)
    # cosfunc params: [y0, yscale, freq (1/x), phase (deg)]
    real = cosfunc(lengths, baseline, amp, rabi_freq, phase_deg_curve)
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
    seed: int = 0,
) -> NDArray[np.complex128]:
    """A Ramsey/echo fringe: a cosine of frequency ``fringe_freq`` (1/us) under an
    exponential envelope of time constant ``t2`` (us), as a complex IQ signal.
    Fits with ``fit_decay_fringe`` to recover ``t2`` and the detune.
    """
    rng = np.random.RandomState(seed)
    # decaycos params: [y0, yscale, freq, phase, decay_time]
    real = decaycos(times, baseline, amp, fringe_freq, phase_deg_curve, t2)
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
