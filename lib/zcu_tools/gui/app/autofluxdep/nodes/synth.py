"""Synthetic signal generators for the dry-run Node bodies.

MockSoc.acquire returns white noise — not fittable. So the dry-run Node bodies
synthesise a physically-plausible *complex* IQ signal (the real fitting pipeline
does ``rotate2real`` first), shaped by the same base functions the fitters fit
(``lorfunc`` for a qubit-freq peak, ``expfunc`` for T1, ...), with additive noise
and an arbitrary readout phase. Fitting the synthetic signal with the real
``fit_*`` recovers the planted parameters, so the whole acquire→fit→produce path
is exercised end-to-end without hardware.

Determinism: a per-call ``seed`` (derived from the flux index) keeps a sweep
reproducible without using process-global RNG state.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from zcu_tools.utils.fitting.base import lorfunc


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
