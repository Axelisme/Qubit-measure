"""Independent analytic resonance traces and phase-spaced frequency grids."""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


def hanger_truth(
    freqs: NDArray[np.float64],
    *,
    freq: float,
    Ql: float,
    Qc_abs: float,
    phi: float,
    a0: complex,
    edelay: float,
    bg_amp_slope: float,
    bg_phase_curvature: float = 0.0,
) -> NDArray[np.complex128]:
    """Return analytic hanger samples with the same shape as ``freqs``.

    Frequencies use MHz; ``freq`` is the nonzero resonance frequency. ``Ql``
    and nonzero ``Qc_abs`` are quality factors, ``phi`` is radians, and ``a0``
    is the complex scale at resonance. ``edelay`` uses inverse MHz,
    ``bg_amp_slope`` uses inverse MHz, and ``bg_phase_curvature`` uses rad/MHz².
    Independent analytic oracle; intentionally does not call production calc.
    """
    detuning = Ql * (freqs / freq - 1.0)
    ideal = a0 * (1.0 - (Ql / Qc_abs) * np.exp(1j * phi) / (1.0 + 2j * detuning))
    centered = freqs - freq
    background = np.exp(bg_amp_slope * centered + 1j * bg_phase_curvature * centered**2)
    delay = np.exp(-1j * 2.0 * np.pi * freqs * edelay)
    return np.asarray(background * ideal * delay, dtype=np.complex128)


def transmission_truth(
    freqs: NDArray[np.float64],
    *,
    freq: float,
    Ql: float,
    a0: complex,
    edelay: float,
    bg_amp_slope: float,
    bg_phase_curvature: float = 0.0,
) -> NDArray[np.complex128]:
    """Return analytic transmission samples with the same shape as ``freqs``.

    Frequencies use MHz; ``freq`` is the nonzero resonance frequency, ``Ql``
    is the loaded quality factor, and ``a0`` is the complex scale at resonance.
    ``edelay`` and ``bg_amp_slope`` use inverse MHz; ``bg_phase_curvature``
    uses rad/MHz².
    Independent analytic oracle; intentionally does not call production calc.
    """
    detuning = Ql * (freqs / freq - 1.0)
    ideal = a0 / (1.0 + 2j * detuning)
    centered = freqs - freq
    background = np.exp(bg_amp_slope * centered + 1j * bg_phase_curvature * centered**2)
    delay = np.exp(-1j * 2.0 * np.pi * freqs * edelay)
    return np.asarray(background * ideal * delay, dtype=np.complex128)


def homophasal_like_grid(
    freq: float, Ql: float, half_span: float, count: int
) -> NDArray[np.float64]:
    """Return ``count`` frequencies equally spaced in analytic resonance phase.

    ``freq`` is the nonzero resonance frequency in MHz, ``Ql`` is a nonzero
    loaded quality factor, and ``half_span`` is the scan half-width in MHz.
    Endpoints are ``freq ± half_span``; inputs are not modified or validated.
    """

    def theta(value: float) -> float:
        return 2.0 * np.arctan(2.0 * Ql * (1.0 - value / freq))

    phases = np.linspace(theta(freq - half_span), theta(freq + half_span), count)
    return np.asarray(
        freq * (1.0 - np.tan(phases / 2.0) / (2.0 * Ql)),
        dtype=np.float64,
    )
