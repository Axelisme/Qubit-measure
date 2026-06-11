"""Analytic-limit tests for the TLS Bloch propagator.

These limits are the correctness gate and pin the sign/convention choices in
``zcu_tools.program.v2.sim.bloch``: free T1 decay, Rabi oscillation (incl.
generalized off-resonance frequency), Ramsey fringe frequency = detuning, and
echo refocusing of a static detuning.
"""

from __future__ import annotations

import numpy as np
import pytest
from zcu_tools.program.v2.sim.bloch import (
    Segment,
    evolve,
    excited_population,
    ground_state,
)


def _fringe_frequency(taus: np.ndarray, signal: np.ndarray) -> float:
    """Return the dominant oscillation frequency of ``signal`` over ``taus``.

    Zero-pads the FFT so the bin spacing is fine enough to resolve a frequency
    that does not fall exactly on a raw DFT bin (otherwise the peak is quantized
    to the coarse 1/T_window grid and biases the estimate).
    """

    dt = float(taus[1] - taus[0])
    n_fft = 1 << 16
    spectrum = np.abs(np.fft.rfft(signal - signal.mean(), n=n_fft))
    freqs = np.fft.rfftfreq(n_fft, dt)
    return float(freqs[int(np.argmax(spectrum))])


def test_free_t1_decay() -> None:
    """Excited population decays as exp(-t/T1) with no drive (Om=Delta=0)."""

    t1 = 2.0
    v_excited = np.array([0.0, 0.0, 1.0])  # opposite of ground_state(0): z=+1
    for t in [0.3, 0.7, 1.5, 3.0]:
        v = evolve(v_excited, [Segment(0.0, 0.0, 0.0, t, t1, None, 0.0)])
        assert excited_population(v) == pytest.approx(np.exp(-t / t1), abs=1e-9)


def test_rabi_on_resonance() -> None:
    """On-resonance Rabi from ground gives P_e = sin^2(Om t / 2), no decay."""

    omega = 2.0 * np.pi * 1.0
    v0 = ground_state(0.0)
    for t in [0.1, 0.25, 0.5, 0.9]:
        v = evolve(v0, [Segment(omega, 0.0, 0.0, t, None, None, 0.0)])
        assert excited_population(v) == pytest.approx(
            np.sin(omega * t / 2.0) ** 2, abs=1e-9
        )


def test_rabi_generalized_frequency() -> None:
    """Off-resonance Rabi oscillates at the generalized rate sqrt(Om^2 + Delta^2)."""

    omega = 2.0 * np.pi * 1.0
    delta = 2.0 * np.pi * 1.5
    v0 = ground_state(0.0)
    ts = np.linspace(0.0, 4.0, 800)
    pe = np.array(
        [
            excited_population(
                evolve(v0, [Segment(omega, delta, 0.0, t, None, None, 0.0)])
            )
            for t in ts
        ]
    )
    expected = np.sqrt(omega**2 + delta**2) / (2.0 * np.pi)
    # P_e ~ sin^2(Om_gen t / 2) oscillates at frequency Om_gen / (2 pi).
    assert _fringe_frequency(ts, pe) == pytest.approx(expected, rel=0.02)


def test_ramsey_fringe_frequency() -> None:
    """Ramsey (pi/2 - free(tau, Delta) - pi/2) fringes at the detuning Delta."""

    omega = 2.0 * np.pi * 1.0
    t_pi2 = (np.pi / 2.0) / omega
    delta = 2.0 * np.pi * 3.0
    v0 = ground_state(0.0)
    taus = np.linspace(0.0, 2.0, 800)
    pe = np.array(
        [
            excited_population(
                evolve(
                    v0,
                    [
                        Segment(omega, 0.0, 0.0, t_pi2, None, None, 0.0),
                        Segment(0.0, delta, 0.0, tau, None, None, 0.0),
                        Segment(omega, 0.0, 0.0, t_pi2, None, None, 0.0),
                    ],
                )
            )
            for tau in taus
        ]
    )
    assert _fringe_frequency(taus, pe) == pytest.approx(delta / (2.0 * np.pi), rel=0.02)


def test_echo_refocuses_static_detuning() -> None:
    """Echo (pi/2 - free - pi - free - pi/2) is insensitive to static Delta.

    Contrast with Ramsey: sweeping Delta barely moves P_e because the pi pulse
    refocuses the accumulated phase. With no decay it stays pinned near 0.
    """

    omega = 2.0 * np.pi * 1.0
    t_pi2 = (np.pi / 2.0) / omega
    t_pi = np.pi / omega
    tau = 0.5
    v0 = ground_state(0.0)
    pes = []
    for delta in [0.0, 2.0 * np.pi * 1.0, 2.0 * np.pi * 3.0, 2.0 * np.pi * 5.0]:
        v = evolve(
            v0,
            [
                Segment(omega, 0.0, 0.0, t_pi2, None, None, 0.0),
                Segment(0.0, delta, 0.0, tau / 2.0, None, None, 0.0),
                Segment(omega, 0.0, 0.0, t_pi, None, None, 0.0),
                Segment(0.0, delta, 0.0, tau / 2.0, None, None, 0.0),
                Segment(omega, 0.0, 0.0, t_pi2, None, None, 0.0),
            ],
        )
        pes.append(excited_population(v))
    pes_arr = np.array(pes)
    # Echo refocuses: P_e stays effectively constant across the Delta sweep.
    assert np.ptp(pes_arr) < 1e-6
