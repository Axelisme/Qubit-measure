from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

import numpy as np
from numpy import ndarray
from zcu_tools.experiment import AbsExperiment
from zcu_tools.program.v1 import TwoToneProgram


class OptimizeExperiment(AbsExperiment):
    """Base class for readout optimization experiments."""

    pass


def measure_dist(soc, soccfg, cfg) -> Tuple[ndarray, ndarray, ndarray, ndarray]:
    """
    Measures the distance between ground and excited states.
    It performs two measurements, one with the qubit pulse on (excited) and one with it off (ground).
    """
    qub_pulse_gain = cfg["dac"]["qub_pulse"]["gain"]

    # Measure ground state
    cfg["dac"]["qub_pulse"]["gain"] = 0
    prog_g = TwoToneProgram(soccfg, cfg)
    (avgi_g, avgq_g), (stdi_g, stdq_g) = prog_g.acquire(soc, progress=False)

    # Measure excited state
    cfg["dac"]["qub_pulse"]["gain"] = qub_pulse_gain
    prog_e = TwoToneProgram(soccfg, cfg)
    (avgi_e, avgq_e), (stdi_e, stdq_e) = prog_e.acquire(soc, progress=False)

    # Restore gain
    cfg["dac"]["qub_pulse"]["gain"] = qub_pulse_gain

    dist_i = avgi_e[0] - avgi_g[0]
    dist_q = avgq_e[0] - avgq_g[0]
    noise_i = np.sqrt(stdi_g[0] ** 2 + stdi_e[0] ** 2)
    noise_q = np.sqrt(stdq_g[0] ** 2 + stdq_e[0] ** 2)

    return dist_i, dist_q, noise_i, noise_q


def result2snr(
    dist_i: ndarray, dist_q: ndarray, noise_i: ndarray, noise_q: ndarray
) -> Tuple[ndarray, ndarray]:
    """Calculates SNR from the distance and noise components."""
    contrast = dist_i + 1j * dist_q
    dist = np.abs(contrast)

    # Avoid division by zero
    dist[dist == 0] = 1e-6

    noise = np.sqrt((noise_i * dist_i) ** 2 + (noise_q * dist_q) ** 2) / dist

    snr = contrast / noise
    snr[np.isinf(snr)] = 0  # handle cases where noise is zero

    return snr, np.zeros_like(dist)


def snr_as_signal(ir: int, sum_d, sum2_d) -> ndarray:
    raise NotImplementedError(
        "This function is not applicable for v1 RO optimize experiments, use result2snr in templates."
    )


def snr_measure_fn(soc, soccfg, cfg) -> Callable:
    def measure_fn(
        cfg: Dict[str, Any], cb: Callable | None
    ) -> tuple[ndarray, ndarray, ndarray, ndarray]:
        return measure_dist(soc, soccfg, cfg)

    return measure_fn
