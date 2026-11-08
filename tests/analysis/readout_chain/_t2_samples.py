"""Independent ideal-pulse Bloch channels with longitudinal delay relaxation."""

import numpy as np
from zcu_tools.analysis.readout_chain import CycleModel, CycleNuisances, CycleSums

from ._cycle_samples import sampled_sums


def t2_nuisance(contrast: float) -> CycleNuisances:
    """Interior, noncollinear three-cloud truth with delay-independent mixing."""
    mixing = np.array([[0.85, 0.1, 0.05], [0.1, 0.8, 0.1], [0.05, 0.1, 0.85]])
    return CycleNuisances(
        np.array(
            [
                [mixing[0], mixing[1]],
                [mixing[0], mixing[1]],
                [(mixing[0] + mixing[1]) / 2, mixing[2]],
            ]
        ),
        contrast * np.array([[0.0, 0.0], [1.0, 0.0], [-0.04, 1.4]]),
        1.0,
        np.array([0.55, 0.3, 0.15]),
        np.array([0.95, 0.8, 0.95]),
        np.array([0.63, 0.27, 0.1]),
    )


def bloch_curves(
    axis: np.ndarray, parameters: dict[str, float], *, t1: float, echo: bool
) -> np.ndarray:
    """Propagate initial g/e Bloch vectors, then read P(e) after the analyzer.

    The affine z relaxation has equilibrium z=0.4. Ideal y-axis pi/2 pulses
    bracket the delay; echo inserts an x-axis pi pulse between half delays.
    The third-state survival is independent of this qubit Bloch channel.
    """
    rotate = np.array([[0.0, 0.0, 1.0], [0.0, 1.0, 0.0], [-1.0, 0.0, 0.0]])
    bloch = np.broadcast_to(
        np.array([[0.0, 0.0, 1.0], [0.0, 0.0, -1.0]]) @ rotate.T, (len(axis), 2, 3)
    ).copy()
    delay = axis / (2 if echo else 1)
    # Echo deliberately sees a nonzero detuning, which the middle pulse refocuses.
    phase = 2 * np.pi * parameters.get("detuning", 0.73) * delay
    for half in range(2 if echo else 1):
        x, y = bloch[:, :, 0].copy(), bloch[:, :, 1].copy()
        decay = np.exp(-delay / parameters["t2"])[:, None]
        bloch[:, :, 0] = decay * (
            np.cos(phase)[:, None] * x - np.sin(phase)[:, None] * y
        )
        bloch[:, :, 1] = decay * (
            np.sin(phase)[:, None] * x + np.cos(phase)[:, None] * y
        )
        bloch[:, :, 2] = 0.4 + (bloch[:, :, 2] - 0.4) * np.exp(-delay / t1)[:, None]
        if echo and half == 0:
            bloch[:, :, 1:] *= -1
    after = bloch @ rotate
    return np.column_stack([(1 - after[:, :, 2]) / 2, np.exp(-axis / parameters["tf"])])


def sampled_t2_sums(
    model: CycleModel,
    axis: np.ndarray,
    parameters: dict[str, float],
    contrast: float,
    count: int,
    seed: int,
    *,
    t1: float = 2.4,
) -> CycleSums:
    """Draw every D transition/read/noise event from a T1-containing channel.

    The channel propagates the Bloch density before drawing the final D state;
    it marginalizes unobserved delay jumps, not the measured cycle features.
    Both g/e rows interpolate from the same ground/excited output channel.
    """
    echo = model.oscillation_frequency is None
    physical = CycleModel(
        lambda x, p: bloch_curves(x, dict(p), t1=t1, echo=echo),
        model.parameter_names,
        model.bounds,
    )
    return sampled_sums(
        physical,
        t2_nuisance(contrast),
        axis,
        parameters,
        count,
        seed,
    )
