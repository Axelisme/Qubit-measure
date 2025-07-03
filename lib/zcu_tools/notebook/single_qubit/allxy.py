from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

from zcu_tools.notebook.single_qubit.process import rotate2real

from .general import figsize


def _calc_predicted_state(sequence: List[Tuple[str, str]]) -> np.ndarray:
    """Calculate the expected ⟨σ_z⟩ after each gate pair.

    The calculation is performed by treating the qubit as a Bloch vector that
    starts at (0, 0, 1) (ground state). Each gate is mapped to a classical
    rotation on the Bloch sphere and applied in the order they appear in the
    *sequence* list.

    Parameters
    ----------
    sequence : List[Tuple[str, str]]
        Sequence of gate pairs, e.g. ("X90", "Y180").

    Returns
    -------
    np.ndarray
        Array of predicted ⟨σ_z⟩ values, taking values in {+1, 0, –1} for ideal
        gates.
    """

    def _rotation_x(theta: float) -> np.ndarray:
        """Rotation matrix around x-axis for Bloch vector representation."""

        return np.array(
            [
                [1, 0, 0],
                [0, np.cos(theta), -np.sin(theta)],
                [0, np.sin(theta), np.cos(theta)],
            ]
        )

    def _rotation_y(theta: float) -> np.ndarray:
        """Rotation matrix around y-axis for Bloch vector representation."""

        return np.array(
            [
                [np.cos(theta), 0, np.sin(theta)],
                [0, 1, 0],
                [-np.sin(theta), 0, np.cos(theta)],
            ]
        )

    # Map each gate to its corresponding rotation matrix
    gate2rot = {
        "I": np.identity(3),
        "X180": _rotation_x(np.pi),
        "Y180": _rotation_y(np.pi),
        "X90": _rotation_x(np.pi / 2),
        "Y90": _rotation_y(np.pi / 2),
    }

    predicted = []
    for gate1, gate2 in sequence:
        # Start from ground state (0, 0, 1)
        vec = np.array([0.0, 0.0, 1.0])

        # Apply the two rotations in order
        vec = gate2rot[gate1] @ vec
        vec = gate2rot[gate2] @ vec

        # z-component corresponds to expectation value of σ_z
        predicted.append(vec[2])

    return np.array(predicted)


def analyze_allxy(sequence: List[Tuple[str, str]], signals: np.ndarray) -> None:
    # Rotate IQ data so that the contrast lies on the real axis and take only
    # the real part for further analysis.
    signals = rotate2real(signals).real

    # ------------------------------------------------------------------
    # 1. Calculate the ideal expectation values (+1, 0, –1)
    # ------------------------------------------------------------------
    predicted_state = _calc_predicted_state(sequence)

    # ------------------------------------------------------------------
    # 2. Use non-linear least squares to fit average signal and contrast
    #    such that:  s_pred = avg + 0.5 * contrast * predicted_state
    # ------------------------------------------------------------------

    def calc_predicted(params: np.ndarray) -> np.ndarray:
        avg, contrast = params
        return avg + 0.5 * contrast * predicted_state

    # Initial guess: use simple statistics of the measured signals
    avg_guess = float(np.mean(signals))
    contrast_guess = float(np.ptp(signals))

    res = least_squares(
        lambda p: calc_predicted(p) - signals, x0=[avg_guess, contrast_guess]
    )

    avg_signal, contrast = res.x  # fitted parameters

    # ------------------------------------------------------------------
    # 3. Plotting
    # ------------------------------------------------------------------

    _, ax = plt.subplots(figsize=figsize)
    ax.plot(signals, marker="o", linestyle="None", label="Measured Signals")
    ax.plot(
        calc_predicted([avg_signal, contrast]),
        marker="x",
        linestyle="-",
        color="red",
        label="Predicted Signals",
    )

    ax.set_xlabel("Gate")
    ax.set_xticks(np.arange(len(sequence)))
    ax.set_xticklabels([f"{g1}-{g2}" for g1, g2 in sequence], rotation=45)

    ax.set_ylabel("Signal")
    ax.legend()

    plt.tight_layout()
    plt.show()
