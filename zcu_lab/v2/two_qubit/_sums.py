"""Joint g/e/other count reductions for two-qubit summed acquisition.

All inputs are float64 arrays with arbitrary leading scan dimensions. Terminal
counts end in Qubit1 then Qubit2 (3, 3). Heralded counts add herald Qubit1
and herald Qubit2 before the terminal axes. Class order is g, e, other.
Unmeasured NaNs propagate. Zero totals return NaNs without divide warnings.
"""

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from zcu_tools.program.v2.modules import MuxReadout
from zcu_tools.program.v2.modules.readout import circle_classifier


def validate_qubit_tones(
    readout: MuxReadout.Cfg, qubit1_tone: int, qubit2_tone: int
) -> None:
    """Require distinct zero-based tone indices into readout and circle radii.

    Invalid indices or missing circle calibration raise ValueError. Calibration
    diagnostics name the position and request GE Post. Other tones are unchanged.
    """
    if qubit1_tone == qubit2_tone:
        raise ValueError("Qubit1 and Qubit2 must use distinct readout tone indices")
    for label, tone in (("Qubit1", qubit1_tone), ("Qubit2", qubit2_tone)):
        if not 0 <= tone < len(readout.tones):
            raise ValueError(
                f"{label} tone index {tone} is outside the {len(readout.tones)} readout tones"
            )
        _require_circle_radius(readout, label, tone)


def _require_circle_radius(readout: MuxReadout.Cfg, label: str, tone: int) -> None:
    try:
        classifier = circle_classifier(readout, ro_ch=readout.tones[tone].ro_ch)
    except ValueError as exc:
        raise ValueError(
            f"{label}: circle classifier required; run GE Post. {exc}"
        ) from exc
    if classifier.radius is None:
        raise ValueError(f"{label}: circle radius required; run GE Post")


def joint_probabilities(counts: NDArray[np.float64]) -> NDArray[np.float64]:
    """Normalize terminal (..., 3, 3) counts over both qubits, including other."""
    total = counts.sum(axis=(-2, -1), keepdims=True)
    return np.divide(counts, total, out=np.full_like(counts, np.nan), where=total != 0)


def qubit_marginal_probabilities(
    counts: NDArray[np.float64], target: Literal["qubit1", "qubit2"]
) -> NDArray[np.float64]:
    """Return (..., 3) P_g/P_e/P_other for the named qubit position.

    counts ends in terminal Qubit1 then Qubit2. Other contributes to the
    denominator. Empty or unmeasured points yield NaN; invalid targets raise
    ValueError.
    """
    if target not in ("qubit1", "qubit2"):
        raise ValueError("target must be qubit1 or qubit2")
    return joint_probabilities(counts).sum(axis=-1 if target == "qubit1" else -2)


def qubit_sigma(
    counts: NDArray[np.float64], target: Literal["qubit1", "qubit2"]
) -> NDArray[np.float64]:
    """Return target's (N_e - N_g)/N_total from terminal Qubit1 × Qubit2 counts.

    Include other in N_total. Empty or unmeasured points yield NaN; invalid
    targets raise ValueError.
    """
    return _sigma(qubit_marginal_probabilities(counts, target))


def _sigma(marginal: NDArray[np.float64]) -> NDArray[np.float64]:
    return marginal[..., 1] - marginal[..., 0]


def terminal_counts(
    sums: NDArray[np.float64],
    herald: tuple[Literal["g", "e", "other"], Literal["g", "e", "other"]],
) -> NDArray[np.float64]:
    """Select terminal (...,3,3) counts for the named herald (Qubit1,Qubit2) classes.

    sums ends in herald Qubit1/Qubit2 then terminal Qubit1/Qubit2, each g/e/other. Leading
    scan dimensions remain unchanged. This returns a view without normalization.
    Invalid class names raise ValueError. Unmeasured NaNs propagate.
    """
    qubit1_class, qubit2_class = (("g", "e", "other").index(state) for state in herald)
    return sums[..., qubit1_class, qubit2_class, :, :]


def herald_probabilities(sums: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return (..., 3, 3) herald probabilities from 81 counts.

    Sum over terminal Qubit1/Qubit2 first. The returned (g,g) cell is pass fraction.
    """
    return joint_probabilities(sums.sum(axis=(-2, -1)))
