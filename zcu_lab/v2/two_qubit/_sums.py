"""Joint g/e/other count reductions for two-qubit summed acquisition.

All inputs are float64 arrays with arbitrary leading scan dimensions. Terminal
counts end in (F1, F2)=(3, 3); heralded counts end in (herald F1, herald F2,
terminal F1, terminal F2)=(3, 3, 3, 3). Class order is g, e, other.
Unmeasured NaNs propagate. Zero totals return NaNs without divide warnings.
"""

from typing import Literal

import numpy as np
from numpy.typing import NDArray
from zcu_tools.program.v2 import Classify
from zcu_tools.program.v2.modules import MuxReadout
from zcu_tools.program.v2.modules.readout import circle_classifier

from ._readout import JointRoles


def validate_classifiers(readout: MuxReadout.Cfg, roles: JointRoles) -> None:
    """Require circle + radius on F1/F2 only, without changing the full mux cfg.

    roles must describe this full readout. Invalid mapping, missing calibration,
    non-circle calibration or missing radius raises ValueError before Run.
    Calibration errors name the role and request GE Post.
    """
    roles.validate_readout(readout)
    for role, tone in (("F1", roles.f1_tone), ("F2", roles.f2_tone)):
        try:
            classifier = circle_classifier(readout, ro_ch=readout.tones[tone].ro_ch)
        except ValueError as exc:
            raise ValueError(f"{role}: circle classifier required; run GE Post. {exc}") from exc
        if classifier.radius is None:
            raise ValueError(f"{role}: circle radius required; run GE Post")


def readout_axes(roles: JointRoles, *, heralded: bool) -> list[Classify]:
    """Return terminal F1/F2 axes, preceded by herald F1/F2 when heralded.

    Tone positions come from the captured full mux mapping. Resolve this list
    with shot_sums only after building the program with these emission tags.
    """
    labels = ("herald", "terminal") if heralded else ("terminal",)
    return [
        Classify(label, tone)
        for label in labels
        for tone in (roles.f1_tone, roles.f2_tone)
    ]


def joint_probabilities(counts: NDArray[np.float64]) -> NDArray[np.float64]:
    """Normalize terminal (..., 3, 3) counts over both qubits, including other."""
    total = counts.sum(axis=(-2, -1), keepdims=True)
    return np.divide(counts, total, out=np.full_like(counts, np.nan), where=total != 0)


def marginal_probabilities(
    counts: NDArray[np.float64], target: Literal["f1", "f2"]
) -> NDArray[np.float64]:
    """Return (..., 3) P_g/P_e/P_other for target f1 or f2.

    counts ends in terminal F1/F2 axes. Invalid target raises ValueError.
    """
    if target not in ("f1", "f2"):
        raise ValueError("target must be f1 or f2")
    return joint_probabilities(counts).sum(axis=-1 if target == "f1" else -2)


def sigma(
    counts: NDArray[np.float64], target: Literal["f1", "f2"]
) -> NDArray[np.float64]:
    """Return target's (N_e - N_g)/N_total; other stays in the denominator."""
    marginal = marginal_probabilities(counts, target)
    return marginal[..., 1] - marginal[..., 0]


def terminal_counts(
    sums: NDArray[np.float64], *, heralded: bool
) -> NDArray[np.float64]:
    """Select herald (g,g) from 81 counts, or return non-heralded 9 counts.

    Leading scan dimensions are retained; this is a view, not a new acquisition
    or renormalization. Herald (g,g) means classified acceptance, not purity.
    """
    return sums[..., 0, 0, :, :] if heralded else sums


def herald_probabilities(sums: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return (..., 3, 3) herald probabilities from 81 counts.

    Sum over terminal F1/F2 first. The returned (g,g) cell is pass fraction.
    """
    return joint_probabilities(sums.sum(axis=(-2, -1)))
