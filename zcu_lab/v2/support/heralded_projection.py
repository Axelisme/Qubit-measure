"""Relabel one herald group's terminal counts for phase projections."""

from typing import Literal

import numpy as np
from numpy.typing import NDArray


def align_terminal_counts(
    counts: NDArray[np.float64],
    target: Literal["qubit1", "qubit2"],
    herald: tuple[Literal["g", "e"], Literal["g", "e"]],
    *,
    spectator_axis: int | None = None,
) -> NDArray[np.float64]:
    """Align selected (..., 3, 3) terminal counts without normalizing them.

    Terminal axes and herald labels are in Qubit1/Qubit2 order, with terminal
    g/e/other. For target herald e, exchange terminal target g/e (negate its
    projection with the same denominator). If spectator_axis names a nominal
    I/Xπ scan axis, reverse its rows for spectator herald e, so the returned
    rows represent actual g/e. Omit that axis when there is no I/Xπ scan.

    Return the same shape and dtype, possibly sharing memory. Preserve NaNs
    in their relabeled positions and never write to counts. Mask any incomplete
    points in nominal order before calling. An invalid spectator axis raises
    ValueError.
    """
    target_index = 0 if target == "qubit1" else 1
    if spectator_axis is not None:
        axis = spectator_axis if spectator_axis >= 0 else counts.ndim + spectator_axis
        if not 0 <= axis < counts.ndim - 2 or counts.shape[axis] != 2:
            raise ValueError("Spectator axis must be a length-2 I/Xπ scan axis")
        if herald[1 - target_index] == "e":
            counts = np.flip(counts, axis=axis)
    if herald[target_index] == "e":
        counts = np.take(counts, [1, 0, 2], axis=target_index - 2)
    return counts
