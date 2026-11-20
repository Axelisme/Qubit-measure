"""Count-array records for two-line duration scans."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class RabiScanResult:
    """Coordinates, requested durations in us and unchanged float64 class counts.

    coordinates are relative phase degrees or common MHz frequency, as named by
    the leaf AXES_SPEC. sums is (Nc,Nd,3,3) after any herald selection.
    Class order is g/e/other, with terminal F1 before F2. Unmeasured points
    remain NaN; completed rounds survive
    cancellation. There is no shot axis or IQ. The RunContext stop signal carries
    only a stop request and reason. Execution errors propagate from run as
    exceptions; a normal return after a stop request may contain partial counts.
    """

    coordinates: NDArray[np.float64]
    durations: NDArray[np.float64]
    sums: NDArray[np.float64]
