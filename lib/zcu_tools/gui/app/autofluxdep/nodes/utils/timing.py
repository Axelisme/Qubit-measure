"""Timing primitives shared by autofluxdep experiment nodes."""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def times_to_cycles_and_axis(
    soccfg: Any, times: NDArray[np.float64]
) -> tuple[list[int], NDArray[np.float64]]:
    """Quantize a requested delay axis to hardware cycles and return actual times."""
    cycles = [int(soccfg.us2cycles(float(time))) for time in times]
    if any(right <= left for left, right in zip(cycles, cycles[1:], strict=False)):
        raise ValueError(
            "delay sweep collapsed after cycle quantization; "
            "reduce expts or widen the delay sweep"
        )
    actual_times = np.asarray(
        [soccfg.cycles2us(int(cycle)) for cycle in cycles], dtype=np.float64
    )
    return cycles, actual_times
