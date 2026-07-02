from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from zcu_tools.experiment.v2.runner import ScheduleStep


@dataclass(slots=True)
class OvernightDeps:
    soc: object
    soccfg: object


@dataclass(slots=True)
class OvernightEnv:
    soc: object
    soccfg: object
    iters: NDArray[np.int64]


def iteration_index(step: ScheduleStep[Any, Any, OvernightEnv]) -> int:
    if not step.path:
        raise ValueError("Overnight update step must include an iteration path")
    index = step.path[0]
    if not isinstance(index, int):
        raise TypeError(f"Expected integer iteration path, got {type(index)}")
    return index
