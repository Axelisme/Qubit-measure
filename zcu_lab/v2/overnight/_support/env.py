from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from zcu_tools.experiment.context import RunContext


@dataclass(slots=True)
class OvernightEnv:
    context: RunContext
    iters: NDArray[np.int64]
