from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray


@dataclass(slots=True)
class OvernightEnv:
    soc: object
    soccfg: object
    iters: NDArray[np.int64]
