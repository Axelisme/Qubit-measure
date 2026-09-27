from __future__ import annotations

from typing import NotRequired

import numpy as np
from numpy.typing import NDArray
from typing_extensions import TypedDict  # extra_items (PEP 728) not in stdlib 3.13

from zcu_tools.analysis.spectrum import SpectrumData


class TransitionDict(TypedDict, extra_items=list[tuple[int, int]]):
    r_f: NotRequired[float]
    sample_f: NotRequired[float]


class PointsData(TypedDict):
    dev_values: NDArray[np.float64]
    fluxs: NDArray[np.float64]
    freqs: NDArray[np.float64]


class SpectrumResult(TypedDict):
    type: NotRequired[str]
    flux_half: float
    flux_int: float
    flux_period: float
    spectrum: SpectrumData
    points: PointsData
