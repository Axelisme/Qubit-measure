from typing import Union

import numpy as np
from numpy.typing import NDArray

from .base import (
    calc_phase,
    fit_circle_params,
    fit_edelay,
    fit_resonant_params,
    get_rough_edelay,
    normalize_signal,
    phase_func,
    remove_edelay,
)
from .hanger import HangerModel
from .transmission import TransmissionModel


def get_proper_model(
    fpts: NDArray[np.float64], signals: NDArray[np.complex128]
) -> Union[HangerModel, TransmissionModel]:
    background = 0.5 * (np.abs(signals[0]) + np.abs(signals[-1]))
    magnitudes = np.abs(signals)

    if magnitudes.max() - background < 3 * (background - magnitudes.min()):
        return HangerModel()
    else:
        return TransmissionModel()
