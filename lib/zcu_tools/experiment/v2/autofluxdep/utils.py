import warnings

import numpy as np


def check_gains(gains: float, name: str) -> np.ndarray:
    if np.any(gains > 1.0):
        warnings.warn(
            f"Some {name} gains are larger than 1.0, force clip to 1.0, which may cause distortion."
        )
        gains = np.clip(gains, 0.0, 1.0)
    return gains
