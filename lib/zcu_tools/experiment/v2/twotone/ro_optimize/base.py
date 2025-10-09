import numpy as np


def signal2snr(signals: np.ndarray) -> np.ndarray:
    return np.abs(signals[0] - signals[1])
