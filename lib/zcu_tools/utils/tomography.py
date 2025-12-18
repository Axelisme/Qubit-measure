import numpy as np
from numpy.typing import NDArray


def xyz2sphere(x: NDArray[np.float64], y: NDArray[np.float64], z: NDArray[np.float64]):
    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r) * 180 / np.pi - 90  # polar angle in degrees
    phi = np.arctan2(y, x) * 180 / np.pi  # azimuthal angle in degrees

    return r, theta, phi


def normalize_signal(x: NDArray[np.float64]) -> NDArray[np.float64]:
    x_min, x_max = np.min(x), np.max(x)
    x_center = (x_max + x_min) / 2
    return 2 * (x - x_center) / (x_max - x_min)
