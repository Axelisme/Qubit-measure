import numpy as np


def xyz2sphere(x, y, z):
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r) * 180 / np.pi - 90  # polar angle in degrees
    phi = np.arctan2(y, x) * 180 / np.pi  # azimuthal angle in degrees

    return r, theta, phi


def normalize_signal(x):
    x = np.asarray(x)
    x_min, x_max = np.min(x), np.max(x)
    x_center = (x_max + x_min) / 2
    return 2 * (x - x_center) / (x_max - x_min)
