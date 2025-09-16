import numpy as np


def xyz2sphere(x, y, z):
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    r = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arccos(z / r)  # polar angle
    phi = np.arctan2(y, x)  # azimuthal angle

    return r, theta, phi
