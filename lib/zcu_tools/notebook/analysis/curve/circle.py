import numpy as np


def circle_fit(signals):
    # ref: https://www.scribd.com/document/14819165/Regressions-coniques-quadriques-circulaire-spherique
    x, y = signals.real, signals.imag
    mx, my = np.mean(x), np.mean(y)
    u = x - mx
    v = y - my
    Suu = np.sum(u**2)
    Svv = np.sum(v**2)
    Suv = np.sum(u * v)
    Suuu = np.sum(u**3)
    Svvv = np.sum(v**3)
    Suvv = np.sum(u * v**2)
    Svuu = np.sum(v * u**2)
    N = len(x)
    A = np.array([[Suu, Suv], [Suv, Svv]])
    B = 0.5 * np.array([Suuu + Suvv, Svvv + Svuu])
    center = np.linalg.solve(A, B)
    radius = np.sqrt((Suu + Svv) / N + center[0] ** 2 + center[1] ** 2)

    center = center + np.array([mx, my])
    return center, radius
