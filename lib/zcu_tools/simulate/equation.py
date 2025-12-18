from math import sqrt

import scipy.constants as sc

PHI0 = sc.hbar / (2 * sc.e)


def C2EC(C: float) -> float:
    """C: fF -> EC: GHz"""
    return sc.e**2 / (2 * sc.h * C * 1e-15) * 1e-9


def invC2EC(invC: float) -> float:
    """invC: 1/fF -> EC: GHz"""
    return sc.e**2 / (2 * sc.h) * invC * 1e15 * 1e-9


def EC2C(EC: float) -> float:
    """EC: GHz -> C: fF"""
    return sc.e**2 / (2 * sc.h * EC * 1e9) * 1e15


def L2EL(L: float) -> float:
    """L: nH -> EL: GHz"""
    return PHI0**2 / (sc.h * L)


def EL2L(EL: float) -> float:
    """EL: GHz -> L: nH"""
    return PHI0**2 / (sc.h * EL)


def LC2freq(L: float, C: float) -> float:
    """L: nH, C: fF -> f: GHz"""
    return 1 / (2 * sc.pi) * sqrt(1 / (L * C * 1e-6))


def Lfreq2C(L: float, freq: float) -> float:
    """L: nH, f: GHz -> C: fF"""
    return 1 / (L * (2 * sc.pi * freq) ** 2) * 1e6


def Cfreq2L(C: float, freq: float) -> float:
    """C: fF, f: GHz -> L: nH"""
    return 1 / (C * 1e-6 * (2 * sc.pi * freq) ** 2)


def n_coeff(EC: float, EL: float) -> float:
    """EC: GHz, EL: GHz -> n_coeff: 1"""
    return (EL / (32 * EC)) ** 0.25


def phi_coeff(EC: float, EL: float) -> float:
    """EC: GHz, EL: GHz -> phi_coeff: 1"""
    return (2 * EC / EL) ** 0.25
