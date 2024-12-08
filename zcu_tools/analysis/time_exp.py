import matplotlib.pyplot as plt
import numpy as np

from . import fitting as ft
from .general import figsize
from .tools import convert2max_contrast


def T1_analyze(x: float, y: float, return_err=False, plot=True, max_contrast=False):
    if max_contrast:
        y, _ = convert2max_contrast(y.real, y.imag)
    else:
        y = np.abs(y)

    pOpt, pCov = ft.fitexp(x, y)
    t1 = pOpt[2]
    sim = ft.expfunc(x, *pOpt)
    err = np.sqrt(np.diag(pCov))

    if plot:
        t1_str = f"{t1:.2f}us +/- {err[2]:.2f}us"

        plt.figure(figsize=figsize)
        plt.plot(x, y, label="meas", ls="-", marker="o", markersize=3)
        plt.plot(x, sim, label="fit")
        plt.title(f"T1 = {t1_str}", fontsize=15)
        plt.xlabel("$t\ (\mu s)$", fontsize=15)
        plt.legend()
        plt.tight_layout()
        plt.show()

    if return_err:
        return t1, err[2]
    return t1


def T2fringe_analyze(
    x: float, y: float, return_err=False, plot=True, max_contrast=False
):
    if max_contrast:
        y, _ = convert2max_contrast(y.real, y.imag)
    else:
        y = np.abs(y)

    pOpt, pCov = ft.fitdecaysin(x, y)
    t2f, detune = pOpt[4], pOpt[2]
    sim = ft.decaysin(x, *pOpt)
    err = np.sqrt(np.diag(pCov))

    if plot:
        t2f_str = f"{t2f:.2f}us +/- {err[4]:.2f}us"
        detune_str = f"{detune:.2f}MHz \pm {err[2]*1e3:.2f}kHz"

        plt.figure(figsize=figsize)
        plt.plot(x, y, label="meas", ls="-", marker="o", markersize=3)
        plt.plot(x, sim, label="fit")
        plt.title(f"T2 fringe = {t2f_str}, detune = {detune_str}", fontsize=15)
        plt.xlabel("$t\ (\mu s)$", fontsize=15)
        plt.ylabel("Population", fontsize=15)
        plt.legend()
        plt.tight_layout()
        plt.show()

    if return_err:
        return t2f, detune, err[4], err[2]
    return t2f, detune


def T2decay_analyze(
    x: float, y: float, return_err=False, plot=True, max_contrast=False
):
    if max_contrast:
        y, _ = convert2max_contrast(y.real, y.imag)
    else:
        y = np.abs(y)

    pOpt, pCov = ft.fitexp(x, y)
    t2e = pOpt[2]
    err = np.sqrt(np.diag(pCov))

    if plot:
        t2e_str = f"{t2e:.2f}us +/- {err[2]:.2f}us"

        plt.figure(figsize=figsize)
        plt.plot(x, y, label="meas", ls="-", marker="o", markersize=3)
        plt.plot(x, ft.expfunc(x, *pOpt), label="fit")
        plt.title(f"T2 decay = {t2e_str}", fontsize=15)
        plt.xlabel("$t\ (\mu s)$", fontsize=15)
        plt.legend()
        plt.tight_layout()
        plt.show()

    if return_err:
        return t2e, err[2]
    return t2e
