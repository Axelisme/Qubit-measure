import matplotlib.pyplot as plt
import numpy as np

from . import fitting as ft
from .experiment import figsize


def T1_analyze(x: float, y: float):
    y = np.abs(y)
    pOpt, pCov = ft.fitexp(x, y)
    t1 = pOpt[2]
    sim = ft.expfunc(x, *pOpt)

    plt.figure(figsize=figsize)
    plt.plot(x, y, label="meas", ls="-", marker="o", markersize=3)
    plt.plot(x, sim, label="fit")
    plt.title(f"T1 = {t1:.2f}$\mu s$", fontsize=15)
    plt.xlabel("$t\ (\mu s)$", fontsize=15)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return t1


def T2fringe_analyze(x: float, y: float):
    y = np.abs(y)
    pOpt, pCov = ft.fitdecaysin(x, y)
    decay, detune = pOpt[4], pOpt[2]
    sim = ft.decaysin(x, *pOpt)
    error = np.sqrt(np.diag(pCov))

    plt.figure(figsize=figsize)
    plt.plot(x, y, label="meas", ls="-", marker="o", markersize=3)
    plt.plot(x, sim, label="fit")
    plt.title(
        f"T2 fringe = {decay:.2f}$\mu s, detune = {detune:.2f}MHz \pm {(error[2])*1e3:.2f}kHz$",
        fontsize=15,
    )
    plt.xlabel("$t\ (\mu s)$", fontsize=15)
    plt.ylabel("Population", fontsize=15)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return decay, detune


def T2decay_analyze(x: float, y: float):
    y = np.abs(y)
    pOpt, pCov = ft.fitexp(x, y)
    decay = pOpt[2]

    plt.figure(figsize=figsize)
    plt.plot(x, y, label="meas", ls="-", marker="o", markersize=3)
    plt.plot(x, ft.expfunc(x, *pOpt), label="fit")
    plt.title(f"T2 decay = {decay:.2f}$\mu s$", fontsize=15)
    plt.xlabel("$t\ (\mu s)$", fontsize=15)
    plt.legend()
    plt.tight_layout()
    plt.show()

    return decay
