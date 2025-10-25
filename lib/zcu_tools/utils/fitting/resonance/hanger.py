from typing import Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

from .base import (
    calc_phase,
    fit_circle_params,
    fit_edelay,
    fit_resonant_params,
    normalize_signal,
    phase_func,
    remove_edelay,
)


def calc_background_signals(
    circle_params: Tuple[float, float, float], theta0: float
) -> complex:
    xc, yc, r0 = circle_params
    center = xc + 1j * yc
    return center - r0 * np.exp(1j * theta0)


def calc_phi(norm_yc: float, norm_r0: float) -> float:
    return -np.arcsin(norm_yc / norm_r0)


def calc_Qc(Ql: float, phi: float, r0: float) -> float:
    return Ql / (2 * r0 * np.exp(-1j * phi))


def calc_Qi(Ql: float, Qc: float) -> float:
    return 1 / (1 / Ql - np.real(1 / Qc))


class HangerModel:
    @classmethod
    def calc_signals(cls, fpts, freq, Ql, Qc, phi, a0, edelay) -> np.ndarray:
        return (
            a0
            * np.exp(-1j * 2 * np.pi * fpts * edelay)
            * (
                1
                - np.exp(1j * phi) * (Ql / abs(Qc)) / (1 + 2j * Ql * (fpts / freq - 1))
            )
        )

    @classmethod
    def fit(
        cls, fpts: np.ndarray, signals: np.ndarray, edelay: Optional[float] = None
    ) -> dict:
        """Dict[freq, kappa, Ql, Qc, Qi, phi, a0, edelay, circle_params]"""
        if edelay is None:
            edelay = fit_edelay(fpts, signals)

        rot_signals = remove_edelay(fpts, signals, edelay)
        circle_params = fit_circle_params(rot_signals.real, rot_signals.imag)
        freq, Ql, theta0 = fit_resonant_params(fpts, rot_signals, circle_params)
        a0 = calc_background_signals(circle_params, theta0)
        _, (norm_xc, norm_yc, norm_r0) = normalize_signal(
            rot_signals, circle_params, a0
        )
        phi = calc_phi(norm_yc, norm_r0)
        Qc = calc_Qc(Ql, phi, norm_r0)
        Qi = calc_Qi(Ql, Qc)

        return dict(
            freq=freq,
            kappa=freq / Ql,
            Ql=Ql,
            Qc=Qc,
            Qi=Qi,
            phi=phi,
            a0=a0,
            edelay=edelay,
            theta0=theta0,
            circle_params=circle_params,
        )

    @classmethod
    def visualize_fit(cls, fpts, signals, param_dict: dict) -> plt.Figure:
        freq = param_dict["freq"]
        kappa = param_dict["kappa"]
        theta0 = param_dict["theta0"]
        Ql = param_dict["Ql"]
        Qc = param_dict["Qc"]
        Qi = param_dict["Qi"]
        phi = param_dict["phi"]
        a0 = param_dict["a0"]
        edelay = param_dict["edelay"]
        circle_params = param_dict["circle_params"]

        fit_signals = cls.calc_signals(fpts, freq, Ql, Qc, phi, a0, edelay)
        rot_signals = remove_edelay(fpts, signals, edelay)
        norm_signals, norm_circle_params = normalize_signal(
            rot_signals, circle_params, a0
        )
        xc, yc, r0 = circle_params
        norm_xc, norm_yc, norm_r0 = norm_circle_params

        fig = plt.figure(figsize=(9, 8))
        spec = fig.add_gridspec(2, 2)
        ax1 = fig.add_subplot(spec[0, 0])
        ax2 = fig.add_subplot(spec[0, 1])
        ax3 = fig.add_subplot(spec[1, :])

        base_info = (
            r"$f_r = $" + f"{freq:.1f} MHz\n" + r"$\kappa = $" + f"{kappa:.1f} MHz"
        )
        Q_info = (
            r"$Q_l = $"
            + f"{Ql:.0f}\n"
            + r"$|Q_c| = $"
            + f"{np.abs(Qc):.0f}\n"
            + r"$Q_i = $"
            + f"{Qi:.0f}"
        )

        ax1.plot(norm_signals.real, norm_signals.imag)
        ax1.add_patch(plt.Circle((norm_xc, norm_yc), norm_r0, fill=False, color="red"))
        ax1.plot([norm_xc, 1], [norm_yc, 0], "kx--")
        ax1.axhline(0, color="k", linestyle="--")
        ax1.set_aspect("equal")
        ax1.grid()
        ax1.set_xlabel(r"$Re(S_{21})$")
        ax1.set_ylabel(r"$Im(S_{21})$")

        ax2.plot(fpts, calc_phase(rot_signals, xc, yc), ".", label="data")
        ax2.plot(fpts, phase_func(fpts, freq, Ql, theta0), label="fit")
        ax2.axvline(freq, color="k", linestyle="--")
        ax2.grid()
        ax2.legend(fontsize=12)
        ax2.set_xlabel("Frequency (MHz)")
        ax2.set_ylabel("Phase (rad)")

        ax3.plot(fpts, np.abs(signals), ".")
        ax3.plot(fpts, np.abs(fit_signals))
        ax3.axvline(freq, color="k", linestyle="--", label=base_info)
        ax3.text(
            0.85,
            0.30,
            Q_info,
            transform=ax3.transAxes,
            verticalalignment="top",
            fontsize=12,
            bbox=dict(facecolor="white", edgecolor="lightgray", alpha=0.5),
        )
        ax3.grid()
        ax3.legend(fontsize=12)
        ax3.set_xlabel("Frequency (MHz)")
        ax3.set_ylabel("Magnitude (a.u.)")

        return fig
