from __future__ import annotations

from typing import TypedDict

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from numpy.typing import NDArray

from .base import (
    align_phase_to_data,
    calc_phase,
    fit_circle_params,
    fit_edelay,
    fit_resonant_params,
    normalize_signal,
    phase_func,
    remove_background,
    remove_edelay,
    run_complex_refinement,
    validate_complex_fit_inputs,
)


def calc_peak_signals(
    circle_params: tuple[float, float, float], theta0: float
) -> complex:
    xc, yc, r0 = circle_params
    center = xc + 1j * yc
    return center + r0 * np.exp(1j * theta0)


class TransmissionParams(TypedDict):
    freq: float
    fwhm: float
    Ql: float
    a0: complex
    edelay: float
    theta0: float
    bg_amp_slope: float
    circle_params: tuple[float, float, float]


class TransmissionModel:
    @classmethod
    def calc_signals(
        cls,
        freqs: NDArray[np.float64],
        freq: float,
        Ql: float,
        a0: complex,
        edelay: float,
        bg_amp_slope: float = 0.0,
        **kwargs,
    ) -> NDArray[np.complex128]:
        dx = Ql * (freqs / freq - 1)
        center = a0 / 2
        vector = (a0 / 2) * (1 - 2j * dx) / (1 + 2j * dx)
        ideal = center + vector
        background = np.exp(bg_amp_slope * (freqs - freq))
        return background * ideal * np.exp(-1j * 2 * np.pi * freqs * edelay)

    @classmethod
    def _fit_sequential(
        cls,
        freqs: NDArray[np.float64],
        signals: NDArray[np.complex128],
        edelay: float,
    ) -> TransmissionParams:
        rot_signals = remove_edelay(freqs, signals, edelay)
        circle_params = fit_circle_params(rot_signals.real, rot_signals.imag)
        freq, Ql, theta0 = fit_resonant_params(
            freqs,
            rot_signals,
            circle_params,
            fit_theta0=False,
        )
        a0 = calc_peak_signals(circle_params, theta0)

        return TransmissionParams(
            freq=freq,
            fwhm=freq / Ql,
            Ql=Ql,
            a0=a0,
            edelay=edelay,
            theta0=theta0,
            bg_amp_slope=0.0,
            circle_params=circle_params,
        )

    @classmethod
    def _refine_complex(
        cls,
        freqs: NDArray[np.float64],
        signals: NDArray[np.complex128],
        initializer: TransmissionParams,
        *,
        refine_edelay: bool,
    ) -> tuple[float, float, complex, float, float] | None:
        span = float(np.ptp(freqs))
        center = 0.5 * float(np.min(freqs) + np.max(freqs))
        amp_scale = max(float(np.sqrt(np.mean(np.abs(signals) ** 2))), 1e-12)
        init_edelay = initializer["edelay"]

        def encode_frequency(value: float) -> float:
            return (value - center) / span

        def decode(
            values: NDArray[np.float64],
        ) -> tuple[float, float, complex, float, float]:
            freq = center + span * values[0]
            Ql = float(np.exp(values[1]))
            a0 = amp_scale * complex(values[2], values[3])
            bg_amp_slope = float(values[4] / span)
            edelay = (
                init_edelay + float(values[5] / span) if refine_edelay else init_edelay
            )
            return freq, Ql, a0, edelay, bg_amp_slope

        freq_lower = encode_frequency(float(np.min(freqs)))
        freq_upper = encode_frequency(float(np.max(freqs)))
        freq_init = float(
            np.clip(
                encode_frequency(initializer["freq"]),
                freq_lower + 1e-9,
                freq_upper - 1e-9,
            )
        )
        initial = [
            freq_init,
            np.log(max(initializer["Ql"], 1.0)),
            initializer["a0"].real / amp_scale,
            initializer["a0"].imag / amp_scale,
            0.0,
        ]
        lower = [freq_lower, np.log(1.0), -1e3, -1e3, -5.0]
        upper = [freq_upper, np.log(1e9), 1e3, 1e3, 5.0]
        if refine_edelay:
            initial.append(0.0)
            lower.append(-1.0)
            upper.append(1.0)

        def residual(values: NDArray[np.float64]) -> NDArray[np.float64]:
            freq, Ql, a0, edelay, bg_amp_slope = decode(values)
            fitted = cls.calc_signals(freqs, freq, Ql, a0, edelay, bg_amp_slope)
            delta = (fitted - signals) / amp_scale
            return np.concatenate((delta.real, delta.imag))

        result = run_complex_refinement(
            residual,
            initial,
            (lower, upper),
            model_name="transmission",
        )
        if result is None:
            return None
        return decode(result)

    @classmethod
    def fit(
        cls,
        freqs: NDArray[np.float64],
        signals: NDArray[np.complex128],
        edelay: float | None = None,
        fit_bg_amp_slope: bool = False,
        edelay_search_radius: float | None = None,
    ) -> TransmissionParams:
        """Fit a transmission response, resolving delay aliases within a radius."""
        validate_complex_fit_inputs(freqs, signals)
        refine_edelay = edelay is None
        if edelay is None:
            edelay = fit_edelay(
                freqs,
                signals,
                search_radius=edelay_search_radius,
            )

        initializer = cls._fit_sequential(freqs, signals, edelay)
        if not fit_bg_amp_slope:
            return initializer

        refined = cls._refine_complex(
            freqs, signals, initializer, refine_edelay=refine_edelay
        )
        if refined is None:
            return initializer
        freq, Ql, a0, edelay, bg_amp_slope = refined

        corrected = remove_background(
            freqs,
            signals,
            freq=freq,
            edelay=edelay,
            bg_amp_slope=bg_amp_slope,
        )
        circle_params = fit_circle_params(corrected.real, corrected.imag)
        data_phases = calc_phase(corrected, circle_params[0], circle_params[1])
        theta0 = align_phase_to_data(float(np.angle(a0)), freqs, data_phases, freq)

        return TransmissionParams(
            freq=freq,
            fwhm=freq / Ql,
            Ql=Ql,
            a0=a0,
            edelay=edelay,
            theta0=theta0,
            bg_amp_slope=bg_amp_slope,
            circle_params=circle_params,
        )

    @classmethod
    def visualize_fit(
        cls,
        freqs: NDArray[np.float64],
        signals: NDArray[np.complex128],
        param_dict: TransmissionParams,
    ) -> Figure:
        freq = param_dict["freq"]
        fwhm = param_dict["fwhm"]
        theta0 = param_dict["theta0"]
        Ql = param_dict["Ql"]
        a0 = param_dict["a0"]
        edelay = param_dict["edelay"]
        bg_amp_slope = param_dict["bg_amp_slope"]
        circle_params = param_dict["circle_params"]

        xc, yc, r0 = circle_params
        fit_signals = cls.calc_signals(freqs, freq, Ql, a0, edelay, bg_amp_slope)
        corrected = remove_background(
            freqs,
            signals,
            freq=freq,
            edelay=edelay,
            bg_amp_slope=bg_amp_slope,
        )

        norm_signals, norm_circle_params = normalize_signal(
            corrected, circle_params, a0
        )
        norm_xc, norm_yc, norm_r0 = norm_circle_params

        fig = plt.figure(figsize=(9, 8))
        spec = fig.add_gridspec(2, 2)
        ax1 = fig.add_subplot(spec[0, 0])
        ax2 = fig.add_subplot(spec[0, 1])
        ax3 = fig.add_subplot(spec[1, :])

        base_info = "freq = " + f"{freq:.1f} MHz\n" + "FWHM = " + f"{fwhm:.1f} MHz"
        Q_info = (
            r"$Q_l = $" + f"{Ql:.0f}\n" + r"$g = $" + f"{bg_amp_slope:.4g} MHz$^{{-1}}$"
        )

        ax1.plot(norm_signals.real, norm_signals.imag, label="corrected data")
        ax1.add_patch(Circle((norm_xc, norm_yc), norm_r0, fill=False, color="red"))
        ax1.plot([norm_xc, 1], [norm_yc, 0], "kx--")
        ax1.set_aspect("equal")
        ax1.grid()
        ax1.set_xlabel(r"$Re(S_{21})$")
        ax1.set_ylabel(r"$Im(S_{21})$")

        ax2.plot(
            freqs,
            calc_phase(corrected, xc, yc),
            ".",
            label="corrected data",
        )
        ax2.plot(
            freqs,
            phase_func(freqs, freq, Ql, theta0),
            label="ideal phase fit",
        )
        ax2.axvline(freq, color="k", linestyle="--")
        ax2.grid()
        ax2.legend(fontsize=12)
        ax2.set_xlabel("Frequency (MHz)")
        ax2.set_ylabel("Phase (rad)")

        ax3.plot(freqs, np.abs(signals), ".", label="raw data")
        ax3.plot(freqs, np.abs(fit_signals), label="total fit")
        ax3.plot(
            freqs,
            np.abs(a0) * np.exp(bg_amp_slope * (freqs - freq)),
            "--",
            label="multiplicative background envelope",
        )
        ax3.axvline(freq, color="k", linestyle="--", label=base_info)
        ax3.text(
            0.98,
            0.90,
            Q_info,
            transform=ax3.transAxes,
            verticalalignment="top",
            horizontalalignment="right",
            fontsize=12,
            bbox=dict(facecolor="white", edgecolor="lightgray", alpha=0.5),
        )
        ax3.grid()
        ax3.legend(loc="lower left", fontsize=12)
        ax3.set_xlabel("Frequency (MHz)")
        ax3.set_ylabel("Magnitude (a.u.)")

        return fig
