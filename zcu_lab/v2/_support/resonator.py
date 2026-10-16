"""GUI-free resonance fit/plot mechanics shared by single-tone and mux scans."""

from collections.abc import Mapping
from dataclasses import dataclass
from numbers import Integral, Real
from typing import Literal

import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray
from zcu_tools.analysis.fitting import HangerModel, TransmissionModel, get_proper_model
from zcu_tools.analysis.fitting.resonance.hanger import HangerParams
from zcu_tools.analysis.fitting.resonance.transmission import TransmissionParams


@dataclass(frozen=True)
class ResonanceOptions:
    """Shared model inputs, with MHz frequencies and inverse-MHz delays.

    model_type selects hanger (hm), transmission (t), or magnitude-based auto.
    edelay fixes delay when supplied; edelay_branch_seed initializes its fit.
    The two background switches enable independent amplitude slope and phase
    curvature. edelay_max_search_radius bounds unseeded search, or None uses the
    fitter default. Invalid model/numeric inputs raise ValueError during fitting.
    """

    model_type: Literal["hm", "t", "auto"] = "auto"
    edelay: float | None = None
    fit_bg_amp_slope: bool = False
    fit_bg_phase_curvature: bool = False
    edelay_branch_seed: float | None = None
    edelay_max_search_radius: float | None = None


@dataclass(frozen=True)
class ResonanceFit:
    """Successful numeric fit; freq/fwhm are MHz, theta0 rad, edelay MHz^-1.

    params retains the selected model's complete named fit output, including
    complex scale/coupling and circle geometry, for single-tone consumers.
    """

    freq: float
    fwhm: float
    theta0: float
    edelay: float
    params: HangerParams | TransmissionParams


def fit_resonance(
    freqs: NDArray[np.float64],
    signals: NDArray[np.complex128],
    options: ResonanceOptions,
    figure: Figure,
) -> ResonanceFit:
    """Fit equal 1-D MHz/IQ arrays, dropping both endpoints, then draw figure.

    The remaining grid must be finite, strictly monotonic, and contain at least
    four observations. Model validation and numerical failures propagate; callers
    own isolation and diagnostic plots. This function does not select a delay
    prior, decide persistence, or interpret experimental coordinates.
    """
    freqs, signals = freqs[1:-1], signals[1:-1]
    if options.model_type == "hm":
        model = HangerModel()
    elif options.model_type == "t":
        model = TransmissionModel()
    elif options.model_type == "auto":
        model = get_proper_model(freqs, signals)
    else:
        raise ValueError(f"Invalid model type: {options.model_type}")
    # Keep each model's typed parameter/renderer pair together.
    if isinstance(model, HangerModel):
        params = model.fit(
            freqs,
            signals,
            options.edelay,
            fit_bg_amp_slope=options.fit_bg_amp_slope,
            fit_bg_phase_curvature=options.fit_bg_phase_curvature,
            edelay_branch_seed=options.edelay_branch_seed,
            edelay_max_search_radius=options.edelay_max_search_radius,
        )
        figure.clear()
        model.visualize_fit(
            freqs,
            signals,
            params,
            figure=figure,
            fit_bg_amp_slope=options.fit_bg_amp_slope,
            fit_bg_phase_curvature=options.fit_bg_phase_curvature,
        )
        freq, fwhm, theta0, edelay = (
            params["freq"],
            params["fwhm"],
            params["theta0"],
            params["edelay"],
        )
        values: HangerParams | TransmissionParams = params
    else:
        transmission = model.fit(
            freqs,
            signals,
            options.edelay,
            fit_bg_amp_slope=options.fit_bg_amp_slope,
            fit_bg_phase_curvature=options.fit_bg_phase_curvature,
            edelay_branch_seed=options.edelay_branch_seed,
            edelay_max_search_radius=options.edelay_max_search_radius,
        )
        figure.clear()
        model.visualize_fit(
            freqs,
            signals,
            transmission,
            figure=figure,
            fit_bg_amp_slope=options.fit_bg_amp_slope,
            fit_bg_phase_curvature=options.fit_bg_phase_curvature,
        )
        freq, fwhm, theta0, edelay = (
            transmission["freq"],
            transmission["fwhm"],
            transmission["theta0"],
            transmission["edelay"],
        )
        values = transmission
    return ResonanceFit(
        freq=float(freq),
        fwhm=float(fwhm),
        theta0=float(theta0),
        edelay=float(edelay),
        params=values,
    )


def finite_real(value: object) -> float | None:
    """Return a finite real scalar as float; reject bool, arrays and nonfinite."""
    if isinstance(value, bool) or not isinstance(value, Real):
        return None
    converted = float(value)
    return converted if np.isfinite(converted) else None


def route_matched_delay(raw: object, route: tuple[int, int]) -> float | None:
    """Read a finite MHz^-1 prior only for exact (generator, ADC) route matches.

    raw must contain exactly edelay, res_ch, ro_ch; channels must be non-bool
    integers. Missing, malformed or mismatched calibration returns None. Source
    lookup failures are not inputs to this parser and must propagate at caller.
    """
    if not isinstance(raw, Mapping) or set(raw) != {"edelay", "res_ch", "ro_ch"}:
        return None
    gen, adc = raw["res_ch"], raw["ro_ch"]
    if any(isinstance(ch, bool) or not isinstance(ch, Integral) for ch in (gen, adc)):
        return None
    if (gen, adc) != route:
        return None
    return finite_real(raw["edelay"])
