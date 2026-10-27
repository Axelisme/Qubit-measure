"""Width-sensitive dip selection on magnitude spectra, without plotting or I/O."""

from dataclasses import dataclass
from math import isfinite

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import gaussian_filter1d
from scipy.signal import find_peaks, peak_widths


@dataclass(frozen=True)
class DipFinderOptions:
    """Controls for magnitude dip selection.

    smoothing_sigma is Gaussian sigma in samples; zero disables smoothing.
    prominence_factor multiplies the adjacent-difference robust noise estimate.
    max_fwhm is the largest accepted half-prominence width in MHz.
    min_separation is the minimum selected center spacing in MHz.
    max_dips limits the score-ranked selection; None selects all separated dips.
    Numeric controls must be finite; sigma and prominence_factor are nonnegative,
    widths and spacing positive, and max_dips a positive integer or None.
    Invalid controls raise ValueError on construction; booleans are rejected.
    """

    smoothing_sigma: float = 0.0
    prominence_factor: float = 5.0
    max_fwhm: float = 25.0
    min_separation: float = 1.0
    max_dips: int | None = None

    def __post_init__(self) -> None:
        _validate_controls(
            (
                ("smoothing_sigma", self.smoothing_sigma, True),
                ("prominence_factor", self.prominence_factor, True),
                ("max_fwhm", self.max_fwhm, False),
                ("min_separation", self.min_separation, False),
            ),
            self.max_dips,
        )


def _validate_controls(
    controls: tuple[tuple[str, object, bool], ...], max_dips: object
) -> None:
    # Editors and JSON callers can supply runtime types outside the annotations.
    for name, value, allow_zero in controls:
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(value)
            or value < 0
            or (not allow_zero and value == 0)
        ):
            raise ValueError(
                f"{name} must be finite and {'>= 0' if allow_zero else '> 0'}"
            )
    if max_dips is not None and (
        isinstance(max_dips, bool) or not isinstance(max_dips, int) or max_dips <= 0
    ):
        raise ValueError("max_dips must be a positive integer or None")


@dataclass(frozen=True)
class SpectrumDip:
    """One interior minimum, measured on the background-subtracted spectrum.

    freq is the sampled center in MHz. depth is prominence in magnitude units.
    fwhm is the half-prominence width in MHz, at least the local mean sample step.
    score is depth / fwhm in magnitude units per MHz.
    """

    freq: float
    depth: float
    fwhm: float
    score: float


def find_spectrum_dips(
    freqs: NDArray[np.float64],
    magnitude: NDArray[np.float64],
    options: DipFinderOptions | None = None,
) -> tuple[SpectrumDip, ...]:
    """Select dips by descending depth / FWHM, with stable low-frequency ties.

    freqs and magnitude are matching finite 1-D arrays with >= 11 points.
    Frequencies are distinct absolute MHz, in any order; magnitude is nonnegative.
    Inputs are not modified. Invalid arrays raise ValueError. options=None uses
    DipFinderOptions defaults.

    Background joins the median frequency and magnitude in each outer 10% of
    sorted points (at least two per edge). Noise is adjacent-difference MAD /
    0.67449 / sqrt(2), before optional smoothing. Prominence must exceed the larger
    of prominence_factor * noise, 1e-6 of the magnitude range and machine epsilon.
    FWHM is interpolated at half prominence, floored at the mean of the two steps
    adjacent to the minimum. Width filtering precedes greedy spacing and count
    selection. Edge minima are excluded. No candidates returns an empty tuple.
    """
    if options is None:
        options = DipFinderOptions()
    freqs = np.asarray(freqs, dtype=np.float64)
    magnitude = np.asarray(magnitude, dtype=np.float64)
    if freqs.ndim != 1 or magnitude.shape != freqs.shape or freqs.size < 11:
        raise ValueError("Dip selection needs matching (P,) arrays with P >= 11")
    if not np.all(np.isfinite(freqs)) or not np.all(np.isfinite(magnitude)):
        raise ValueError("Dip selection needs a complete finite spectrum")
    if np.any(magnitude < 0):
        raise ValueError("magnitude must be nonnegative")
    order = np.argsort(freqs)
    freqs, magnitude = freqs[order], magnitude[order]
    if np.any(np.diff(freqs) <= 0):
        raise ValueError("Dip selection needs distinct frequencies")
    edge = max(2, freqs.size // 10)
    x0, x1 = np.median(freqs[:edge]), np.median(freqs[-edge:])
    y0, y1 = np.median(magnitude[:edge]), np.median(magnitude[-edge:])
    corrected = magnitude - (y0 + (y1 - y0) * (freqs - x0) / (x1 - x0))
    difference = np.diff(corrected)
    noise = float(np.median(np.abs(difference - np.median(difference)))) / (
        0.67449 * np.sqrt(2)
    )
    threshold = max(
        options.prominence_factor * noise,
        1e-6 * float(np.ptp(magnitude)),
        np.finfo(float).eps,
    )
    if options.smoothing_sigma > 0:
        corrected = gaussian_filter1d(corrected, sigma=options.smoothing_sigma)
    indices, properties = find_peaks(-corrected, prominence=threshold)
    _, _, left, right = peak_widths(
        -corrected,
        indices,
        rel_height=0.5,
        prominence_data=(
            properties["prominences"],
            properties["left_bases"],
            properties["right_bases"],
        ),
    )
    sample_indices = np.arange(freqs.size)
    widths = np.interp(right, sample_indices, freqs) - np.interp(
        left, sample_indices, freqs
    )
    steps = (freqs[indices + 1] - freqs[indices - 1]) / 2
    dips = tuple(
        SpectrumDip(
            float(freqs[index]), float(depth), float(width), float(depth / width)
        )
        for index, depth, width in zip(
            indices, properties["prominences"], np.maximum(widths, steps), strict=True
        )
        if width <= options.max_fwhm
    )
    selected: list[SpectrumDip] = []
    for dip in sorted(dips, key=lambda dip: -dip.score):
        if all(
            abs(dip.freq - other.freq) >= options.min_separation for other in selected
        ):
            selected.append(dip)
            if len(selected) == options.max_dips:
                break
    return tuple(selected)
