"""Published spectrum evidence without interactive input or Qt presentation."""

import numpy as np
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.figure import Figure

from zcu_tools.analysis.fluxdep.models import PointsData, SpectrumData
from zcu_tools.analysis.fluxdep.processing import contrast_limits, onetone_dip_depth


def make_spectrum_figure(
    raw: SpectrumData,
    points: PointsData,
    *,
    alignment: tuple[float, float] | None,
    onetone: bool,
) -> Figure:
    """Return an Agg device/GHz heatmap with published lines and sampled points.

    raw contains native device positions, GHz frequencies and complex signals
    shaped (device, frequency). points uses the same units. alignment is the
    committed (half, integer) pair in native device units, or None when unaligned.
    onetone draws a OneTone spectrum like its sample picker: ``onetone_dip_depth``
    within ``contrast_limits``, a dark dip on white. Otherwise the background
    is the raw magnitude ``|signals|``.
    Empty axes produce an empty plot. Rendering does not modify inputs, create
    interactive sessions, register pyplot figures, or save files. Matplotlib
    errors propagate; the caller owns the returned figure.
    """
    figure = Figure(figsize=(8, 5))
    FigureCanvasAgg(figure)
    axes = figure.subplots()
    if raw["dev_values"].size and raw["freqs"].size:
        if onetone:
            background = onetone_dip_depth(raw["signals"])
            vmin, vmax = contrast_limits(background)
        else:
            background, vmin, vmax = np.abs(raw["signals"]), None, None
        axes.pcolormesh(
            raw["dev_values"],
            raw["freqs"],
            background.T,
            shading="auto",
            cmap="gray_r",
            vmin=vmin,
            vmax=vmax,
        )
    if alignment is not None:
        axes.axvline(alignment[0], color="tab:blue", label="half flux")
        axes.axvline(alignment[1], color="tab:orange", label="integer flux")
        axes.legend()
    if points["freqs"].size:
        axes.scatter(points["dev_values"], points["freqs"], color="red", s=12)
    axes.set_xlabel("Device value (native)")
    axes.set_ylabel("Frequency (GHz)")
    figure.tight_layout()
    return figure
