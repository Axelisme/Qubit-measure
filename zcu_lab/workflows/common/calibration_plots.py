"""Run-local calibration traces and multi-panel workflow figures."""

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from matplotlib.axes import Axes
from matplotlib.colors import Normalize
from matplotlib.figure import Figure
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray
from zcu_tools.experiment.v2.utils import estimate_snr
from zcu_tools.plotting.plots import Plots


@dataclass(frozen=True)
class CalibrationTrace:
    """One flux row: native sweep coordinates and projected signal, not persisted.

    point_index identifies a sweep point even when flux setpoints repeat.
    flux uses the plan's device unit. x is detune in MHz for spectroscopy or
    pulse duration in us for Rabi. observed/fitted share x's coordinates;
    fitted is None when analysis failed (observed then shows raw I).
    accepted is the pi duration in us for Rabi or fitted detune in MHz for
    spectroscopy, otherwise None.
    """

    point_index: int
    flux: float
    x: NDArray[np.float64]
    observed: NDArray[np.float64]
    fitted: NDArray[np.float64] | None = None
    accepted: float | None = None


def retain_trace(rows: list[CalibrationTrace], trace: CalibrationTrace) -> None:
    """Append a point's row, replacing its last attempt when widening repeats it."""
    if rows and rows[-1].point_index == trace.point_index:
        rows[-1] = trace
    else:
        rows.append(trace)


def _edges(centres: NDArray[np.float64]) -> NDArray[np.float64]:
    if len(centres) == 1:
        half_width = max(abs(float(centres[0])) * 0.01, 1e-6)
        return np.array([centres[0] - half_width, centres[0] + half_width])
    midpoints = (centres[:-1] + centres[1:]) / 2
    return np.concatenate(
        ([2 * centres[0] - midpoints[0]], midpoints, [2 * centres[-1] - midpoints[-1]])
    )


def _heatmap(axis: Axes, rows: list[CalibrationTrace]) -> None:
    # Each row keeps its own actual grid: widening and adaptive Rabi ranges
    # need not share endpoints. Sort only for display, never for acquisition.
    ordered = sorted(rows, key=lambda row: row.flux)
    flux_edges = _edges(np.array([row.flux for row in ordered]))
    signal = np.concatenate([row.observed for row in ordered])
    finite = signal[np.isfinite(signal)]
    norm = (
        Normalize(float(finite.min()), float(finite.max()))
        if finite.size
        else Normalize()
    )
    for index, row in enumerate(ordered):
        axis.pcolormesh(
            _edges(row.x),
            flux_edges[index : index + 2],
            row.observed[None, :],
            norm=norm,
            shading="flat",
        )


def draw_frequency(
    plots: Plots,
    rows: list[CalibrationTrace],
    accepted: list[tuple[float, float]],
    *,
    flux_unit: str,
    directory: Path,
) -> None:
    """Draw f01 history, detune heatmap and latest three flux slices with fit/SNR.

    Rows contain at most one attempt per point; the caller replaces widened
    attempts. Save the combined figure as qubit-freq.png in this point directory.
    """
    name = "calibration/qubit_freq"
    if name not in plots:
        plots.subplots(name)
    current = rows[-1]

    def draw(figure: Figure) -> None:
        figure.clear()
        figure.set_layout_engine("constrained")
        grid = figure.add_gridspec(2, 2)
        trend = figure.add_subplot(grid[0, :])
        heatmap = figure.add_subplot(grid[1, 0])
        trace = figure.add_subplot(grid[1, 1])
        if accepted:
            fluxes, frequencies = zip(*accepted, strict=True)
            trend.scatter(fluxes, frequencies)
        trend.set(xlabel=f"Flux ({flux_unit})", ylabel="Accepted f01 (MHz)")
        trend.xaxis.set_major_locator(MaxNLocator(nbins=4))
        _heatmap(heatmap, rows)
        heatmap.set(xlabel="Detune (MHz)", ylabel=f"Flux ({flux_unit})")
        for row in rows[-3:]:
            trace.plot(row.x, row.observed, ".-", label=f"{row.flux:.6g} {flux_unit}")
        if current.fitted is not None:
            trace.plot(current.x, current.fitted, label="fit")
        if current.accepted is not None:
            trace.axvline(current.accepted, color="red", linestyle="--")
        trace.set(
            xlabel="Detune (MHz)",
            ylabel="Signal (a.u.)",
            title=f"Recent flux slices, SNR={estimate_snr(current.observed):.1f}",
        )
        trace.legend()

    plots.draw(name, draw)
    plots.save(name, directory / "qubit-freq.png", figsize=(12.8, 9.6))


def draw_rabi(
    plots: Plots,
    rows: list[CalibrationTrace],
    *,
    flux_unit: str,
    directory: Path,
) -> None:
    """Draw native length rows, accepted pi durations in red and current fit.

    Save the combined figure as len-rabi.png in this point directory.
    """
    name = "calibration/len_rabi"
    if name not in plots:
        plots.subplots(name)
    current = rows[-1]

    def draw(figure: Figure) -> None:
        figure.clear()
        figure.set_layout_engine("constrained")
        heatmap, trace = figure.subplots(1, 2)
        _heatmap(heatmap, rows)
        accepted = sorted(
            (row for row in rows if row.accepted is not None), key=lambda row: row.flux
        )
        heatmap.plot(
            [row.accepted for row in accepted],
            [row.flux for row in accepted],
            "r.-",
            label="Accepted pi",
        )
        heatmap.set(xlabel="Pulse length (us)", ylabel=f"Flux ({flux_unit})")
        heatmap.legend()
        trace.plot(current.x, current.observed, ".", label="observed")
        if current.fitted is not None:
            trace.plot(current.x, current.fitted, label="fit")
        if current.accepted is not None:
            trace.axvline(current.accepted, color="red", linestyle="--", label="pi")
        trace.set(
            xlabel="Pulse length (us)",
            ylabel="Signal (a.u.)",
            title=f"Current flux\n{current.flux:.6g} {flux_unit}",
        )
        trace.legend()

    plots.draw(name, draw)
    plots.save(name, directory / "len-rabi.png", figsize=(12.8, 4.8))


def draw_readout(
    plots: Plots,
    freqs: NDArray[np.float64],
    gains: NDArray[np.float64],
    snr: NDArray[np.float64],
    *,
    inputs_flux: tuple[float, str],
    directory: Path,
) -> None:
    """Draw the latest frequency-by-gain SNR scan and mark its finite maximum.

    snr has shape (len(freqs), len(gains)); axes use native sample centres.
    Save readout.png. Calling only on a completed cadence scan retains the
    previous figure on cadence skips. inputs_flux is (device setpoint, plan
    unit) for the title. Nonfinite scans have no best marker.
    """
    name = "calibration/readout"
    if name not in plots:
        plots.subplots(name)

    def draw(figure: Figure) -> None:
        figure.clear()
        figure.set_layout_engine("constrained")
        axis = figure.subplots()
        freq_edges, gain_edges = _edges(freqs), _edges(gains)
        image = axis.imshow(
            snr.T,
            origin="lower",
            aspect="auto",
            extent=(freq_edges[0], freq_edges[-1], gain_edges[0], gain_edges[-1]),
        )
        figure.colorbar(image, ax=axis, label="SNR")
        if np.any(np.isfinite(snr)):
            row, column = np.unravel_index(
                np.argmax(np.where(np.isfinite(snr), snr, -np.inf)), snr.shape
            )
            axis.plot(freqs[row], gains[column], "rx", markersize=10, label="Best SNR")
            axis.legend()
        flux, unit = inputs_flux
        axis.set(
            xlabel="Readout frequency (MHz)",
            ylabel="Readout gain (a.u.)",
            title=f"Readout SNR\nFlux = {flux:.6g} {unit}",
        )

    plots.draw(name, draw)
    plots.save(name, directory / "readout.png")
