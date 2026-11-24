"""Two-qubit Bloch, accumulated-phase and conditional-phase plot components."""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from matplotlib.axes import Axes
from matplotlib.patches import Circle
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray
from zcu_tools.analysis.fitting.phase_slopes import PhaseSlopeFit

from zcu_lab.v2.support.paired_ramsey import phase_near


def wrapped_phase_plot_data(
    x: NDArray[np.int64] | NDArray[np.float64],
    phase_deg: Sequence[float] | NDArray[np.float64],
    *aligned: NDArray[np.float64],
) -> tuple[NDArray[np.float64], ...]:
    """Return x, wrapped degrees and any aligned vectors (e.g. fit SE) for plotting.

    Supply matching 1D vectors in scan order and use NaN for a missing angle.
    Add NaN separators for adjacent plotted angles at least 180° apart.
    Matplotlib breaks the line there while keeping each original marker and error bar.
    """
    phases = np.asarray(phase_deg, dtype=np.float64)
    breaks = np.flatnonzero(np.abs(np.diff(phases)) >= 180) + 1
    return tuple(
        np.insert(values, breaks, np.nan)
        for values in (np.asarray(x, dtype=np.float64), phases, *aligned)
    )


@dataclass(frozen=True)
class BlochSeries:
    """Legend label and matching 1D, dimensionless measured X/Y per CZ count."""

    label: str
    x: NDArray[np.float64]
    y: NDArray[np.float64]


@dataclass(frozen=True)
class AccumulatedPhaseSeries:
    """Legend label, measured degree angles per CZ count and optional fit curve.

    sem_deg shares the measured axis; NaN omits the error bar. fit_n and
    fit_phase_deg are a matching dense count grid and fitted degree angles, or
    both None for an unavailable fit. fit_label overrides the model's legend.
    """

    label: str
    phase_deg: NDArray[np.float64]
    sem_deg: NDArray[np.float64]
    fit_n: NDArray[np.float64] | None
    fit_phase_deg: NDArray[np.float64] | None
    fit_label: str | None = None


def plot_bloch_vectors(
    ax: Axes, n: NDArray[np.int64], series: Sequence[BlochSeries]
) -> None:
    """Draw each finite X/Y pair as a vector, with a CZ count label, on ax.

    N is the count axis shared by the series. Use equal X/Y scale and show the
    reference unit circle. Missing pairs are omitted. The caller sets labels,
    title and legend and owns presentation.
    """
    ax.add_patch(Circle((0, 0), 1, fill=False, edgecolor="0.55", linestyle=":"))
    ax.axhline(0, linewidth=0.7, color="0.7")
    ax.axvline(0, linewidth=0.7, color="0.7")
    radii = [np.hypot(group.x, group.y) for group in series]
    radius = max(1.0, *(float(np.max(r[np.isfinite(r)], initial=0)) for r in radii))
    limit = 1.23 * radius
    for i, group in enumerate(series):
        mask = np.isfinite(group.x) & np.isfinite(group.y)
        x, y = group.x[mask], group.y[mask]
        color = f"C{i}"
        ax.quiver(
            np.zeros_like(x),
            np.zeros_like(y),
            x,
            y,
            angles="xy",
            scale_units="xy",
            scale=1,
            color=color,
            alpha=0.55,
        )
        ax.plot(x, y, ("o", "s")[i % 2], color=color, label=group.label)
        _label_vectors(ax, n[mask], np.stack((x, y)), limit, i)
    ax.set_anchor("N")
    ax.set(
        aspect="equal",
        xlim=(-limit, limit),
        ylim=(-limit, limit),
    )


def _label_vectors(
    ax: Axes, n: NDArray[np.int64], xy: NDArray[np.float64], limit: float, index: int
) -> None:
    """Stagger count labels in left/right columns and point back to their tips."""
    order = np.argsort(xy[1], kind="stable")
    if not order.size:
        return
    levels = xy[1, order].copy()
    gap = min(0.14, 1.8 / len(order)) * limit
    for j in range(1, len(levels)):
        levels[j] = max(levels[j], levels[j - 1] + gap)
    levels -= max(0.0, levels[-1] - 0.91 * limit)
    side = 1 if index % 2 == 0 else -1
    color = f"C{index}"
    for j, y in zip(order, levels, strict=True):
        ax.annotate(
            f"n={n[j]}",
            tuple(xy[:, j]),
            xytext=(side * 0.93 * limit, y),
            ha="right" if side == 1 else "left",
            va="center",
            fontsize=10,
            color=color,
            arrowprops={"arrowstyle": "-", "color": color, "alpha": 0.55},
        )


def plot_accumulated_phase(
    ax: Axes,
    n: NDArray[np.int64],
    series: Sequence[AccumulatedPhaseSeries],
    *,
    connect_unfitted: bool = False,
) -> None:
    """Draw measured angles ± shot SEM and their fitted lines on ax.

    N is the common measured count axis. Missing angles have no point; missing
    SEM has no bar. Plot the supplied model curves without refitting. With
    connect_unfitted, connect adjacent measured angles only for series without
    a fit, breaking at missing angles and jumps of at least 180°. By default,
    all measurements are unconnected. The caller sets labels, title and legend.
    """
    for i, group in enumerate(series):
        color = f"C{i}"
        marker = ("o", "s")[i % 2]
        if connect_unfitted and (group.fit_n is None or group.fit_phase_deg is None):
            counts, phase, sem = wrapped_phase_plot_data(
                n, group.phase_deg, group.sem_deg
            )
            marker += "-"
        else:
            counts, phase, sem = n, group.phase_deg, group.sem_deg
        ax.errorbar(
            counts,
            phase,
            yerr=sem,
            fmt=marker,
            color=color,
            label=f"{group.label} data",
        )
        if group.fit_n is not None and group.fit_phase_deg is not None:
            ax.plot(
                group.fit_n,
                group.fit_phase_deg,
                color=color,
                label=group.fit_label or f"{group.label} fit",
            )
    format_accumulated_phase_axis(ax)


def format_accumulated_phase_axis(ax: Axes) -> None:
    """Use integer ticks on the CZ count axis."""
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))


def format_conditional_phase_axis(ax: Axes, frequencies: NDArray[np.float64]) -> None:
    """Set a 0–360° phase window and draw the π reference on a sweep axis.

    Frequencies is the nonempty planned MHz sweep. The caller sets axis labels.
    """
    ax.set(ylim=(0, 360), yticks=np.arange(0, 361, 90))
    low, high = float(frequencies[0]), float(frequencies[-1])
    padding = 0.03 * (high - low) if high > low else 0.05
    ax.set_xlim(low - padding, high + padding)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=4))
    ax.ticklabel_format(useOffset=False, style="plain", axis="x")
    ax.axhline(180, color="0.45", linestyle="--", label="π = 180°")


def conditional_phase_plot_data(
    frequencies: NDArray[np.float64], phase_deg: NDArray[np.float64]
) -> tuple[NDArray[np.float64], ...]:
    """Return aligned MHz/degree plot arrays in [0,360), breaking jumps at the cut.

    Supply matching 1D arrays in scan order. A π crossing stays continuous;
    missing values and half-turn jumps remain breaks. No fit is performed.
    """
    return wrapped_phase_plot_data(frequencies, phase_near(phase_deg, 180))


def reference_phase_series(
    n: NDArray[np.int64],
    fitted: PhaseSlopeFit,
    xy: NDArray[np.float64],
    sem: NDArray[np.float64],
    idx: int,
    *,
    herald_index: int = 0,
) -> list[AccumulatedPhaseSeries]:
    """Prepare measured angles and fitted lines for the selected frequency.

    Supply the CZ counts and the selected sample's X/Y and phase SEM in
    spectator g/e order for the chosen fit group. The sample index refers to
    the fitted frequency axis; herald_index selects the group's offset (group
    0 by default). Lift measured angles onto that group's fitted branch when
    available. Do not refit.
    """
    series = []
    grid = np.linspace(n.min(), n.max(), 201)
    for b, label in enumerate(("Spectator g", "Spectator e")):
        x, y = xy[b]
        angles = np.where(np.hypot(x, y) > 0, np.rad2deg(np.arctan2(y, x)), np.nan)
        measured_fit = fitted.accumulated_phase_deg(
            n, channel=b, coordinate=idx, group=herald_index
        )
        fit_deg = fitted.accumulated_phase_deg(
            grid, channel=b, coordinate=idx, group=herald_index
        )
        fit_label = None
        if measured_fit is not None:
            angles = phase_near(angles, measured_fit)
            fit_label = f"{label} fit: {fitted.slopes_deg[b, idx]:+.1f} °/CZ"
        series.append(
            AccumulatedPhaseSeries(label, angles, sem[b], grid, fit_deg, fit_label)
        )
    return series


def plot_conditional_phase_scan(
    ax: Axes,
    frequencies: NDArray[np.float64],
    phase_deg: NDArray[np.float64],
    sample: float,
    roots: tuple[float, ...],
) -> None:
    """Draw fitted phase, the reference sample and crossing markers on ax.

    Supply the MHz scan and aligned per-CZ phase in degrees, the reference sample
    in MHz, and the interpolated π crossing frequencies in MHz (possibly empty).
    The caller sets labels, title and legend and owns presentation.
    """
    format_conditional_phase_axis(ax, frequencies)
    ax.plot(
        *conditional_phase_plot_data(frequencies, phase_deg),
        ".-",
        color="C0",
        label="Spectator e minus g",
    )
    ax.axvline(sample, color="0.55", linestyle=":", label="Sample in 1–2")
    if roots:
        ax.scatter(
            roots,
            np.full(len(roots), 180),
            color="C3",
            marker="x",
            zorder=4,
            label="Interpolated π crossing",
        )
