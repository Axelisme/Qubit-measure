"""Two-qubit Bloch, accumulated-phase and conditional-phase plot components."""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from textwrap import fill

import numpy as np
from matplotlib.artist import Artist
from matplotlib.axes import Axes
from matplotlib.patches import Circle
from matplotlib.ticker import MaxNLocator
from numpy.typing import NDArray
from zcu_tools.analysis.fitting.phase_slopes import PhaseSlopeFit
from zcu_tools.plotting.plots import Plots

from ._phase import (
    CountQuadratures,
    PhaseCrossings,
    complete_projections,
    conditional_phase_deg,
    phase_near,
    quadrature_phase_sem,
)


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
    reference unit circle. Missing pairs are omitted. The caller sets the title
    and legend and owns drawing/presentation.
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
        xlabel="Target Bloch X",
        ylabel="Target Bloch Y",
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
    ax: Axes, n: NDArray[np.int64], series: Sequence[AccumulatedPhaseSeries]
) -> None:
    """Draw measured angles ± shot SEM and their fitted lines on ax.

    N is the common measured count axis. Missing angles have no point; missing
    SEM has no bar. Plot the supplied model curves without refitting. The
    caller sets title and legend and owns drawing/presentation.
    """
    for i, group in enumerate(series):
        color = f"C{i}"
        ax.errorbar(
            n,
            group.phase_deg,
            yerr=group.sem_deg,
            fmt=("o", "s")[i % 2],
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
    """Label an accumulated phase axis (degrees versus integer CZ counts)."""
    ax.set(xlabel="CZ gates, n (count)", ylabel="Accumulated phase (deg)")
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))


def format_conditional_phase_axis(ax: Axes, frequencies: NDArray[np.float64]) -> None:
    """Set MHz, a 0–360° phase window and the π reference on a sweep axis.

    Frequencies is the nonempty planned sweep. Format the axis without selecting
    or interpolating a crossing. Caller owns rendering.
    """
    ax.set(
        xlabel="Common CZ frequency (MHz)",
        ylabel="Conditional phase per CZ (deg)",
        ylim=(0, 360),
        yticks=np.arange(0, 361, 90),
    )
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


@dataclass(frozen=True)
class ConditionalPhaseScan:
    """The sweep's MHz and CZ-count axes, per-spectator fits, and π crossings."""

    frequencies: NDArray[np.float64]
    n: NDArray[np.int64]
    fitted: PhaseSlopeFit
    crossings: PhaseCrossings

    def reference_index(self) -> int:
        """Choose a sampled MHz nearest the selected root or a phase nearest π.

        Use the middle sample if no conditional phase fit is available.
        """
        reason = self.crossings.reason
        if reason is None:
            return int(
                np.argmin(np.abs(self.frequencies - self.crossings.frequencies_mhz[0]))
            )
        phase = phase_near(conditional_phase_deg(self.fitted), 180)
        if np.isfinite(phase).any():
            return int(np.nanargmin(np.abs(phase - 180)))
        return len(self.frequencies) // 2


def plot_conditional_phase(
    plots: Plots,
    scan: ConditionalPhaseScan,
    observed: CountQuadratures,
    *,
    title: str,
    reference_label: str = "",
) -> None:
    """Derive conditional phase left to right in one named Analyze figure.

    Observed supplies complete X/Y counts reduced to sigma for fit group 0, in
    actual spectator g/e order. Group 0 anchors the offset and observed points;
    the fit supplies the experiment's shared slopes. Label the reference group
    for a joint fit. No selection, fitting or hardware access occurs.
    """
    fig, _ = plots.subplots(
        num="conditional_phase",
        ncols=3,
        gridspec_kw={"width_ratios": (1, 1.14, 1.25)},
        figsize=(12, 9),
        layout="constrained",
    )
    bloch_ax, phase_ax, scan_ax = fig.axes
    idx = scan.reference_index()
    sample = float(scan.frequencies[idx])
    detail = f"Measured sample: {sample:.3f} MHz"
    fig.suptitle(f"{title}\n{detail} for panels 1–2")
    bloch_ax.set_title(
        "1 · Bloch vectors"
        + (f"\n{reference_label} measurements" if reference_label else "")
    )
    phase_ax.set_title(
        "2 · Accumulated phase"
        + ("\nJoint slopes; gg offsets" if reference_label else "")
    )
    xy = observed.values[..., idx]
    labels = ("Spectator g", "Spectator e")
    plot_bloch_vectors(
        bloch_ax, scan.n, [BlochSeries(label, *xy[b]) for b, label in enumerate(labels)]
    )
    _plot_reference_phase(phase_ax, scan, observed, idx)
    _mark_conditional_crossings(scan_ax, scan, sample)
    for ax in fig.axes:
        ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.13))


def _plot_reference_phase(
    ax: Axes, scan: ConditionalPhaseScan, observed: CountQuadratures, idx: int
) -> None:
    """Show measured angles lifted to the existing fitted line's branch."""
    series = []
    grid = np.linspace(scan.n.min(), scan.n.max(), 201)
    sem = quadrature_phase_sem(observed)[..., idx]
    for b, label in enumerate(("Spectator g", "Spectator e")):
        x, y = observed.values[b, :, :, idx]
        angles = np.where(np.hypot(x, y) > 0, np.rad2deg(np.arctan2(y, x)), np.nan)
        measured_fit = scan.fitted.accumulated_phase_deg(
            scan.n, channel=b, coordinate=idx, group=0
        )
        fit_deg = scan.fitted.accumulated_phase_deg(
            grid, channel=b, coordinate=idx, group=0
        )
        fit_label = None
        if measured_fit is not None:
            angles = phase_near(angles, measured_fit)
            fit_label = (
                f"{label} fit: {scan.fitted.slopes_deg[b, idx]:+.1f}"
                f" ± {scan.fitted.errors_deg[b, idx]:.1f} °/CZ"
            )
        series.append(
            AccumulatedPhaseSeries(label, angles, sem[b], grid, fit_deg, fit_label)
        )
    plot_accumulated_phase(ax, scan.n, series)


def _mark_conditional_crossings(
    ax: Axes, scan: ConditionalPhaseScan, sample: float
) -> None:
    """Draw the same conditional phase as Live and mark original crossings at π."""
    format_conditional_phase_axis(ax, scan.frequencies)
    ax.plot(
        *conditional_phase_plot_data(
            scan.frequencies, conditional_phase_deg(scan.fitted)
        ),
        ".-",
        color="C0",
        label="Spectator e minus g",
    )
    ax.axvline(sample, color="0.55", linestyle=":", label="Sample in 1–2")
    roots = scan.crossings.frequencies_mhz
    if roots:
        ax.scatter(
            roots,
            np.full(len(roots), 180),
            color="C3",
            marker="x",
            zorder=4,
            label="Interpolated π crossing",
        )
    status = (
        f"π crossing = {roots[0]:.3f} MHz\nWriteback proposal available"
        if scan.crossings.reason is None
        else f"No writeback proposal\n{fill(scan.crossings.reason, width=35)}"
    )
    ax.set_title(f"3 · Conditional phase per CZ\n{status}")


def start_conditional_phase_live(
    plots: Plots,
    frequencies: NDArray[np.float64],
    title: str,
    *,
    expected_shots: int,
    fit: Callable[[NDArray[np.float64]], PhaseSlopeFit],
) -> Callable[[NDArray[np.float64]], None]:
    """Show the phase scan and return a callback for cumulative raw counts.

    Callback input axes are n × spectator × X/Y × frequency × count classes.
    Fit receives only newly complete frequencies in spectator × X/Y × n ×
    frequency × count classes order. A frequency is complete when every planned
    projection has the specified number of shots. Fit on the acquisition thread;
    only mutate artists on the plot owner thread. Call until all updates drain.
    """
    fig, ax = plots.subplots(num="conditional_phase", layout="constrained")
    fig.suptitle(title)
    format_conditional_phase_axis(ax, frequencies)
    (curve,) = ax.plot([], [], ".-", color="C0", label="Spectator e minus g")
    status = ax.set_title("Waiting for shots")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncols=2)
    fig.show()
    processed = np.zeros(len(frequencies), dtype=bool)
    phase_deg = np.full_like(frequencies, np.nan)

    @plots.on_plot_thread
    def render(payload: tuple[NDArray[np.float64], str]) -> tuple[Artist, ...]:
        angles_deg, progress = payload
        curve.set_data(*conditional_phase_plot_data(frequencies, angles_deg))
        status.set_text(progress)
        return curve, status

    def update(sums: NDArray[np.float64]) -> None:
        nominal_shots = sums.sum(axis=tuple(range(4, sums.ndim)))
        complete = complete_projections(nominal_shots, expected_shots)
        added = complete.all(axis=(0, 1, 2)) & ~processed
        if added.any():
            phase_deg[added] = conditional_phase_deg(
                fit(np.moveaxis(sums[:, :, :, added], 0, 2))
            )
            processed[added] = True
        progress = (
            f"{np.count_nonzero(processed)}/{len(frequencies)} frequencies acquired\n"
            f"{np.count_nonzero(complete)}/{complete.size} projections at {expected_shots} shots"
        )
        render((phase_deg, progress))

    return update
