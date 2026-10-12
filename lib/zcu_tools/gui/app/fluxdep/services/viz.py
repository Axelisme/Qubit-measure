"""viz — matplotlib rendering of the fit result (rewrite of the notebook's
Plotly ``FreqFluxDependVisualizer``).

A pure function that draws every layer onto a caller-supplied Figure, so the GUI
owns the Figure/canvas lifecycle (consistent with the interactive widgets) and
tests can render headless. The layers mirror the notebook visualiser:

  1. background heatmap   — each spectrum's normalised amplitude (``gray_r``,
                            matching the v1 interactive colouring: white→black,
                            high value = dark, simulation lines / points stand out)
  2. simulation lines     — the fitted (EJ,EC,EL) transition frequencies vs flux
  3. selected points      — the fit's input (flux, freq) cloud (blue)
  4. constant-freq lines  — r_f / half·sample_f / (sample_f − r_f) horizontals
  5. dev-value secondary axis — top x-axis showing the device value for each flux

render_fit_result computes the fitted simulation for both Show and RPC;
render_fit_figure also accepts precomputed energies for numerical callers.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from zcu_tools.analysis.fluxdep.models import TransitionDict, energy2transition
from zcu_tools.analysis.fluxdep.processing import cast2real_and_norm
from zcu_tools.analysis.fluxdep.search import DatabaseSearchResult
from zcu_tools.analysis.fluxdep.search_models import count_max_evals
from zcu_tools.gui.app.fluxdep.state import (
    FitState,
    SpectrumEntry,
    transitions_with_freqs,
)
from zcu_tools.simulate import flux2value
from zcu_tools.simulate.fluxonium import calculate_energy_vs_flux

logger = logging.getLogger(__name__)


@dataclass(frozen=True, slots=True)
class _Background:
    """Spectrum and nonempty axes/signals after dropping all-NaN rows/columns."""

    entry: SpectrumEntry
    fluxs: NDArray[np.float64]
    freqs: NDArray[np.float64]
    signals: NDArray[np.complex128]


def render_selection_result(
    figure: Figure, spectrums: dict[str, SpectrumEntry], *, fit: FitState | None = None
) -> None:
    """Draw published sampled points and per-spectrum masks, never live drafts.

    figure is caller-owned and cleared. Backgrounds use native normalization;
    excluded points are gray crosses, selected points blue dots in flux/GHz.
    fit optionally supplies stored EJ/EC/EL and transition choices for J6 model
    curves, including stale or params-only results. Filter points have no residual
    markings. Empty collections produce empty axes. No State, draft or file changes.
    """
    figure.clear()
    ax = figure.add_subplot(111)
    _plot_background(ax, spectrums)
    for entry in spectrums.values():
        points = entry.points
        kept = entry.filter_mask
        if kept is None:
            kept = np.ones(entry.point_count, dtype=np.bool_)
        ax.scatter(
            points["fluxs"][~kept],
            points["freqs"][~kept],
            color="gray",
            marker="x",
            zorder=2,
        )
        ax.scatter(
            points["fluxs"][kept], points["freqs"][kept], color="blue", s=12, zorder=3
        )
    if fit is not None and fit.params is not None:
        xlim, _ = derive_auto_limits(
            spectrums, np.empty(0), np.empty(0), fit.r_f, fit.sample_f
        )
        t_fluxs, energies = _fit_simulation(fit, xlim, fit.transitions)
        _plot_simulation_lines(
            ax,
            t_fluxs,
            energies,
            transitions_with_freqs(fit.transitions, fit.r_f, fit.sample_f),
        )
    ax.set_xlabel("Flux")
    ax.set_ylabel("Frequency (GHz)")
    figure.tight_layout()


def derive_auto_limits(
    spectrums: dict[str, SpectrumEntry],
    s_fluxs: NDArray[np.float64],
    s_freqs: NDArray[np.float64],
    r_f: float | None = None,
    sample_f: float | None = None,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Auto (x, y) limits matching the notebook visualiser's ``auto_derive_limits``.

    Accumulates the min/max over the same traces the plotly ``FreqFluxDependVisualizer``
    does: each spectrum's background (flux × freq) range, the selected points, and
    the constant-frequency reference lines (r_f / half·sample_f / mirror r_f).
    Returns ``((x_min, x_max), (y_min, y_max))``; falls back to a unit box if there
    is nothing to bound.
    """
    xs_lo, xs_hi = np.inf, -np.inf
    ys_lo, ys_hi = np.inf, -np.inf

    def _acc_x(lo: float, hi: float) -> None:
        nonlocal xs_lo, xs_hi
        xs_lo, xs_hi = min(xs_lo, lo), max(xs_hi, hi)

    def _acc_y(lo: float, hi: float) -> None:
        nonlocal ys_lo, ys_hi
        ys_lo, ys_hi = min(ys_lo, lo), max(ys_hi, hi)

    for entry in spectrums.values():
        usable = _usable_background(entry)
        if usable is None:
            continue
        # Signals are unused here; the shared filtering keeps limits consistent
        # with the rendered background.
        _acc_x(float(usable.fluxs.min()), float(usable.fluxs.max()))
        _acc_y(float(usable.freqs.min()), float(usable.freqs.max()))

    if s_fluxs.size:
        _acc_x(float(np.min(s_fluxs)), float(np.max(s_fluxs)))
    if s_freqs.size:
        _acc_y(float(np.min(s_freqs)), float(np.max(s_freqs)))

    for freq, _label in _constant_freqs(r_f, sample_f):
        _acc_y(freq, freq)

    if not np.isfinite(xs_lo):
        xs_lo, xs_hi = 0.0, 1.0
    if not np.isfinite(ys_lo):
        ys_lo, ys_hi = 0.0, 1.0
    return (xs_lo, xs_hi), (ys_lo, ys_hi)


def _constant_freqs(
    r_f: float | None, sample_f: float | None
) -> list[tuple[float, str]]:
    """The (freq, label) constant-frequency reference lines that are set (non-None)."""
    lines: list[tuple[float, str]] = []
    if r_f is not None:
        lines.append((r_f, "r_f"))
    if sample_f is not None:
        lines.append((0.5 * sample_f, "half sample_f"))
    if sample_f is not None and r_f is not None:
        lines.append((sample_f - r_f, "mirror r_f"))
    return lines


def _axis_resolution(axis: NDArray[np.float64]) -> float:
    """Points per unit along one axis, (points - 1) / range; 0 without a range."""
    span = float(np.max(axis) - np.min(axis))
    return (axis.size - 1) / span if span > 0 else 0.0


def _usable_background(entry: SpectrumEntry) -> _Background | None:
    """Return background without all-NaN rows/columns, or None if empty."""
    spect = entry.raw
    signals = spect["signals"]
    flux_mask = np.any(~np.isnan(signals), axis=1)
    freq_mask = np.any(~np.isnan(signals), axis=0)
    if not flux_mask.any() or not freq_mask.any():
        return None
    return _Background(
        entry,
        np.asarray(spect["fluxs"], dtype=np.float64)[flux_mask],
        np.asarray(spect["freqs"], dtype=np.float64)[freq_mask],
        signals[flux_mask, :][:, freq_mask],
    )


def _plot_background(ax, spectrums: dict[str, SpectrumEntry]) -> None:
    """Draw each spectrum's normalised amplitude as a gray_r heatmap.

    Spectra paint from low to high resolution, so a fine scan stays visible
    where it overlaps a coarse one. Resolution is the geometric mean of both
    axes' (points - 1) / range after dropping all-NaN rows; an axis with a
    single point or no range counts as 0. Equal resolutions keep insertion
    order. All backgrounds stay below the simulation lines and points.
    """
    backgrounds = []
    for entry in spectrums.values():
        usable = _usable_background(entry)
        if usable is not None:
            backgrounds.append(usable)
    backgrounds.sort(
        key=lambda background: np.sqrt(
            _axis_resolution(background.fluxs) * _axis_resolution(background.freqs)
        )
    )
    for background in backgrounds:
        real = cast2real_and_norm(
            background.signals, use_phase=background.entry.spec_type != "OneTone"
        )
        ax.pcolormesh(
            background.fluxs,
            background.freqs,
            real.T,
            cmap="gray_r",
            shading="auto",
            zorder=0,
        )


def _plot_simulation_lines(
    ax,
    t_fluxs: NDArray[np.float64],
    energies: NDArray[np.float64],
    transitions: TransitionDict,
    reference_transitions: TransitionDict | None = None,
) -> None:
    """Draw model frequencies with colors indexed by the full compiled fit.

    A displayed subset retains attribution colors. Repeated labels occupy
    distinct occurrence slots; explicitly added display lines receive new slots.
    """
    freqs, labels = energy2transition(energies, transitions)
    reference_labels = labels
    if reference_transitions is not None:
        _, reference_labels = energy2transition(energies, reference_transitions)
    palette: dict[tuple[str, int], int] = {}
    occurrences: dict[str, int] = {}
    for index, label in enumerate(reference_labels):
        occurrence = occurrences.get(label, 0)
        palette[(label, occurrence)] = index
        occurrences[label] = occurrence + 1
    occurrences.clear()
    for index, label in enumerate(labels):
        occurrence = occurrences.get(label, 0)
        slot = (label, occurrence)
        if slot not in palette:
            palette[slot] = len(palette)
        ax.plot(
            t_fluxs,
            freqs[:, index],
            label=label,
            color=f"C{palette[slot] % 10}",
            linewidth=1.0,
            zorder=1,
        )
        occurrences[label] = occurrence + 1


def _plot_constant_freqs(ax, r_f: float | None, sample_f: float | None) -> None:
    """Draw the r_f / half·sample_f / mirror-r_f horizontal reference lines."""
    for freq, name in _constant_freqs(r_f, sample_f):
        ax.axhline(freq, linestyle="--", linewidth=1.0, color="gray", zorder=1)
        ax.text(
            ax.get_xlim()[0],
            freq,
            f" {name}",
            va="bottom",
            ha="left",
            fontsize=8,
            color="gray",
        )


def _add_dev_value_axis(
    ax, flux_half: float, flux_period: float, num_ticks: int = 8
) -> None:
    """Add a top secondary x-axis showing the device value for each flux."""
    secax = ax.secondary_xaxis(
        "top",
        functions=(
            lambda flux: flux2value(flux, flux_half, flux_period),
            lambda value: (value - flux_half) / flux_period + 0.5,
        ),
    )
    secax.set_xlabel("Device value")
    secax.ticklabel_format(style="sci", scilimits=(-2, 3), useMathText=True)
    _ = num_ticks  # secondary_xaxis auto-ticks; kept for signature parity


@dataclass(frozen=True)
class FitDisplay:
    """Show overlay presentation, independent of the search inputs.

    xlim/ylim are normalized flux/GHz axis endpoints; None uses auto limits.
    show_const_freq toggles frequency references. plot_transitions overrides
    displayed line choices only; None uses the fit choices.
    """

    xlim: tuple[float, float] | None = None
    ylim: tuple[float, float] | None = None
    show_const_freq: bool = True
    plot_transitions: TransitionDict | None = None
    # Root-change descriptions from State; empty means the stored fit is fresh.
    stale_reasons: tuple[str, ...] = ()


def _fit_simulation(
    fit: FitState,
    xlim: tuple[float, float],
    transitions: TransitionDict,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Compute the shared J5/J6 model grid with all requested compiled levels."""
    if fit.params is None:
        raise ValueError("fit params are required for simulation")
    flux_lo, flux_hi = xlim
    if flux_hi <= flux_lo:
        flux_lo, flux_hi = 0.0, 1.0
    t_fluxs = np.linspace(flux_lo, flux_hi, 1000)
    _, energies = calculate_energy_vs_flux(
        fit.params,
        t_fluxs,
        cutoff=40,
        evals_count=max(
            15, count_max_evals(fit.transitions), count_max_evals(transitions)
        ),
    )
    return t_fluxs, energies


def render_fit_result(
    figure: Figure,
    fit: FitState,
    spectrums: dict[str, SpectrumEntry],
    s_fluxs: NDArray[np.float64],
    s_freqs: NDArray[np.float64],
    display: FitDisplay,
) -> None:
    """Simulate and draw the current fitted Show overlay into figure.

    fit must contain EJ/EC/EL params in GHz; missing params raise ValueError.
    spectra supply background/calibration; s_fluxs/s_freqs are paired normalized
    flux/GHz fit points. Limits and display options follow render_fit_figure.
    The simulation spans the displayed flux range. Does not mutate fit or spectra;
    only replaces figure contents. Recorded original coordinates and kernel
    attribution replace the provided live point cloud when quality exists.
    display.stale_reasons adds an explicit outdated title without discarding data.
    Native simulation/render failures propagate.
    """
    if fit.params is None:
        raise ValueError("fit params are required for rendering")
    if fit.result is not None:
        s_fluxs, s_freqs = fit.result_fluxs, fit.result_freqs
    auto_x, _ = derive_auto_limits(spectrums, s_fluxs, s_freqs, fit.r_f, fit.sample_f)

    display_transitions = (
        display.plot_transitions
        if display.plot_transitions is not None
        else fit.transitions
    )
    t_fluxs, energies = _fit_simulation(
        fit,
        display.xlim if display.xlim is not None else auto_x,
        display_transitions,
    )
    title = (
        f"EJ/EC/EL = ({fit.params[0]:.2f}, {fit.params[1]:.2f}, {fit.params[2]:.2f})"
    )
    if display.stale_reasons:
        title += "\n過期: " + "; ".join(display.stale_reasons)
    aligned = next((entry for entry in spectrums.values() if entry.aligned), None)
    render_fit_figure(
        figure,
        spectrums,
        t_fluxs,
        energies,
        fit.transitions,
        s_fluxs,
        s_freqs,
        r_f=fit.r_f,
        sample_f=fit.sample_f,
        flux_half=aligned.flux_half if aligned else None,
        flux_period=aligned.flux_period if aligned else None,
        title=title,
        attribution=fit.result,
        xlim=display.xlim,
        ylim=display.ylim,
        show_const_freq=display.show_const_freq,
        plot_transitions=display.plot_transitions,
    )


def render_fit_figure(
    figure: Figure,
    spectrums: dict[str, SpectrumEntry],
    t_fluxs: NDArray[np.float64],
    energies: NDArray[np.float64],
    transitions: TransitionDict,
    s_fluxs: NDArray[np.float64],
    s_freqs: NDArray[np.float64],
    r_f: float | None = None,
    sample_f: float | None = None,
    *,
    flux_half: float | None = None,
    flux_period: float | None = None,
    title: str = "",
    xlim: tuple[float, float] | None = None,
    ylim: tuple[float, float] | None = None,
    show_const_freq: bool = True,
    plot_transitions: TransitionDict | None = None,
    attribution: DatabaseSearchResult | None = None,
) -> None:
    """Render the full fit visualisation onto ``figure`` (cleared first).

    ``t_fluxs`` / ``energies`` are the simulated grid (caller computes the latter
    via ``calculate_energy_vs_flux``). ``s_fluxs`` / ``s_freqs`` are the selected
    fit points. ``flux_half`` / ``flux_period`` drive the dev-value secondary axis
    (omit to skip it).

    Show-tab controls:
    - ``xlim`` / ``ylim`` set the axis limits; ``None`` (default) derives them via
      :func:`derive_auto_limits` (matching the notebook's ``auto_derive_limits``).
    - ``show_const_freq`` toggles the r_f / sample_f reference lines.
    - ``plot_transitions`` overrides which transition categories are drawn (the
      display subset); ``None`` uses ``transitions`` (the fit set).
    - attribution is the same-order kernel result for s_fluxs/s_freqs. Its
      transition_indices supply compiled-slot colors; outlier_mask supplies
      hollow markers. None draws plain points, never synthetic residual marks.
    """
    figure.clear()
    ax = figure.add_subplot(1, 1, 1)

    # The transition model needs r_f / sample_f keys present for the categories
    # that use them; inject the provided ones so a 'mirror' / 'red side' display
    # subset never raises a missing-key error.
    sim_transitions = transitions_with_freqs(
        plot_transitions if plot_transitions is not None else transitions,
        r_f,
        sample_f,
    )

    _plot_background(ax, spectrums)
    _plot_simulation_lines(
        ax,
        t_fluxs,
        energies,
        sim_transitions,
        transitions_with_freqs(transitions, r_f, sample_f),
    )
    if attribution is None:
        ax.scatter(
            s_fluxs, s_freqs, color="blue", s=12, alpha=0.5, zorder=2, label="_points"
        )
    else:
        colors = np.array(
            [f"C{int(index) % 10}" for index in attribution.transition_indices]
        )
        outliers = attribution.outlier_mask
        ax.scatter(
            s_fluxs[~outliers],
            s_freqs[~outliers],
            c=colors[~outliers],
            s=16,
            alpha=0.8,
            zorder=2,
            label="_points",
        )
        ax.scatter(
            s_fluxs[outliers],
            s_freqs[outliers],
            facecolors="none",
            edgecolors=[str(color) for color in colors[outliers]],
            s=30,
            linewidths=1.2,
            zorder=3,
            label="_outliers",
        )
    if show_const_freq:
        _plot_constant_freqs(ax, r_f, sample_f)

    ax.set_xlabel("Flux")
    ax.set_ylabel("Frequency (GHz)")
    if title:
        ax.set_title(title)
    handles, labels = ax.get_legend_handles_labels()
    keep = [
        (handle, label)
        for handle, label in zip(handles, labels, strict=False)
        if not label.startswith("_")
    ]
    if keep:
        ax.legend(
            [handle for handle, _ in keep],
            [label for _, label in keep],
            fontsize=7,
            loc="upper right",
        )

    # Limits: explicit override, else auto (notebook auto_derive_limits parity).
    auto_x, auto_y = derive_auto_limits(spectrums, s_fluxs, s_freqs, r_f, sample_f)
    ax.set_xlim(*(xlim if xlim is not None else auto_x))
    ax.set_ylim(*(ylim if ylim is not None else auto_y))

    if flux_half is not None and flux_period:
        _add_dev_value_axis(ax, flux_half, flux_period)
