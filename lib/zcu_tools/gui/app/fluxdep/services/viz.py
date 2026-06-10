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

The energies are computed by the caller (``calculate_energy_vs_flux``); this
module only turns them into transition lines via ``energy2transition``.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

from zcu_tools.gui.app.fluxdep.state import SpectrumEntry, transitions_with_freqs
from zcu_tools.notebook.analysis.fluxdep.models import energy2transition
from zcu_tools.notebook.analysis.fluxdep.processing import cast2real_and_norm
from zcu_tools.notebook.persistance import TransitionDict
from zcu_tools.simulate import flux2value

logger = logging.getLogger(__name__)


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
        spect = entry.raw
        signals = spect["signals"]
        flux_mask = np.any(~np.isnan(signals), axis=1)
        freq_mask = np.any(~np.isnan(signals), axis=0)
        if not flux_mask.any() or not freq_mask.any():
            continue
        fluxs = np.asarray(spect["fluxs"], dtype=np.float64)[flux_mask]
        freqs = np.asarray(spect["freqs"], dtype=np.float64)[freq_mask]
        _acc_x(float(fluxs.min()), float(fluxs.max()))
        _acc_y(float(freqs.min()), float(freqs.max()))

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


def _plot_background(ax, spectrums: dict[str, SpectrumEntry]) -> None:
    """Draw each spectrum's normalised amplitude as a gray_r heatmap."""
    for entry in spectrums.values():
        spect = entry.raw
        signals = spect["signals"]
        flux_mask = np.any(~np.isnan(signals), axis=1)
        freq_mask = np.any(~np.isnan(signals), axis=0)
        if not flux_mask.any() or not freq_mask.any():
            continue
        signals = signals[flux_mask, :][:, freq_mask]
        fluxs = np.asarray(spect["fluxs"], dtype=np.float64)[flux_mask]
        freqs = np.asarray(spect["freqs"], dtype=np.float64)[freq_mask]
        real = cast2real_and_norm(signals, use_phase=entry.spec_type != "OneTone")
        ax.pcolormesh(
            fluxs,
            freqs,
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
) -> None:
    """Draw the fitted transition frequencies vs flux (one line per transition)."""
    freqs, labels = energy2transition(energies, transitions)
    for i, label in enumerate(labels):
        ax.plot(t_fluxs, freqs[:, i], label=label, linewidth=1.0, zorder=1)


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
    _plot_simulation_lines(ax, t_fluxs, energies, sim_transitions)
    ax.scatter(
        s_fluxs, s_freqs, color="blue", s=12, alpha=0.5, zorder=2, label="_points"
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
        for handle, label in zip(handles, labels)
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
