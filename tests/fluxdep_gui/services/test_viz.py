"""Tests for the matplotlib fit visualiser (headless, Agg backend)."""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib.figure import Figure
from zcu_tools.fluxdep_gui.services.viz import render_fit_figure
from zcu_tools.fluxdep_gui.state import SpectrumEntry
from zcu_tools.notebook.persistance import PointsData, SpectrumData, TransitionDict


def _entry() -> SpectrumEntry:
    fluxs = np.linspace(0.0, 0.5, 6).astype(np.float64)
    freqs = np.linspace(4.0, 6.0, 5).astype(np.float64)
    rng = np.random.RandomState(0)
    signals = (
        rng.randn(len(fluxs), len(freqs)) + 1j * rng.randn(len(fluxs), len(freqs))
    ).astype(np.complex128)
    raw = SpectrumData(
        dev_values=fluxs.copy(), fluxs=fluxs.copy(), freqs=freqs.copy(), signals=signals
    )
    empty = np.empty(0, dtype=np.float64)
    points = PointsData(dev_values=empty.copy(), fluxs=empty.copy(), freqs=empty.copy())
    return SpectrumEntry(
        name="s1",
        spec_type="TwoTone",
        raw=raw,
        points=points,
        flux_half=0.0,
        flux_int=0.5,
        flux_period=1.0,
        aligned=True,
    )


def _energies(t_fluxs):
    # (N, L) synthetic energies for energy2transition
    L = 4
    e = np.zeros((len(t_fluxs), L), dtype=np.float64)
    for lvl in range(L):
        e[:, lvl] = lvl + np.cos(2 * np.pi * t_fluxs)
    return e


def test_render_fit_figure_draws_all_layers():
    fig = Figure()
    spectrums = {"s1": _entry()}
    t_fluxs = np.linspace(0.0, 0.5, 50).astype(np.float64)
    energies = _energies(t_fluxs)
    transitions = TransitionDict({"transitions": [(0, 1), (0, 2)]})
    s_fluxs = np.array([0.1, 0.2, 0.3])
    s_freqs = np.array([5.0, 5.1, 5.2])

    render_fit_figure(
        fig,
        spectrums,
        t_fluxs,
        energies,
        transitions,
        s_fluxs,
        s_freqs,
        r_f=5.5,
        sample_f=9.0,
        flux_half=0.0,
        flux_period=1.0,
        title="EJ/EC/EL test",
    )

    assert len(fig.axes) >= 1
    ax = fig.axes[0]
    # background pcolormesh + 2 sim lines + points scatter + 3 const-freq hlines
    assert len(ax.collections) >= 1  # pcolormesh + scatter
    assert len(ax.lines) >= 2  # at least the two transition lines
    assert ax.get_title() == "EJ/EC/EL test"


def test_render_fit_figure_clears_prior():
    fig = Figure()
    fig.add_subplot(1, 1, 1).plot([0, 1], [0, 1])  # stale content
    spectrums = {"s1": _entry()}
    t_fluxs = np.linspace(0.0, 0.5, 20).astype(np.float64)
    render_fit_figure(
        fig,
        spectrums,
        t_fluxs,
        _energies(t_fluxs),
        TransitionDict({"transitions": [(0, 1)]}),
        np.array([0.1]),
        np.array([5.0]),
    )
    # cleared and rebuilt to a single axis (no dev-value secondary axis here)
    assert len(fig.axes) == 1


def test_render_fit_figure_without_dev_axis_ok():
    fig = Figure()
    t_fluxs = np.linspace(0.0, 0.5, 10).astype(np.float64)
    render_fit_figure(
        fig,
        {"s1": _entry()},
        t_fluxs,
        _energies(t_fluxs),
        TransitionDict({"transitions": [(0, 1)]}),
        np.array([0.1]),
        np.array([5.0]),
        flux_half=None,
        flux_period=None,
    )
    assert len(fig.axes) == 1  # no secondary axis when alignment omitted
