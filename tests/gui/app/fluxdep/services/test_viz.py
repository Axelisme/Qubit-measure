"""Tests for the matplotlib fit visualiser (headless, Agg backend)."""

from __future__ import annotations

from zcu_tools.analysis.fluxdep.search import ParamBounds
import matplotlib

matplotlib.use("Agg")

import numpy as np
from matplotlib.figure import Figure
from zcu_tools.analysis.fluxdep.models import PointsData, TransitionDict
from zcu_tools.analysis.spectrum import SpectrumData
from zcu_tools.gui.app.fluxdep.services.viz import render_fit_figure
from zcu_tools.gui.app.fluxdep.state import SpectrumEntry


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


def test_published_filter_overlays_complete_high_order_kernel_frequencies():
    from zcu_tools.gui.app.fluxdep.controller import Controller
    from zcu_tools.gui.app.fluxdep.state import FluxDepState

    state = FluxDepState()
    state.put_spectrum(_entry())
    state.set_fit_params(
        "",
        ParamBounds(EJ=(2, 15), EC=(0.2, 2), EL=(0.1, 2)),
        {"transitions": [(0, 16)], "mirror20": [(0, 1)]},
        None,
        20.0,
    )
    state.set_fit_result((3.0, 0.8, 0.5))
    ctrl = Controller(state)
    try:
        figure = Figure()
        ctrl.render_selection(figure)
        lines = figure.axes[0].lines
        assert [line.get_label() for line in lines] == ["0 -> 16", "20 0 -> 1 mirror"]
        assert all(np.isfinite(line.get_ydata()).all() for line in lines)
    finally:
        ctrl.interactive.dispose()


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


def _grid_entry(
    name: str, fluxs, freqs, *, nan_freq_rows: slice | None = None
) -> SpectrumEntry:
    """A TwoTone spectrum on the given axes; nan_freq_rows become all-NaN."""
    signals = np.ones((len(fluxs), len(freqs)), dtype=np.complex128)
    if nan_freq_rows is not None:
        signals[:, nan_freq_rows] = np.nan
    empty = np.empty(0, dtype=np.float64)
    return SpectrumEntry(
        name=name,
        spec_type="TwoTone",
        raw=SpectrumData(
            dev_values=np.asarray(fluxs, dtype=np.float64).copy(),
            fluxs=np.asarray(fluxs, dtype=np.float64),
            freqs=np.asarray(freqs, dtype=np.float64),
            signals=signals,
        ),
        points=PointsData(
            dev_values=empty.copy(), fluxs=empty.copy(), freqs=empty.copy()
        ),
        flux_half=0.0,
        flux_int=0.5,
        flux_period=1.0,
        aligned=True,
    )


def _background_order(spectrums: dict[str, SpectrumEntry]) -> list[int]:
    """Painted order of background meshes, identified by their cell counts."""
    from matplotlib.collections import QuadMesh

    fig = Figure()
    t_fluxs = np.linspace(0.0, 0.5, 10)
    render_fit_figure(
        fig,
        spectrums,
        t_fluxs,
        _energies(t_fluxs),
        TransitionDict({"transitions": [(0, 1)]}),
        np.array([0.1]),
        np.array([5.0]),
    )
    meshes = [c for c in fig.axes[0].collections if isinstance(c, QuadMesh)]
    assert len({mesh.get_zorder() for mesh in meshes}) == 1
    sizes = []
    for mesh in meshes:
        cells = mesh.get_array()
        assert cells is not None
        sizes.append(cells.size)
    return sizes


def test_background_spectra_paint_from_low_to_high_resolution():
    # Resolution = sqrt(r_flux * r_freq), r = (points - 1) / range per axis.
    coarse = _grid_entry(  # sqrt(50 * 200) = 100
        "coarse", np.linspace(0.0, 1.0, 51), np.linspace(4.0, 4.5, 101)
    )
    fine = _grid_entry(  # sqrt(200 * 2000) = 632
        "fine", np.linspace(0.2, 0.4, 41), np.linspace(4.2, 4.3, 201)
    )
    order = _background_order({"fine": fine, "coarse": coarse})
    assert order == [51 * 101, 41 * 201]


def test_background_resolution_ignores_all_nan_rows():
    # padded: 401 raw freq rows over 0.4 GHz (raw r = 1000), but only the first
    # 11 and last 10 hold data: r = 20 / 0.4 = 50, sqrt(50 * 50) = 50.
    # dense: 101 rows over 0.2 GHz, sqrt(50 * 500) = 158. Raw counts would put
    # padded on top; its usable resolution puts it below.
    padded = _grid_entry(
        "padded",
        np.linspace(0.0, 1.0, 51),
        np.linspace(4.0, 4.4, 401),
        nan_freq_rows=slice(11, 391),
    )
    dense = _grid_entry("dense", np.linspace(0.0, 1.0, 51), np.linspace(4.0, 4.2, 101))
    order = _background_order({"dense": dense, "padded": padded})
    assert order == [51 * 21, 51 * 101]
