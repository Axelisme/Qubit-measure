from __future__ import annotations

import gc
from concurrent.futures import ThreadPoolExecutor
from io import BytesIO
from weakref import ref

import matplotlib
import numpy as np
import pytest
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from zcu_tools.plotting.figures import FigureCollection


@pytest.fixture(autouse=True)
def collect_figure_cycles():
    yield
    gc.collect()


def test_subplots_are_native_registered_and_saveable_without_presentation() -> None:
    figures = FigureCollection()
    managers = plt.get_fignums()
    backend = matplotlib.get_backend()
    figure, ax = figures.subplots("fit", figsize=(4, 3), dpi=80)
    assert isinstance(ax, Axes)
    assert ax.figure is figure
    assert figures["fit"] is figure
    ax.plot([0, 1], [1, 0])
    output = BytesIO()
    figure.savefig(output, format="png")
    assert output.getvalue().startswith(b"\x89PNG\r\n\x1a\n")
    assert tuple(figure.get_size_inches()) == (4, 3)
    assert plt.get_fignums() == managers
    assert matplotlib.get_backend() == backend


def test_subplots_preserve_native_array_shape_and_shared_axes() -> None:
    figures = FigureCollection()
    figure, axes = figures.subplots("panels", nrows=2, ncols=2, sharex="col")
    assert isinstance(axes, np.ndarray)
    assert axes.shape == (2, 2)
    top, bottom = axes[0, 0], axes[1, 0]
    assert isinstance(top, Axes)
    assert isinstance(bottom, Axes)
    assert top.get_shared_x_axes().joined(top, bottom)
    for ax in axes.flat:
        assert isinstance(ax, Axes)
        assert ax.figure is figure
    _, unsqueezed = figures.subplots("single", squeeze=False)
    assert isinstance(unsqueezed, np.ndarray)
    assert unsqueezed.shape == (1, 1)


def test_subplots_conflict_and_seal_leave_existing_figure_unchanged() -> None:
    figures = FigureCollection()
    figure, ax = figures.subplots("fit")
    ax.set_title("original")
    with pytest.raises(ValueError, match="already belongs"):
        figures.subplots("fit")
    figures.seal()
    with pytest.raises(RuntimeError, match="sealed"):
        figures.subplots("new")
    assert dict(figures) == {"fit": figure}
    assert ax.get_title() == "original"


def test_named_figures_preserve_order_identity_and_idempotence() -> None:
    figures = FigureCollection()
    fit, diagnostic = Figure(), Figure()
    figures.adopt("fit", fit)
    figures.adopt("diagnostic", diagnostic)
    figures.adopt("fit", fit)
    assert list(figures) == ["fit", "diagnostic"]
    assert len(figures) == 2
    assert figures["fit"] is fit
    assert figures["diagnostic"] is diagnostic
    with pytest.raises(KeyError, match="missing"):
        figures["missing"]


def test_conflicts_preserve_both_owners_and_canvas() -> None:
    figures, other = FigureCollection(), FigureCollection()
    fit, diagnostic = Figure(), Figure()
    canvas = fit.canvas
    figures.adopt("fit", fit)
    other.adopt("diagnostic", diagnostic)
    with pytest.raises(ValueError, match="already belongs"):
        figures.adopt("fit", diagnostic)
    with pytest.raises(ValueError, match="already has a name"):
        figures.adopt("alias", fit)
    with pytest.raises(ValueError, match="owned by another"):
        other.adopt("fit", fit)
    assert dict(figures) == {"fit": fit}
    assert dict(other) == {"diagnostic": diagnostic}
    assert fit.canvas is canvas


def test_empty_name_does_not_claim_figure() -> None:
    figures, other = FigureCollection(), FigureCollection()
    figure = Figure()
    with pytest.raises(ValueError, match="must not be empty"):
        figures.adopt("", figure)
    other.adopt("valid", figure)
    assert not figures
    assert other["valid"] is figure


def test_sealed_collection_retains_editable_saveable_figures() -> None:
    figures = FigureCollection()
    figure = Figure()
    ax = figure.subplots()
    figures.adopt("fit", figure)
    figures.seal()
    figures.seal()
    figures.adopt("fit", figure)
    rejected = Figure()
    with pytest.raises(RuntimeError, match="sealed"):
        figures.adopt("extra", rejected)
    other = FigureCollection()
    other.adopt("extra", rejected)
    with pytest.raises(ValueError, match="owned by another"):
        other.adopt("fit", figure)
    canvas = figure.canvas
    ax.set_title("edited after completion")
    ax.plot([0, 1], [1, 0])
    output = BytesIO()
    figures["fit"].savefig(output, format="png")
    assert output.getvalue().startswith(b"\x89PNG\r\n\x1a\n")
    assert figure.canvas is canvas
    assert list(figures) == ["fit"]


def test_releasing_collection_releases_ownership_without_closing_figure() -> None:
    figures = FigureCollection()
    figure = Figure()
    canvas = figure.canvas
    figures.adopt("fit", figure)
    figures.seal()
    owner_ref = ref(figures)
    del figures
    gc.collect()
    assert owner_ref() is None
    other = FigureCollection()
    other.adopt("retained", figure)
    assert other["retained"] is figure
    assert figure.canvas is canvas


def test_registry_does_not_keep_unreferenced_figures_alive() -> None:
    figures = FigureCollection()
    figure = Figure()
    figure_ref = ref(figure)
    figures.adopt("fit", figure)
    del figure
    gc.collect()
    assert figure_ref() is not None
    del figures
    gc.collect()
    assert figure_ref() is None


def test_figure_callback_cycle_does_not_keep_owner_alive() -> None:
    figures = FigureCollection()
    figure = Figure()
    figures.adopt("fit", figure)
    figure.canvas.mpl_connect("draw_event", lambda _event, owner=figures: len(owner))
    owner_ref, figure_ref = ref(figures), ref(figure)
    del figures, figure
    gc.collect()
    assert owner_ref() is None
    assert figure_ref() is None


def test_concurrent_claims_have_one_owner() -> None:
    figure = Figure()
    collections = [FigureCollection(), FigureCollection()]

    def claim(figures: FigureCollection) -> bool:
        try:
            figures.adopt("fit", figure)
        except ValueError:
            return False
        return True

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(claim, collections))
    assert sorted(results) == [False, True]
    winner = collections[results.index(True)]
    loser = collections[results.index(False)]
    assert winner["fit"] is figure
    assert not loser
