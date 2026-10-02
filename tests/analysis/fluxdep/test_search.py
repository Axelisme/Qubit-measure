"""Search kernel and pyplot diagnostic integration contracts."""

from io import BytesIO

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.typing import NDArray
from zcu_tools.analysis.fluxdep.models import TransitionDict
from zcu_tools.analysis.fluxdep.search import ParamBounds, search_database
from zcu_tools.notebook.analysis.fluxdep.fitting import search_in_database
from zcu_tools.plotting.fluxdep import make_search_diagnostic_figure
from zcu_tools.progress_bar import use_pbar_factory
from zcu_tools.progress_bar.backend.tqdm import TQDMProgressBar


def _database(tmp_path):
    path = tmp_path / "search.h5"
    flux_grid = np.linspace(0, 0.5, 9)
    params = np.array([[3.0, 0.8, 0.5], [4.0, 1.1, 0.6]])
    energies = np.zeros((2, 9, 3))
    for i in range(2):
        energies[i, :, 1] = (i + 1) * (1 + flux_grid)
        energies[i, :, 2] = (i + 1) * (2 + flux_grid)
    with h5py.File(path, "w") as file:
        file.create_dataset("fluxs", data=flux_grid)
        file.create_dataset("params", data=params)
        file.create_dataset("energies", data=energies)
    return str(path)


def _search_args(
    tmp_path,
) -> tuple[
    NDArray[np.float64],
    NDArray[np.float64],
    str,
    TransitionDict,
    tuple[float, float],
    tuple[float, float],
    tuple[float, float],
]:
    return (
        np.array([0.0, 0.2, 0.4]),
        np.array([1.0, 1.2, 1.4]),
        _database(tmp_path),
        TransitionDict({"transitions": [(0, 1)]}),
        (0.1, 10.0),
        (0.1, 10.0),
        (0.1, 10.0),
    )


def test_kernel_agrees_with_notebook_and_entry_results(tmp_path):
    args = _search_args(tmp_path)
    result = search_database(*args[:4], ParamBounds(EJ=args[4], EC=args[5], EL=args[6]))
    params, figure = search_in_database(*args, plot=False)
    assert figure is None
    assert result.params == params
    assert result.entry_results.shape == (2, 2)
    assert result.best_index == np.nanargmin(result.entry_results[:, 0])
    assert result.best_distance == result.entry_results[result.best_index, 0]
    np.testing.assert_allclose(
        result.params, result.entry_params[result.best_index] * result.best_scale
    )
    np.testing.assert_allclose(result.predicted_freqs, args[1])


def test_builder_matches_notebook_figure_and_show(tmp_path, monkeypatch):
    args = _search_args(tmp_path)
    result = search_database(*args[:4], ParamBounds(EJ=args[4], EC=args[5], EL=args[6]))
    fig = make_search_diagnostic_figure(result)
    shows = []
    monkeypatch.setattr(
        "zcu_tools.notebook.analysis.fluxdep.fitting.display", shows.append
    )
    _, notebook_fig = search_in_database(*args, plot=True)
    try:
        assert notebook_fig is not None
        image = BytesIO()
        notebook_fig.savefig(image, format="png")
        assert len(shows) == 1
        assert shows[0].data == image.getvalue()
        assert fig.canvas.manager is None
        assert notebook_fig.canvas.manager is None
        params, hidden = search_in_database(*args, plot=False)
        assert params == result.params
        assert hidden is None
        assert len(shows) == 1
        assert len(fig.axes) == len(notebook_fig.axes) == 4
        assert fig.get_suptitle() == notebook_fig.get_suptitle()
        assert fig.axes[0].get_xlabel() == notebook_fig.axes[0].get_xlabel() == "Flux"
        assert fig.axes[0].get_ylabel() == "Frequency (GHz)"
        for left, right in zip(fig.axes, notebook_fig.axes, strict=True):
            assert len(left.collections) == len(right.collections)
            for a, b in zip(left.collections, right.collections, strict=True):
                np.testing.assert_array_equal(a.get_offsets(), b.get_offsets())
        for ax in fig.axes[1:]:
            assert ax.collections[0].get_rasterized()
    finally:
        plt.close(fig)
        plt.close(notebook_fig)


def test_infeasible_bounds_raise_runtime_error(tmp_path):
    args = _search_args(tmp_path)
    with pytest.raises(RuntimeError, match="No valid candidate"):
        search_database(
            *args[:4], ParamBounds(EJ=(100, 101), EC=(100, 101), EL=(100, 101))
        )


def test_interrupt_after_progress_preserves_best_so_far(tmp_path):
    args = _search_args(tmp_path)
    with h5py.File(args[2], "r+") as file:
        params_dataset = file["params"]
        energies_dataset = file["energies"]
        assert isinstance(params_dataset, h5py.Dataset)
        assert isinstance(energies_dataset, h5py.Dataset)
        first_param = params_dataset[0]
        first_energy = energies_dataset[0]
        del file["params"]
        del file["energies"]
        file.create_dataset("params", data=np.repeat(first_param[None], 65, axis=0))
        file.create_dataset("energies", data=np.repeat(first_energy[None], 65, axis=0))

    class InterruptingProgress(TQDMProgressBar):
        def update(self, value=1):
            raise KeyboardInterrupt

    with (
        use_pbar_factory(InterruptingProgress),
        pytest.warns(RuntimeWarning, match="best-so-far"),
    ):
        result = search_database(
            *args[:4], ParamBounds(EJ=args[4], EC=args[5], EL=args[6])
        )
    assert result.best_index == 0
    assert result.params == (3.0, 0.8, 0.5)
    assert result.entry_results[64, 0] == 0.0
    assert np.isnan(result.entry_results[64, 1])


def test_empty_transitions_have_no_valid_candidate(tmp_path):
    args = _search_args(tmp_path)
    with pytest.raises(RuntimeError, match="No valid candidate"):
        search_database(
            *args[:3],
            TransitionDict({}),
            ParamBounds(EJ=args[4], EC=args[5], EL=args[6]),
        )
