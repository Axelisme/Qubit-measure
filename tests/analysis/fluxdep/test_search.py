"""Search kernel and pyplot diagnostic integration contracts."""

from io import BytesIO
from threading import Event

import h5py
import matplotlib.pyplot as plt
import numpy as np
import pytest
from numpy.typing import NDArray
from zcu_tools.analysis.fluxdep.models import TransitionDict
from zcu_tools.analysis.fluxdep.search import (
    ParamBounds,
    SearchCancelled,
    SearchExecution,
    search_database,
)
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


def test_pre_cancel_does_not_load_database(tmp_path):
    with pytest.raises(SearchCancelled):
        search_database(
            np.array([0.0]),
            np.array([1.0]),
            str(tmp_path / "missing.h5"),
            TransitionDict({"transitions": [(0, 1)]}),
            ParamBounds(EJ=(0.1, 10.0), EC=(0.1, 10.0), EL=(0.1, 10.0)),
            execution=SearchExecution(cancel_requested=lambda: True),
        )


def test_false_cancel_predicate_preserves_numeric_result(tmp_path):
    args = _search_args(tmp_path)
    bounds = ParamBounds(EJ=args[4], EC=args[5], EL=args[6])
    ordinary = search_database(*args[:4], bounds)
    cancellable = search_database(
        *args[:4], bounds, execution=SearchExecution(cancel_requested=lambda: False)
    )
    assert cancellable.params == ordinary.params
    assert cancellable.best_index == ordinary.best_index
    assert cancellable.best_distance == ordinary.best_distance
    assert cancellable.best_scale == ordinary.best_scale
    np.testing.assert_array_equal(cancellable.entry_results, ordinary.entry_results)
    np.testing.assert_array_equal(cancellable.entry_params, ordinary.entry_params)
    np.testing.assert_array_equal(cancellable.fluxs, ordinary.fluxs)
    np.testing.assert_array_equal(cancellable.freqs, ordinary.freqs)
    np.testing.assert_array_equal(cancellable.predicted_freqs, ordinary.predicted_freqs)


@pytest.mark.parametrize("progress_fails", [False, True])
def test_stop_during_scan_closes_progress_without_partial_result(
    tmp_path, progress_fails
):
    args = _search_args(tmp_path)
    _repeat_database_entries(args[2], 65)
    stop = Event()
    closed = Event()

    class StoppingProgress(TQDMProgressBar):
        def update(self, value=1):
            stop.set()
            if progress_fails:
                raise ArithmeticError("progress failed")
            super().update(value)

        def close(self):
            super().close()
            closed.set()

    expected = ArithmeticError if progress_fails else SearchCancelled
    with use_pbar_factory(StoppingProgress), pytest.raises(expected):
        search_database(
            *args[:4],
            ParamBounds(EJ=args[4], EC=args[5], EL=args[6]),
            execution=SearchExecution(cancel_requested=stop.is_set),
        )
    assert stop.is_set()
    assert closed.is_set()


def test_predicate_interrupt_during_scan_does_not_return_partial_result(tmp_path):
    args = _search_args(tmp_path)
    _repeat_database_entries(args[2], 65)
    progressed = Event()

    class ObservingProgress(TQDMProgressBar):
        def update(self, value=1):
            progressed.set()
            super().update(value)

    def interrupted_predicate() -> bool:
        if progressed.is_set():
            raise KeyboardInterrupt
        return False

    with use_pbar_factory(ObservingProgress), pytest.raises(KeyboardInterrupt):
        search_database(
            *args[:4],
            ParamBounds(EJ=args[4], EC=args[5], EL=args[6]),
            execution=SearchExecution(cancel_requested=interrupted_predicate),
        )
    assert progressed.is_set()


def test_cancel_predicate_failure_propagates(tmp_path):
    args = _search_args(tmp_path)

    def broken_predicate() -> bool:
        raise LookupError("stop source failed")

    with pytest.raises(LookupError, match="stop source failed"):
        search_database(
            *args[:4],
            ParamBounds(EJ=args[4], EC=args[5], EL=args[6]),
            execution=SearchExecution(cancel_requested=broken_predicate),
        )


def test_infeasible_bounds_raise_runtime_error(tmp_path):
    args = _search_args(tmp_path)
    with pytest.raises(RuntimeError, match="No valid candidate"):
        search_database(
            *args[:4], ParamBounds(EJ=(100, 101), EC=(100, 101), EL=(100, 101))
        )


def _repeat_database_entries(path: str, count: int) -> None:
    with h5py.File(path, "r+") as file:
        params_dataset = file["params"]
        energies_dataset = file["energies"]
        assert isinstance(params_dataset, h5py.Dataset)
        assert isinstance(energies_dataset, h5py.Dataset)
        first_param = params_dataset[0]
        first_energy = energies_dataset[0]
        del file["params"]
        del file["energies"]
        file.create_dataset("params", data=np.repeat(first_param[None], count, axis=0))
        file.create_dataset(
            "energies", data=np.repeat(first_energy[None], count, axis=0)
        )


def test_interrupt_after_progress_preserves_best_so_far(tmp_path):
    args = _search_args(tmp_path)
    _repeat_database_entries(args[2], 65)

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


@pytest.mark.parametrize("notebook", [False, True])
def test_final_progress_interrupt_preserves_best_so_far(tmp_path, notebook: bool):
    args = _search_args(tmp_path)
    closed = []

    class InterruptingProgress(TQDMProgressBar):
        def set_description(self, text: str) -> None:
            if text == "Done! ":
                raise KeyboardInterrupt
            super().set_description(text)

        def close(self):
            closed.append(True)
            super().close()

    with (
        use_pbar_factory(InterruptingProgress),
        pytest.warns(RuntimeWarning, match="best-so-far"),
    ):
        try:
            if notebook:
                params, figure = search_in_database(*args, plot=False)
                assert figure is None
            else:
                result = search_database(
                    *args[:4], ParamBounds(EJ=args[4], EC=args[5], EL=args[6])
                )
                params = result.params
                assert result.best_index == 0
                np.testing.assert_allclose(result.predicted_freqs, args[1])
        except KeyboardInterrupt:
            pytest.fail("final progress interruption discarded the best-so-far result")
    assert params == (3.0, 0.8, 0.5)
    assert closed


def test_empty_transitions_have_no_valid_candidate(tmp_path):
    args = _search_args(tmp_path)
    with pytest.raises(RuntimeError, match="No valid candidate"):
        search_database(
            *args[:3],
            TransitionDict({}),
            ParamBounds(EJ=args[4], EC=args[5], EL=args[6]),
        )
