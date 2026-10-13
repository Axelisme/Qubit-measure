"""Tests for FitService — selected point cloud, compute_search, export_params.

``compute_search`` is exercised against a tiny synthetic fluxonium database
(params / fluxs / energies datasets) so the search converges deterministically
without the multi-hundred-MB real database. The key invariant under test (per the
search State refactor) is that ``compute_search`` performs NO State write — only
``record_result`` does.
"""

from __future__ import annotations

import os
from copy import deepcopy

import h5py
import numpy as np
import pytest
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.collections import PathCollection, QuadMesh
from matplotlib.figure import Figure
from zcu_tools.analysis.fluxdep.models import PointsData, TransitionDict
from zcu_tools.analysis.fluxdep.search import (
    DatabaseSearchResult,
    ParamBounds,
    SearchCancelled,
    SearchQuality,
    TransitionResidual,
)
from zcu_tools.analysis.spectrum import SpectrumData
from zcu_tools.gui.app.fluxdep.controller import Controller
from zcu_tools.gui.app.fluxdep.event_bus import FitChangedPayload
from zcu_tools.gui.app.fluxdep.services.fit import (
    FitService,
    default_params_path,
)
from zcu_tools.gui.app.fluxdep.services.store import SelectionService
from zcu_tools.gui.app.fluxdep.state import (
    FIT_VERSION_KEY,
    FluxDepState,
    SpectrumEntry,
)
from zcu_tools.gui.expected_error import FailedPreconditionError
from zcu_tools.gui.project import ProjectInfo

# --- fixtures --------------------------------------------------------------


@pytest.fixture
def tiny_database(tmp_path):
    """A 4-entry fluxonium database: params(4,3), fluxs(M,), energies(4,M,L).

    Energies are smooth synthetic curves vs flux; entry index 2 is the planted
    "answer" so the search has a clear best match for a point set generated from
    it. The exact physics is irrelevant — only that search_in_database can read
    the datasets and pick a finite-distance best candidate.
    """
    M, L = 21, 4
    fluxs = np.linspace(0.0, 0.5, M).astype(np.float64)
    params = np.array(
        [
            [3.0, 1.0, 0.5],
            [4.0, 0.8, 0.3],
            [5.0, 1.2, 0.4],  # planted answer
            [6.0, 0.9, 0.6],
        ],
        dtype=np.float64,
    )
    # energies[n, m, l] = level l energy at flux m for entry n (monotone-ish).
    energies = np.zeros((len(params), M, L), dtype=np.float64)
    for n, (EJ, EC, EL) in enumerate(params):
        for lvl in range(L):
            energies[n, :, lvl] = lvl * (EC + EL) + EJ * np.cos(2 * np.pi * fluxs) * 0.1
    path = tmp_path / "tiny_db.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("fluxs", data=fluxs)
        f.create_dataset("params", data=params)
        f.create_dataset("energies", data=energies)
    return str(path), params, fluxs, energies


def _aligned_entry_with_points(name: str, fluxs, freqs) -> SpectrumEntry:
    n = len(fluxs)
    raw = SpectrumData(
        dev_values=np.asarray(fluxs, dtype=np.float64),
        fluxs=np.asarray(fluxs, dtype=np.float64),
        freqs=np.linspace(4.0, 6.0, 5).astype(np.float64),
        signals=np.zeros((n, 5), dtype=np.complex128),
    )
    points = PointsData(
        dev_values=np.asarray(fluxs, dtype=np.float64),
        fluxs=np.asarray(fluxs, dtype=np.float64),
        freqs=np.asarray(freqs, dtype=np.float64),
    )
    return SpectrumEntry(
        name=name,
        spec_type="TwoTone",
        raw=raw,
        points=points,
        flux_half=0.0,
        flux_int=0.5,
        flux_period=1.0,
        aligned=True,
        points_completed=True,
    )


def _state_with_points() -> FluxDepState:
    st = FluxDepState()
    st.put_spectrum(
        _aligned_entry_with_points("s1", [0.0, 0.1, 0.2, 0.3], [5.0, 5.05, 5.1, 5.15])
    )
    return st


# --- selected_pointcloud ---------------------------------------------------


def test_selected_pointcloud_no_mask_returns_all():
    st = _state_with_points()
    fluxs, freqs = FitService(st).selected_pointcloud()
    assert fluxs.shape == (4,)
    np.testing.assert_allclose(freqs, [5.0, 5.05, 5.1, 5.15])


def test_selected_pointcloud_skips_empty_completed_spectra():
    st = _state_with_points()
    st.put_spectrum(_aligned_entry_with_points("noise", [], []))
    svc = FitService(st)
    fluxs, freqs = svc.selected_pointcloud()
    assert fluxs.size == 4
    np.testing.assert_array_equal(freqs, [5.0, 5.05, 5.1, 5.15])


def test_selected_pointcloud_applies_mask():
    st = _state_with_points()
    st.set_selection(np.array([True, False, True, False]))
    fluxs, freqs = FitService(st).selected_pointcloud()
    assert fluxs.shape == (2,)
    np.testing.assert_allclose(freqs, [5.0, 5.1])


def test_selected_pointcloud_includes_new_spectrum_without_changing_filter():
    st = _state_with_points()
    st.set_selection(np.array([True, False, True, False]))
    st.put_spectrum(_aligned_entry_with_points("s2", [0.4], [5.2]))
    before = st.version.snapshot()
    fluxs, freqs = FitService(st).selected_pointcloud()
    np.testing.assert_allclose(fluxs, [0.0, 0.2, 0.4])
    np.testing.assert_allclose(freqs, [5.0, 5.1, 5.2])
    assert st.version.snapshot() == before


# --- compute_search / record_result ----------------------------------------


# Wide bounds so some scale is always feasible for the synthetic database (the
# breakpoint search scales each entry's params; narrow bounds can reject every
# scale, which is a real "infeasible" error, not what these tests probe).
_WIDE = ((0.1, 50.0), (0.01, 10.0), (0.01, 10.0))


def test_removed_last_spectrum_has_empty_cloud_and_no_selected_points(tiny_database):
    st = _state_with_points()
    SelectionService(st).set_selection(np.array([True, False, True, False]))
    svc = FitService(st)
    svc.set_params(tiny_database[0], *_WIDE, TransitionDict({}), 0.0, 0.0)
    st.remove_spectrum("s1")
    before = st.version.snapshot()
    fluxs, freqs = svc.selected_pointcloud()
    assert fluxs.size == freqs.size == 0
    with pytest.raises(FailedPreconditionError) as caught:
        svc.capture_search()
    assert caught.value.reason_code == "no_selected_points"
    assert st.version.snapshot() == before
    assert st.selection_mask() is None


@pytest.mark.parametrize("publish_empty_selection", [False, True])
def test_empty_collection_without_stale_selection(
    tiny_database, publish_empty_selection
):
    st = FluxDepState()
    if publish_empty_selection:
        SelectionService(st).set_selection(np.empty(0, dtype=np.bool_))
    svc = FitService(st)
    svc.set_params(tiny_database[0], *_WIDE, TransitionDict({}), 0.0, 0.0)
    before = st.version.snapshot()

    fluxs, freqs = svc.selected_pointcloud()
    assert fluxs.shape == freqs.shape == (0,)
    assert fluxs.dtype == freqs.dtype == np.dtype(np.float64)
    with pytest.raises(FailedPreconditionError) as caught:
        svc.capture_search()
    assert caught.value.category == "failed_precondition"
    assert caught.value.reason_code == "no_selected_points"
    assert st.version.snapshot() == before


def test_compute_search_does_not_touch_state(tiny_database):
    db_path = tiny_database[0]
    st = _state_with_points()
    svc = FitService(st)
    svc.set_params(
        db_path,
        *_WIDE,
        TransitionDict({"transitions": [(0, 1), (0, 2)]}),
        0.0,
        0.0,
    )
    fit_version_before = st.version.get(FIT_VERSION_KEY)

    result = svc.compute_search(svc.capture_search())

    # compute_search must NOT write State (no result recorded, no extra bump).
    assert isinstance(result, DatabaseSearchResult)
    assert len(result.params) == 3
    assert st.fit.params is None  # still no result on State
    assert st.version.get(FIT_VERSION_KEY) == fit_version_before


@pytest.mark.parametrize("synchronous", [False, True])
def test_controller_cancelled_search_keeps_published_fit(tiny_database, synchronous):
    st = _state_with_points()
    ctrl = Controller(st)
    ctrl.set_fit_params(
        tiny_database[0],
        *_WIDE,
        TransitionDict({"transitions": [(0, 1), (0, 2)]}),
        0.0,
        0.0,
    )
    st.set_fit_result((3.0, 1.0, 0.5))
    version = st.version.get(FIT_VERSION_KEY)
    changes: list[FitChangedPayload] = []
    subscription = ctrl.bus.subscribe(FitChangedPayload, changes.append)
    search = (
        ctrl.search_database
        if synchronous
        else lambda **kwargs: ctrl.compute_search(ctrl.capture_search(), **kwargs)
    )
    try:
        with pytest.raises(SearchCancelled):
            search(cancel_requested=lambda: True)
        assert st.fit.params == (3.0, 1.0, 0.5)
        assert st.version.get(FIT_VERSION_KEY) == version
        assert changes == []
    finally:
        subscription.unsubscribe()
        ctrl.interactive.dispose()


def test_headless_publication_keeps_captured_unwrapped_coordinates_and_attribution(
    monkeypatch,
):
    from zcu_tools.gui.app.fluxdep.services import fit, viz

    source_fluxs = np.array([-0.2, 1.2, 1.5, 2.2])
    source_freqs = np.array([4.0, 5.0, 6.0, 7.0])
    state = FluxDepState()
    state.put_spectrum(_aligned_entry_with_points("source", source_fluxs, source_freqs))
    state.set_fit_params(
        "synthetic.h5", *_WIDE, {"transitions": [(0, 1), (0, 2)]}, None, None
    )
    numerical = DatabaseSearchResult(
        params=(3.0, 0.8, 0.5),
        best_distance=0.15,
        best_scale=1.0,
        best_index=0,
        entry_results=np.array([[0.15, 1.0]]),
        entry_params=np.array([[3.0, 0.8, 0.5]]),
        fluxs=source_fluxs % 1,
        freqs=source_freqs.copy(),
        predicted_freqs=source_freqs.copy(),
        bounds=ParamBounds(EJ=_WIDE[0], EC=_WIDE[1], EL=_WIDE[2]),
        transition_indices=np.array([1, 0, 1, 0], dtype=np.int64),
        residuals=np.array([0.1, 0.6, 0.2, 0.1]),
        outlier_mask=np.array([False, True, False, False]),
        quality=SearchQuality(
            0.15,
            1,
            0.45,
            (
                TransitionResidual("transitions 0-1", 2, 0.35, 0.6),
                TransitionResidual("transitions 0-2", 2, 0.15, 0.2),
            ),
        ),
    )
    monkeypatch.setattr(fit, "search_database", lambda *args, **kwargs: numerical)

    def simulate(params, fluxs, *, cutoff, evals_count):
        return fluxs, np.tile(np.arange(evals_count, dtype=np.float64), (fluxs.size, 1))

    monkeypatch.setattr(viz, "calculate_energy_vs_flux", simulate)
    ctrl = Controller(state)
    try:
        result = ctrl.search_database()
        assert result is numerical
        assert state.fit_stale_reasons == []
        figure = Figure()
        ctrl.render_fit(figure)
        scatters = [
            artist
            for artist in figure.axes[0].collections
            if isinstance(artist, PathCollection)
        ]
        np.testing.assert_allclose(
            np.asarray(scatters[0].get_offsets(), dtype=np.float64),
            [[-0.2, 4.0], [1.5, 6.0], [2.2, 7.0]],
        )
        np.testing.assert_allclose(
            np.asarray(scatters[1].get_offsets(), dtype=np.float64), [[1.2, 5.0]]
        )
        recorded = state.fit.result
        if recorded is None:
            pytest.fail("full search publication must retain attribution")
        np.testing.assert_array_equal(recorded.transition_indices, [1, 0, 1, 0])
        state.set_selection(np.array([True, False, False, False]))
        source_fluxs[:] = 9
        source_freqs[:] = 10
        assert state.fit_stale_reasons == ["filter changed"]
        ctrl.render_fit(figure)
        scatters = [
            artist
            for artist in figure.axes[0].collections
            if isinstance(artist, PathCollection)
        ]
        np.testing.assert_allclose(
            np.asarray(scatters[0].get_offsets(), dtype=np.float64),
            [[-0.2, 4.0], [1.5, 6.0], [2.2, 7.0]],
        )
        np.testing.assert_allclose(
            np.asarray(scatters[1].get_offsets(), dtype=np.float64), [[1.2, 5.0]]
        )
    finally:
        ctrl.interactive.dispose()


def test_record_result_writes_state(tiny_database):
    db_path = tiny_database[0]
    st = _state_with_points()
    svc = FitService(st)
    svc.set_params(
        db_path,
        *_WIDE,
        TransitionDict({"transitions": [(0, 1), (0, 2)]}),
        0.0,
        0.0,
    )
    before = st.version.get(FIT_VERSION_KEY)
    inputs = svc.capture_search()
    result = svc.compute_search(inputs)
    svc.record_result(result, inputs)
    assert st.fit.params == result.params
    assert st.fit.has_result
    assert st.version.get(FIT_VERSION_KEY) == before + 1


def test_capture_search_fast_fails_without_database():
    st = _state_with_points()
    before = st.version.snapshot()
    with pytest.raises(FailedPreconditionError) as caught:
        FitService(st).capture_search()
    assert caught.value.reason_code == "no_search_db"
    assert st.version.snapshot() == before


def test_capture_search_fast_fails_without_points(tiny_database):
    db_path = tiny_database[0]
    st = FluxDepState()
    st.put_spectrum(_aligned_entry_with_points("noise", [], []))
    assert st.spectrums["noise"].points_completed
    svc = FitService(st)
    svc.set_params(db_path, *_WIDE, TransitionDict({}), 0.0, 0.0)
    before = st.version.snapshot()
    with pytest.raises(FailedPreconditionError) as caught:
        svc.capture_search()
    assert caught.value.reason_code == "no_selected_points"
    assert st.version.snapshot() == before


@pytest.mark.parametrize("shift", [1.0, 0.37])
def test_show_keeps_search_capture_frame_after_alignment_and_source_publication(
    monkeypatch, shift
):
    from tests.gui.app.fluxdep._search_result import perfect_fit_result
    from zcu_tools.gui.app.fluxdep.services import viz

    def simulate(params, fluxs, *, cutoff, evals_count):
        return fluxs, np.tile(np.arange(evals_count, dtype=np.float64), (fluxs.size, 1))

    monkeypatch.setattr(viz, "calculate_energy_vs_flux", simulate)
    state = FluxDepState()
    entry = SpectrumEntry(
        name="same-spectrum",
        spec_type="TwoTone",
        raw=SpectrumData(
            dev_values=np.array([0.0, 0.2]),
            fluxs=np.array([0.0, 0.2]),
            freqs=np.array([3.0, 4.0]),
            signals=np.array([[1 + 0j, 1 + 1j], [1 - 1j, -1 + 0j]]),
        ),
        points=PointsData(
            dev_values=np.array([0.1]),
            fluxs=np.array([0.1]),
            freqs=np.array([3.5]),
        ),
        points_completed=True,
    )
    state.put_spectrum(entry)
    ctrl = Controller(state)
    service = FitService(state)
    figure = Figure()
    canvas = FigureCanvasAgg(figure)

    def record():
        inputs = service.capture_search()
        result = perfect_fit_result(inputs.fluxs % 1, inputs.freqs, inputs.bounds)
        ctrl.record_search_result(result, inputs)

    def expect_frame(centers, point, device_limits, frequency_limits):
        ctrl.render_fit(figure)
        canvas.draw()
        ax = figure.axes[0]
        mesh = next(c for c in ax.collections if isinstance(c, QuadMesh))
        scatter = next(c for c in ax.collections if isinstance(c, PathCollection))
        edges = np.asarray(mesh.get_coordinates(), dtype=np.float64)[0, :, 0]
        np.testing.assert_allclose((edges[:-1] + edges[1:]) / 2, centers)
        np.testing.assert_allclose(
            np.asarray(scatter.get_offsets(), dtype=np.float64), [[point, 3.5]]
        )
        np.testing.assert_allclose(ax.get_xlim(), centers)
        np.testing.assert_allclose(ax.get_ylim(), frequency_limits)
        np.testing.assert_allclose(ax.child_axes[0].get_xlim(), device_limits, atol=1e-12)
        np.testing.assert_allclose(
            np.asarray(ax.lines[0].get_xdata(), dtype=np.float64)[[0, -1]], centers
        )
        return ax.get_title()

    try:
        ctrl.set_alignment(entry.name, 0.6, -0.4)
        state.set_fit_params("synthetic", *_WIDE, {"transitions": [(0, 1)]}, None, None)
        record()
        assert "Outdated" not in expect_frame([0.2, 0.3], 0.25, [0.0, 0.2], [3.0, 4.0])
        ctrl.set_alignment(entry.name, 0.6 + shift, -0.4 + shift)
        assert "Outdated: same-spectrum re-aligned" in expect_frame(
            [0.2, 0.3], 0.25, [0.0, 0.2], [3.0, 4.0]
        )
        # Source changes cannot rewrite the recorded background or calibration.
        replacement = deepcopy(state.get_spectrum(entry.name))
        replacement.raw["freqs"][:] = [7.0, 8.0]
        replacement.raw["signals"][:] = 0
        state.put_spectrum(replacement)
        expect_frame([0.2, 0.3], 0.25, [0.0, 0.2], [3.0, 4.0])
        record()
        new_centers = np.array([0.2, 0.3]) - shift / 2
        new_point = 0.25 - shift / 2
        assert "Outdated" not in expect_frame(
            new_centers, new_point, [0.0, 0.2], [3.5, 8.0]
        )
        # Params-only publication retires the old search frame as well as quality.
        state.set_fit_result((5.0, 1.0, 0.5))
        ctrl.remove_spectrum(entry.name)
        ctrl.render_fit(figure)
        assert not any(isinstance(c, QuadMesh) for c in figure.axes[0].collections)
    finally:
        ctrl.interactive.dispose()


# --- export_params ---------------------------------------------------------


def test_export_params_writes_json(tmp_path):
    st = _state_with_points()
    st.project.chip_name = "Q9"
    st.project.qub_name = "Q1"
    svc = FitService(st)
    st.set_fit_result((5.0, 1.2, 0.4))

    out = str(tmp_path / "params.json")
    path = svc.export_params(out)
    assert path == out

    from zcu_tools.resources.qubit_params import QubitParams

    fluxdep_fit = QubitParams(out, readonly=True).require_fluxdep_fit()
    assert fluxdep_fit.params == (5.0, 1.2, 0.4)
    assert fluxdep_fit.flux_half == 0.0
    assert fluxdep_fit.flux_period == 1.0
    assert fluxdep_fit.timestamp is not None


def test_export_params_fast_fails_without_result(tmp_path):
    st = _state_with_points()
    path = tmp_path / "params.json"
    before = st.version.snapshot()
    with pytest.raises(FailedPreconditionError) as caught:
        FitService(st).export_params(str(path))
    assert caught.value.reason_code == "no_fit_result"
    assert not path.exists()
    assert st.version.snapshot() == before


def test_export_params_fast_fails_without_aligned(tmp_path):
    st = FluxDepState()
    st.set_fit_result((5.0, 1.2, 0.4))
    path = tmp_path / "params.json"
    before = st.version.snapshot()
    with pytest.raises(FailedPreconditionError) as caught:
        FitService(st).export_params(str(path))
    assert caught.value.reason_code == "no_aligned_spectrum"
    assert not path.exists()
    assert st.version.snapshot() == before


def test_default_params_path():
    assert default_params_path(os.path.join("result", "Q9", "Q1")) == os.path.join(
        "result", "Q9", "Q1", "params.json"
    )


def test_export_params_uses_project_result_dir(tmp_path, monkeypatch):
    # No savepath → write under the project's result_dir, which ProjectInfo derives
    # from chip/qubit (result/<chip>/<qub>) in __post_init__. Reproduces (now fixed)
    # the "[Errno 2] ... ''" crash that came from a bare 'params.json'.
    monkeypatch.chdir(tmp_path)
    st = _state_with_points()
    st.project = ProjectInfo(chip_name="Q9", qub_name="Q1")
    st.set_fit_result((5.0, 1.2, 0.4))
    path = FitService(st).export_params()  # no savepath
    assert path == os.path.join("result", "Q9", "Q1", "params.json")
    assert os.path.isfile(path)
