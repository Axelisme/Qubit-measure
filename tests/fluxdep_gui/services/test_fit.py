"""Tests for FitService — selected point cloud, compute_search, export_params.

``compute_search`` is exercised against a tiny synthetic fluxonium database
(params / fluxs / energies datasets) so the search converges deterministically
without the multi-hundred-MB real database. The key invariant under test (per the
search State refactor) is that ``compute_search`` performs NO State write — only
``record_result`` does.
"""

from __future__ import annotations

import os

import h5py
import numpy as np
import pytest
from zcu_tools.fluxdep_gui.services.fit import (
    FitService,
    SearchResult,
    default_params_path,
)
from zcu_tools.fluxdep_gui.state import FIT_VERSION_KEY, FluxDepState, SpectrumEntry
from zcu_tools.notebook.persistance import PointsData, SpectrumData, TransitionDict

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
        points_selected=True,
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


def test_selected_pointcloud_applies_mask():
    st = _state_with_points()
    st.set_selection(np.array([True, False, True, False]))
    fluxs, freqs = FitService(st).selected_pointcloud()
    assert fluxs.shape == (2,)
    np.testing.assert_allclose(freqs, [5.0, 5.1])


def test_selected_pointcloud_stale_mask_raises():
    st = _state_with_points()
    # selection set, then a point added → mask length now disagrees
    st.set_selection(np.array([True, False, True, False]))
    st.put_spectrum(_aligned_entry_with_points("s2", [0.4], [5.2]))
    with pytest.raises(ValueError):
        FitService(st).selected_pointcloud()


# --- compute_search / record_result ----------------------------------------


# Wide bounds so some scale is always feasible for the synthetic database (the
# breakpoint search scales each entry's params; narrow bounds can reject every
# scale, which is a real "infeasible" error, not what these tests probe).
_WIDE = ((0.1, 50.0), (0.01, 10.0), (0.01, 10.0))


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

    result = svc.compute_search(plot=False)

    # compute_search must NOT write State (no result recorded, no extra bump).
    assert isinstance(result, SearchResult)
    assert len(result.params) == 3
    assert st.fit.params is None  # still no result on State
    assert st.version.get(FIT_VERSION_KEY) == fit_version_before


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
    result = svc.compute_search(plot=False)
    svc.record_result(result)
    assert st.fit.params == result.params
    assert st.fit.has_result
    assert st.version.get(FIT_VERSION_KEY) == before + 1


def test_compute_search_fast_fails_without_database():
    st = _state_with_points()
    with pytest.raises(ValueError, match="database"):
        FitService(st).compute_search()


def test_compute_search_fast_fails_without_points(tiny_database):
    db_path = tiny_database[0]
    st = FluxDepState()  # no spectra
    svc = FitService(st)
    svc.set_params(db_path, *_WIDE, TransitionDict({}), 0.0, 0.0)
    with pytest.raises(ValueError, match="selected points"):
        svc.compute_search()


# --- export_params ---------------------------------------------------------


def test_export_params_writes_json(tmp_path):
    st = _state_with_points()
    st.project.chip_name = "Q9"
    st.project.qub_name = "Q1"
    svc = FitService(st)
    st.set_fit_result((5.0, 1.2, 0.4), best_dist=0.01)

    out = str(tmp_path / "params.json")
    path = svc.export_params(out)
    assert path == out

    from zcu_tools.notebook.persistance import load_result

    result = load_result(out)
    fluxdep_fit = result.get("fluxdep_fit")
    assert fluxdep_fit is not None
    assert fluxdep_fit["params"] == {"EJ": 5.0, "EC": 1.2, "EL": 0.4}
    assert fluxdep_fit["flux_half"] == 0.0
    assert fluxdep_fit["flux_period"] == 1.0


def test_export_params_fast_fails_without_result():
    st = _state_with_points()
    with pytest.raises(ValueError, match="no fit result"):
        FitService(st).export_params("/tmp/x.json")


def test_export_params_fast_fails_without_aligned():
    st = FluxDepState()  # no aligned spectrum
    st.set_fit_result((5.0, 1.2, 0.4), best_dist=0.01)
    with pytest.raises(ValueError, match="aligned"):
        FitService(st).export_params("/tmp/x.json")


def test_default_params_path():
    assert default_params_path("result/Q9/Q1", "Q9", "Q1") == "result/Q9/Q1/params.json"


def test_default_params_path_falls_back_to_chip_qub():
    # empty result_dir → derive from chip/qubit (never a bare 'params.json',
    # which would make dump_result's makedirs(dirname='') fail)
    path = default_params_path("", "Q9", "Q1")
    assert path == os.path.join("result", "Q9", "Q1", "params.json")
    assert os.path.dirname(path)  # non-empty dirname (the bug was '')


def test_export_params_empty_result_dir(tmp_path, monkeypatch):
    # the reported crash: export with no result_dir → "[Errno 2] ... ''".
    # With the fix it writes to result/<chip>/<qub>/params.json (cwd-relative).
    monkeypatch.chdir(tmp_path)
    st = _state_with_points()
    st.project.chip_name = "Q9"
    st.project.qub_name = "Q1"
    st.set_fit_result((5.0, 1.2, 0.4), best_dist=0.01)
    path = FitService(st).export_params()  # no savepath, no result_dir
    assert path.endswith(os.path.join("result", "Q9", "Q1", "params.json"))
    assert os.path.isfile(path)
