"""Tests for the fluxdep-gui Controller façade — pipeline actions emit events."""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest
from zcu_tools.analysis.fluxdep.search import ParamBounds
from zcu_tools.gui.app.fluxdep.controller import Controller
from zcu_tools.gui.app.fluxdep.event_bus import (
    ActiveSpectrumChangedPayload,
    Payload,
    ProjectChangedPayload,
    SelectionChangedPayload,
    SpectrumAddedPayload,
    SpectrumChangedPayload,
    SpectrumRemovedPayload,
)
from zcu_tools.gui.app.fluxdep.state import FluxDepState
from zcu_tools.gui.expected_error import FailedPreconditionError, InvalidInputError


def _ctrl() -> Controller:
    return Controller(FluxDepState())


def _record[P: Payload](ctrl: Controller, payload_type: type[P]) -> list[P]:
    seen: list[P] = []
    ctrl.bus.subscribe(payload_type, lambda p: seen.append(p))
    return seen


def test_get_project_root_returns_injected_root():
    # The entry script injects the repo root so default paths anchor there, not
    # cwd (the .bat launcher cd's into scripts/).
    ctrl = Controller(FluxDepState(), project_root="/repo")
    assert ctrl.get_project_root() == "/repo"


def test_get_project_root_falls_back_to_cwd_when_not_injected():
    import os

    ctrl = Controller(FluxDepState())
    assert ctrl.get_project_root() == os.getcwd()


def test_setup_project_mutates_and_emits(fluxonium_entry):
    ctrl = _ctrl()
    seen = _record(ctrl, ProjectChangedPayload)
    ctrl.setup_project(str(fluxonium_entry.result_path.parent.parent), "synthetic")
    assert ctrl.state.project is not None
    assert ctrl.state.project.component == "Q2"
    assert len(seen) == 1


def test_load_spectrum_emits_added(spectrum_hdf5):
    filepath, *_ = spectrum_hdf5
    ctrl = _ctrl()
    seen = _record(ctrl, SpectrumAddedPayload)
    name = ctrl.load_spectrum(filepath, spec_type="OneTone")
    assert name in ctrl.state.spectrums
    assert [p.name for p in seen] == [name]


def test_set_alignment_emits_changed(spectrum_hdf5):
    filepath, *_ = spectrum_hdf5
    ctrl = _ctrl()
    name = ctrl.load_spectrum(filepath, spec_type="OneTone")
    seen = _record(ctrl, SpectrumChangedPayload)
    ctrl.set_alignment(name, flux_half=0.0, flux_int=1.0)
    assert ctrl.state.spectrums[name].aligned is True
    assert [p.name for p in seen] == [name]


def test_set_points_emits_changed(spectrum_hdf5):
    filepath, *_ = spectrum_hdf5
    ctrl = _ctrl()
    name = ctrl.load_spectrum(filepath, spec_type="OneTone")
    ctrl.set_alignment(name, flux_half=0.0, flux_int=1.0)
    seen = _record(ctrl, SpectrumChangedPayload)
    ctrl.set_points(name, np.array([0.0, 1.0]), np.array([5.0, 5.1]))
    assert ctrl.state.spectrums[name].points_completed is True
    assert len(seen) == 1


@pytest.mark.parametrize("empty", [False, True])
def test_reset_and_realignment_preserve_native_points_and_publish_once(
    spectrum_hdf5, empty
):
    from zcu_tools.gui.app.fluxdep.state import spectrum_version_key

    filepath, *_ = spectrum_hdf5
    ctrl = _ctrl()
    name = ctrl.load_spectrum(filepath, "OneTone")
    ctrl.set_alignment(name, 0.0, 1.0)
    devs = np.array([], dtype=np.float64) if empty else np.array([1.0, 0.0])
    freqs = np.array([], dtype=np.float64) if empty else np.array([5.1, 5.0])
    ctrl.set_points(name, devs, freqs)
    seen = _record(ctrl, SpectrumChangedPayload)
    before = ctrl.state.version.get(spectrum_version_key(name))
    ctrl.reset_alignment(name)
    entry = ctrl.state.spectrums[name]
    assert not entry.aligned and entry.points_completed and entry.alignment_seeded
    assert entry.point_count == devs.size
    assert ctrl.state.version.get(spectrum_version_key(name)) == before + 1
    assert len(seen) == 1

    with pytest.raises(FailedPreconditionError) as caught:
        ctrl.reset_points(name)
    assert caught.value.reason_code == "spectrum_not_aligned"
    with pytest.raises(InvalidInputError) as caught:
        ctrl.reset_points("absent")
    assert caught.value.reason_code == "unknown_spectrum"
    with pytest.raises(InvalidInputError) as caught:
        ctrl.reset_alignment("absent")
    assert caught.value.reason_code == "unknown_spectrum"
    assert ctrl.state.spectrums[name] is entry
    assert len(seen) == 1
    assert ctrl.state.version.get(spectrum_version_key(name)) == before + 1

    ctrl.set_alignment(name, 0.25, 0.75)
    entry = ctrl.state.spectrums[name]
    assert entry.aligned and entry.points_completed
    np.testing.assert_array_equal(entry.points["dev_values"], np.sort(devs))
    np.testing.assert_array_equal(entry.points["freqs"], np.sort(freqs))
    np.testing.assert_allclose(entry.points["fluxs"], np.sort(devs) + 0.25)
    np.testing.assert_allclose(entry.raw["fluxs"], entry.raw["dev_values"] + 0.25)
    assert len(seen) == 2
    assert ctrl.state.version.get(spectrum_version_key(name)) == before + 2

    ctrl.reset_points(name)
    entry = ctrl.state.spectrums[name]
    assert entry.aligned and entry.alignment_seeded and not entry.points_completed
    assert entry.point_count == 0
    assert len(seen) == 3
    assert ctrl.state.version.get(spectrum_version_key(name)) == before + 3
    ctrl.set_points(name, np.empty(0), np.empty(0))
    assert ctrl.state.spectrums[name].points_completed
    assert len(seen) == 4


def test_set_active_emits(spectrum_hdf5):
    filepath, *_ = spectrum_hdf5
    ctrl = _ctrl()
    name = ctrl.load_spectrum(filepath, spec_type="OneTone")
    seen = _record(ctrl, ActiveSpectrumChangedPayload)
    ctrl.set_active_spectrum(name)
    assert ctrl.state.active_spectrum == name
    assert [p.name for p in seen] == [name]


def test_remove_spectrum_emits(spectrum_hdf5):
    filepath, *_ = spectrum_hdf5
    ctrl = _ctrl()
    name = ctrl.load_spectrum(filepath, spec_type="OneTone")
    seen = _record(ctrl, SpectrumRemovedPayload)
    ctrl.remove_spectrum(name)
    assert name not in ctrl.state.spectrums
    assert [p.name for p in seen] == [name]


def test_selection_pipeline_emits(spectrum_hdf5):
    filepath, *_ = spectrum_hdf5
    ctrl = _ctrl()
    name = ctrl.load_spectrum(filepath, spec_type="OneTone")
    ctrl.set_alignment(name, flux_half=0.0, flux_int=1.0)
    ctrl.set_points(name, np.array([0.0, 1.0, 2.0]), np.array([5.0, 5.1, 5.2]))
    seen = _record(ctrl, SelectionChangedPayload)
    ctrl.set_selection(np.array([True, False, True]))
    assert ctrl.state.selection_mask() is not None
    assert len(seen) == 1


def test_full_pipeline_export_roundtrip(spectrum_hdf5, tmp_path):
    """End-to-end: load → align → points → export through the Controller."""
    filepath, *_ = spectrum_hdf5
    ctrl = _ctrl()
    name = ctrl.load_spectrum(filepath, spec_type="OneTone")
    ctrl.set_alignment(name, flux_half=0.0, flux_int=1.0)
    ctrl.set_points(name, np.array([0.0, 2.0]), np.array([5.0, 5.5]))
    out = str(tmp_path / "spectrums.hdf5")
    resolved = ctrl.export_spectrums(filepath=out)

    from zcu_tools.analysis.fluxdep.io import load_spectrums

    loaded = load_spectrums(resolved)
    assert name in loaded
    assert loaded[name]["flux_period"] == 2.0


# --- database-search fit (v2) ----------------------------------------------


def _fit_db_file(tmp_path) -> str:
    import h5py

    M, L = 21, 4
    fluxs = np.linspace(0.0, 0.5, M).astype(np.float64)
    params = np.array([[3.0, 1.0, 0.5], [5.0, 1.2, 0.4]], dtype=np.float64)
    energies = np.zeros((len(params), M, L), dtype=np.float64)
    for n, (EJ, EC, EL) in enumerate(params):
        for lvl in range(L):
            energies[n, :, lvl] = lvl * (EC + EL) + EJ * np.cos(2 * np.pi * fluxs) * 0.1
    path = tmp_path / "db.h5"
    with h5py.File(path, "w") as f:
        f.create_dataset("fluxs", data=fluxs)
        f.create_dataset("params", data=params)
        f.create_dataset("energies", data=energies)
    return str(path)


def _seed_aligned_points(ctrl: Controller) -> None:
    from zcu_tools.analysis.fluxdep.models import PointsData
    from zcu_tools.analysis.spectrum import SpectrumData
    from zcu_tools.gui.app.fluxdep.state import SpectrumEntry

    fluxs = np.array([0.0, 0.1, 0.2, 0.3])
    raw = SpectrumData(
        dev_values=fluxs.copy(),
        fluxs=fluxs.copy(),
        freqs=np.linspace(4.0, 6.0, 5).astype(np.float64),
        signals=np.zeros((4, 5), dtype=np.complex128),
    )
    points = PointsData(
        dev_values=fluxs.copy(),
        fluxs=fluxs.copy(),
        freqs=np.array([5.0, 5.1, 5.2, 5.3]),
    )
    ctrl.state.put_spectrum(
        SpectrumEntry(
            name="s1",
            spec_type="TwoTone",
            raw=raw,
            points=points,
            flux_half=0.0,
            flux_int=0.5,
            flux_period=1.0,
            aligned=True,
            points_completed=True,
        )
    )


def test_set_fit_params_emits_fit_changed(tmp_path):
    from zcu_tools.analysis.fluxdep.models import TransitionDict
    from zcu_tools.gui.app.fluxdep.event_bus import FitChangedPayload

    ctrl = _ctrl()
    seen = _record(ctrl, FitChangedPayload)
    ctrl.set_fit_params(
        _fit_db_file(tmp_path),
        ParamBounds(EJ=(0.1, 50.0), EC=(0.01, 10.0), EL=(0.01, 10.0)),
        TransitionDict({"transitions": [(0, 1)]}),
        0.0,
        0.0,
    )
    assert len(seen) == 1
    assert ctrl.state.fit.search_db != ""


def test_compute_search_does_not_emit_or_record(tmp_path):
    from zcu_tools.analysis.fluxdep.models import TransitionDict
    from zcu_tools.gui.app.fluxdep.event_bus import FitChangedPayload

    ctrl = _ctrl()
    _seed_aligned_points(ctrl)
    ctrl.set_fit_params(
        _fit_db_file(tmp_path),
        ParamBounds(EJ=(0.1, 50.0), EC=(0.01, 10.0), EL=(0.01, 10.0)),
        TransitionDict({"transitions": [(0, 1), (0, 2)]}),
        0.0,
        0.0,
    )
    seen = _record(ctrl, FitChangedPayload)
    inputs = ctrl.capture_search()
    result = ctrl.compute_search(inputs)
    # compute_search is pure: no event, no recorded result
    assert seen == []
    assert ctrl.state.fit.params is None
    # recording it emits + writes
    ctrl.record_search_result(result, inputs)
    assert len(seen) == 1
    assert ctrl.state.fit.has_result


def test_search_database_records_and_emits(tmp_path):
    from zcu_tools.analysis.fluxdep.models import TransitionDict
    from zcu_tools.gui.app.fluxdep.event_bus import FitChangedPayload

    ctrl = _ctrl()
    _seed_aligned_points(ctrl)
    ctrl.set_fit_params(
        _fit_db_file(tmp_path),
        ParamBounds(EJ=(0.1, 50.0), EC=(0.01, 10.0), EL=(0.01, 10.0)),
        TransitionDict({"transitions": [(0, 1), (0, 2)]}),
        0.0,
        0.0,
    )
    seen = _record(ctrl, FitChangedPayload)
    result = ctrl.search_database()
    assert len(result.params) == 3
    assert ctrl.state.fit.has_result
    assert len(seen) == 1


# --- spectrum_rows: one row of published facts per spectrum ----------------


def _stages(ctrl: Controller) -> list[tuple[str, str, str, int]]:
    return [
        (row.name, row.spec_type, row.stage, row.point_count)
        for row in ctrl.spectrum_rows()
    ]


def test_spectrum_rows_report_published_stage_and_point_count(spectrum_hdf5):
    filepath, *_ = spectrum_hdf5
    ctrl = _ctrl()
    name = ctrl.load_spectrum(filepath, spec_type="OneTone")
    assert _stages(ctrl) == [(name, "OneTone", "loaded", 0)]
    ctrl.set_alignment(name, flux_half=0.0, flux_int=1.0)
    assert _stages(ctrl) == [(name, "OneTone", "aligned", 0)]
    ctrl.set_points(name, np.array([0.0, 1.0]), np.array([5.0, 5.1]))
    assert _stages(ctrl) == [(name, "OneTone", "sampled", 2)]
    # Reopened alignment keeps the points, but the published stage is loaded.
    ctrl.reset_alignment(name)
    assert _stages(ctrl) == [(name, "OneTone", "loaded", 2)]


def test_spectrum_rows_count_zero_point_completion_as_sampled(spectrum_hdf5):
    filepath, *_ = spectrum_hdf5
    ctrl = _ctrl()
    name = ctrl.load_spectrum(filepath, spec_type="OneTone")
    ctrl.set_alignment(name, flux_half=0.0, flux_int=1.0)
    ctrl.set_points(name, np.empty(0), np.empty(0))
    assert _stages(ctrl) == [(name, "OneTone", "sampled", 0)]


def test_spectrum_rows_count_only_published_filter_exclusions(cross_controller):
    ctrl = cross_controller

    def excluded() -> list[tuple[str, int]]:
        return [(row.name, row.filter_excluded_count) for row in ctrl.spectrum_rows()]

    # No published mask counts as zero exclusions.
    assert excluded() == [("a", 0), ("empty", 0), ("b", 0)]
    # Joint cloud order is a (3 points), empty, b (1 point).
    ctrl.set_selection(np.array([True, False, True, False]))
    assert excluded() == [("a", 1), ("empty", 0), ("b", 1)]
    # An open Filter draft does not change the published count.
    ctrl.interactive.begin_cross_selection()
    assert excluded() == [("a", 1), ("empty", 0), ("b", 1)]
    ctrl.interactive.cancel_selection()
    # min_distance downsampling is not a mask exclusion.
    ctrl.set_selection(np.array([True, True, True, True]), min_distance=0.9)
    assert excluded() == [("a", 0), ("empty", 0), ("b", 0)]


def test_spectrum_rows_report_pending_draft_kind(cross_controller):
    ctrl = cross_controller

    def pending() -> list[tuple[str, str | None]]:
        return [(row.name, row.pending) for row in ctrl.spectrum_rows()]

    assert pending() == [("a", None), ("empty", None), ("b", None)]
    ctrl.interactive.begin_line_pick("b")
    assert pending() == [("a", None), ("empty", None), ("b", "align")]
    ctrl.interactive.cancel("b")
    ctrl.interactive.begin_sample_pick("a")
    assert pending() == [("a", "sample"), ("empty", None), ("b", None)]
    ctrl.interactive.cancel("a")
    # The global Filter draft is not a per-spectrum pending kind.
    ctrl.interactive.begin_cross_selection()
    assert pending() == [("a", None), ("empty", None), ("b", None)]


def test_spectrum_rows_do_not_attribute_the_filter_draft_to_a_spectrum_named_selection(
    cross_controller,
):
    ctrl = cross_controller
    entry = ctrl.state.spectrums["b"]
    ctrl.remove_spectrum("b")
    ctrl.state.put_spectrum(replace(entry, name="selection"))
    ctrl.interactive.begin_cross_selection()
    assert [(row.name, row.pending) for row in ctrl.spectrum_rows()] == [
        ("a", None),
        ("empty", None),
        ("selection", None),
    ]
