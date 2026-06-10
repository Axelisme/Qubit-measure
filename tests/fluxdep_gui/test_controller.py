"""Tests for the fluxdep-gui Controller façade — pipeline actions emit events."""

from __future__ import annotations

import numpy as np
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
from zcu_tools.gui.project import ProjectInfo


def _ctrl() -> Controller:
    return Controller(FluxDepState())


def _record(ctrl: Controller, payload_type: type[Payload]) -> list:
    seen: list = []
    ctrl.bus.subscribe(payload_type, lambda p: seen.append(p))
    return seen


def test_get_project_root_returns_injected_root():
    # The entry script injects the repo root so default paths anchor there, not
    # cwd (the .bat launcher cd's into script/).
    ctrl = Controller(FluxDepState(), project_root="/repo")
    assert ctrl.get_project_root() == "/repo"


def test_get_project_root_falls_back_to_cwd_when_not_injected():
    import os

    ctrl = Controller(FluxDepState())
    assert ctrl.get_project_root() == os.getcwd()


def test_setup_project_mutates_and_emits():
    ctrl = _ctrl()
    seen = _record(ctrl, ProjectChangedPayload)
    ctrl.setup_project(ProjectInfo(chip_name="Q5_2D", qub_name="Q1"))
    assert ctrl.state.project.chip_name == "Q5_2D"
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
    assert ctrl.state.spectrums[name].points_selected is True
    assert len(seen) == 1


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
    assert ctrl.state.selection.selected is not None
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

    from zcu_tools.notebook.persistance import load_spectrums

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
    from zcu_tools.gui.app.fluxdep.state import SpectrumEntry
    from zcu_tools.notebook.persistance import PointsData, SpectrumData

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
            points_selected=True,
        )
    )


def test_set_fit_params_emits_fit_changed(tmp_path):
    from zcu_tools.gui.app.fluxdep.event_bus import FitChangedPayload
    from zcu_tools.notebook.persistance import TransitionDict

    ctrl = _ctrl()
    seen = _record(ctrl, FitChangedPayload)
    ctrl.set_fit_params(
        _fit_db_file(tmp_path),
        (0.1, 50.0),
        (0.01, 10.0),
        (0.01, 10.0),
        TransitionDict({"transitions": [(0, 1)]}),
        0.0,
        0.0,
    )
    assert len(seen) == 1
    assert ctrl.state.fit.database_path != ""


def test_compute_search_does_not_emit_or_record(tmp_path):
    from zcu_tools.gui.app.fluxdep.event_bus import FitChangedPayload
    from zcu_tools.notebook.persistance import TransitionDict

    ctrl = _ctrl()
    _seed_aligned_points(ctrl)
    ctrl.set_fit_params(
        _fit_db_file(tmp_path),
        (0.1, 50.0),
        (0.01, 10.0),
        (0.01, 10.0),
        TransitionDict({"transitions": [(0, 1), (0, 2)]}),
        0.0,
        0.0,
    )
    seen = _record(ctrl, FitChangedPayload)
    result = ctrl.compute_search(plot=False)
    # compute_search is pure: no event, no recorded result
    assert seen == []
    assert ctrl.state.fit.params is None
    # recording it emits + writes
    ctrl.record_search_result(result)
    assert len(seen) == 1
    assert ctrl.state.fit.has_result


def test_search_database_records_and_emits(tmp_path):
    from zcu_tools.gui.app.fluxdep.event_bus import FitChangedPayload
    from zcu_tools.notebook.persistance import TransitionDict

    ctrl = _ctrl()
    _seed_aligned_points(ctrl)
    ctrl.set_fit_params(
        _fit_db_file(tmp_path),
        (0.1, 50.0),
        (0.01, 10.0),
        (0.01, 10.0),
        TransitionDict({"transitions": [(0, 1), (0, 2)]}),
        0.0,
        0.0,
    )
    seen = _record(ctrl, FitChangedPayload)
    result = ctrl.search_database(plot=False)
    assert len(result.params) == 3
    assert ctrl.state.fit.has_result
    assert len(seen) == 1
