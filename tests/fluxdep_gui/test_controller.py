"""Tests for the fluxdep-gui Controller façade — pipeline actions emit events."""

from __future__ import annotations

import numpy as np
from zcu_tools.fluxdep_gui.controller import Controller
from zcu_tools.fluxdep_gui.event_bus import (
    ActiveSpectrumChangedPayload,
    Payload,
    ProjectChangedPayload,
    SelectionChangedPayload,
    SpectrumAddedPayload,
    SpectrumChangedPayload,
    SpectrumRemovedPayload,
)
from zcu_tools.fluxdep_gui.state import FluxDepState, ProjectInfo


def _ctrl() -> Controller:
    return Controller(FluxDepState())


def _record(ctrl: Controller, payload_type: type[Payload]) -> list:
    seen: list = []
    ctrl.bus.subscribe(payload_type, lambda p: seen.append(p))
    return seen


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
