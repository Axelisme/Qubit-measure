"""Tests for ExportService — write spectrums.hdf5 and round-trip via load_spectrums."""

from __future__ import annotations

import os

import numpy as np
import pytest
from zcu_tools.fluxdep_gui.services.alignment import AlignmentService, PointsService
from zcu_tools.fluxdep_gui.services.export import ExportService
from zcu_tools.fluxdep_gui.services.load import LoadService
from zcu_tools.fluxdep_gui.state import FluxDepState, ProjectInfo
from zcu_tools.notebook.persistance import load_spectrums


def test_export_empty_raises():
    st = FluxDepState()
    with pytest.raises(ValueError):
        ExportService(st).export_spectrums(filepath="/tmp/should_not_write.hdf5")


def test_export_roundtrips_via_load_spectrums(spectrum_hdf5, tmp_path):
    filepath, *_ = spectrum_hdf5
    st = FluxDepState()
    name = LoadService(st).load_spectrum(filepath, spec_type="OneTone")
    AlignmentService(st).set_alignment(name, flux_half=0.0, flux_int=1.0)
    PointsService(st).set_points(name, np.array([0.0, 2.0]), np.array([5.0, 5.5]))

    out = str(tmp_path / "spectrums.hdf5")
    resolved = ExportService(st).export_spectrums(filepath=out)
    assert resolved == out
    assert os.path.exists(out)

    loaded = load_spectrums(out)
    assert name in loaded
    result = loaded[name]
    # NOTE: dump_spectrums/load_spectrums do NOT persist the "type" field (a
    # NotRequired key in SpectrumResult) — spec_type is lost on round-trip. This
    # is existing persistance behaviour, recorded as a known v1 limitation.
    assert result["flux_half"] == 0.0
    assert result["flux_period"] == 2.0
    # points round-trip
    np.testing.assert_allclose(result["points"]["freqs"], [5.0, 5.5])
    # spectrum signals round-trip (shape preserved)
    assert (
        result["spectrum"]["signals"].shape == st.spectrums[name].raw["signals"].shape
    )


def test_default_export_path_under_result_dir():
    from zcu_tools.fluxdep_gui.services.export import default_export_path

    assert default_export_path(os.path.join("result", "Q5_2D", "Q1")) == os.path.join(
        "result", "Q5_2D", "Q1", "data", "fluxdep", "spectrums.hdf5"
    )


def test_export_service_default_path_from_project():
    # ProjectInfo derives result_dir from chip/qubit in __post_init__.
    st = FluxDepState(ProjectInfo(chip_name="Q5_2D", qub_name="Q1"))
    assert ExportService(st).default_path() == os.path.join(
        "result", "Q5_2D", "Q1", "data", "fluxdep", "spectrums.hdf5"
    )
