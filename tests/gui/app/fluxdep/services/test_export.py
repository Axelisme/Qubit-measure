"""Tests for ExportService — write spectrums.hdf5 and round-trip via load_spectrums."""

from __future__ import annotations

import os

import numpy as np
import pytest
from zcu_tools.analysis.fluxdep.io import load_spectrums
from zcu_tools.gui.app.fluxdep.services.alignment import AlignmentService, PointsService
from zcu_tools.gui.app.fluxdep.services.export import ExportService
from zcu_tools.gui.app.fluxdep.services.load import LoadService
from zcu_tools.gui.app.fluxdep.state import FluxDepState
from zcu_tools.gui.expected_error import FailedPreconditionError


def test_export_empty_raises(tmp_path):
    st = FluxDepState()
    path = tmp_path / "empty.hdf5"
    with pytest.raises(FailedPreconditionError) as caught:
        ExportService(st).export_spectrums(filepath=str(path))
    assert caught.value.reason_code == "no_spectrums"
    assert not path.exists()
    assert st.version.snapshot() == {}


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
    assert result["offset"] == 0.5
    assert result["period"] == 2.0
    assert result["device"] == "unnamed_device"
    assert result["unit"] == "native"
    # points round-trip
    np.testing.assert_allclose(result["points"]["freqs"], [5.0, 5.5])
    # spectrum signals round-trip (shape preserved)
    assert (
        result["spectrum"]["signals"].shape == st.spectrums[name].raw["signals"].shape
    )
