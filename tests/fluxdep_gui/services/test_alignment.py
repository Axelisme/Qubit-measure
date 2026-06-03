"""Tests for AlignmentService + PointsService."""

from __future__ import annotations

import numpy as np
import pytest
from zcu_tools.fluxdep_gui.services.alignment import AlignmentService, PointsService
from zcu_tools.fluxdep_gui.services.load import LoadService
from zcu_tools.fluxdep_gui.state import FluxDepState
from zcu_tools.simulate import value2flux


def _loaded(spectrum_hdf5) -> tuple[FluxDepState, str]:
    filepath, *_ = spectrum_hdf5
    st = FluxDepState()
    name = LoadService(st).load_spectrum(filepath, spec_type="OneTone")
    return st, name


def test_set_alignment_derives_period_and_remaps_fluxs(spectrum_hdf5):
    st, name = _loaded(spectrum_hdf5)
    AlignmentService(st).set_alignment(name, flux_half=1.0, flux_int=2.0)
    entry = st.spectrums[name]
    assert entry.aligned is True
    assert entry.flux_period == 2.0  # 2*|2-1|
    expected = value2flux(entry.raw["dev_values"], 1.0, 2.0)
    np.testing.assert_allclose(entry.raw["fluxs"], expected)


def test_set_alignment_zero_period_raises(spectrum_hdf5):
    st, name = _loaded(spectrum_hdf5)
    with pytest.raises(ValueError):
        AlignmentService(st).set_alignment(name, flux_half=1.0, flux_int=1.0)


def test_set_points_sorts_and_derives_fluxs(spectrum_hdf5):
    st, name = _loaded(spectrum_hdf5)
    AlignmentService(st).set_alignment(name, flux_half=0.0, flux_int=1.0)
    # unsorted device values
    devs = np.array([3.0, -1.0, 2.0])
    freqs = np.array([5.3, 5.1, 5.2])
    PointsService(st).set_points(name, devs, freqs)
    pts = st.spectrums[name].points
    # sorted by device value
    np.testing.assert_allclose(pts["dev_values"], [-1.0, 2.0, 3.0])
    np.testing.assert_allclose(pts["freqs"], [5.1, 5.2, 5.3])
    # fluxs derived from alignment (period=2, half=0)
    expected = value2flux(np.array([-1.0, 2.0, 3.0]), 0.0, 2.0)
    np.testing.assert_allclose(pts["fluxs"], expected)
    assert st.spectrums[name].points_selected is True
