"""Tests for LoadService — load an hdf5 spectrum into FluxDepState."""

from __future__ import annotations

import numpy as np
import pytest
from zcu_tools.fluxdep_gui.services.load import LoadService
from zcu_tools.fluxdep_gui.state import FluxDepState, spectrum_version_key


def test_load_spectrum_populates_state(spectrum_hdf5):
    filepath, dev_values, freqs_ghz, _signals = spectrum_hdf5
    st = FluxDepState()
    svc = LoadService(st)

    name = svc.load_spectrum(filepath, spec_type="OneTone")

    assert name == "Q1_flux_1.hdf5"
    assert name in st.spectrums
    entry = st.spectrums[name]
    assert entry.spec_type == "OneTone"
    # device values + freqs round-trip (freqs come back in GHz, already increasing)
    np.testing.assert_allclose(entry.raw["dev_values"], dev_values)
    np.testing.assert_allclose(entry.raw["freqs"], freqs_ghz)
    assert entry.raw["signals"].shape == (len(dev_values), len(freqs_ghz))
    # newly loaded: not aligned, no points yet
    assert entry.aligned is False
    assert entry.points_selected is False
    assert entry.points["freqs"].size == 0
    # version keys bumped (per-spectrum + set)
    assert st.version.get(spectrum_version_key(name)) == 1


def test_load_spectrum_default_alignment_identity(spectrum_hdf5):
    filepath, dev_values, _, _ = spectrum_hdf5
    st = FluxDepState()
    name = LoadService(st).load_spectrum(filepath, spec_type="OneTone")
    entry = st.spectrums[name]
    # identity default: flux_half=0, period=1 → fluxs = dev_values + 0.5
    assert (entry.flux_half, entry.flux_int, entry.flux_period) == (0.0, 0.0, 1.0)
    np.testing.assert_allclose(entry.raw["fluxs"], dev_values + 0.5)


def test_load_spectrum_inherits_alignment(spectrum_hdf5):
    filepath, dev_values, _, _ = spectrum_hdf5
    st = FluxDepState()
    svc = LoadService(st)

    first = svc.load_spectrum(filepath, spec_type="OneTone")
    st.set_alignment(first, flux_half=1.0, flux_int=2.0, flux_period=2.0)

    # load again (same file → same basename would clash; use a second file)
    second = svc.load_spectrum(filepath, spec_type="TwoTone", inherit_from=first)
    # same basename → replaced entry, but it inherits first's (now aligned) values
    entry = st.spectrums[second]
    assert (entry.flux_half, entry.flux_int, entry.flux_period) == (1.0, 2.0, 2.0)
    # fluxs derived from inherited alignment: (dev - 1.0)/2.0 + 0.5
    np.testing.assert_allclose(entry.raw["fluxs"], (dev_values - 1.0) / 2.0 + 0.5)


def test_load_spectrum_inherit_from_unknown_raises(spectrum_hdf5):
    filepath, *_ = spectrum_hdf5
    st = FluxDepState()
    with pytest.raises(KeyError):
        LoadService(st).load_spectrum(
            filepath, spec_type="OneTone", inherit_from="nope"
        )
