"""Tests for LoadService — load an hdf5 spectrum into FluxDepState."""

from __future__ import annotations

import numpy as np
import pytest
from zcu_tools.gui.app.fluxdep.services.load import (
    LoadService,
    transpose_spectrum_data,
)
from zcu_tools.gui.app.fluxdep.state import FluxDepState, spectrum_version_key


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


# --- transpose -------------------------------------------------------------


def test_transpose_spectrum_data_swaps_axes_and_signal():
    z = np.arange(6, dtype=np.complex128).reshape(2, 3)  # (x=2, y=3)
    x = np.array([0.0, 1.0])
    y = np.array([10.0, 20.0, 30.0])
    z2, x2, y2 = transpose_spectrum_data(z, x, y)
    np.testing.assert_array_equal(z2, z.T)  # (3, 2)
    np.testing.assert_array_equal(x2, y)  # x ← old y
    np.testing.assert_array_equal(y2, x)  # y ← old x


def test_load_transpose_recovers_canonical_axes(transposed_spectrum_hdf5):
    """A legacy x=freq/y=flux file loaded with transpose_axes=True must come back
    with dev_values=flux and freqs in GHz."""
    filepath, flux, freqs_ghz, signals = transposed_spectrum_hdf5
    st = FluxDepState()
    name = LoadService(st).load_spectrum(
        filepath, spec_type="OneTone", transpose_axes=True
    )
    entry = st.spectrums[name]
    np.testing.assert_allclose(entry.raw["dev_values"], flux)
    np.testing.assert_allclose(entry.raw["freqs"], freqs_ghz)
    assert entry.raw["signals"].shape == (len(flux), len(freqs_ghz))


def test_load_without_transpose_keeps_legacy_axes_wrong(transposed_spectrum_hdf5):
    """Sanity: loading the legacy file WITHOUT transpose mis-reads the axes
    (freqs ≈ 0 after the Hz→GHz scaling of what is really the flux axis)."""
    filepath, flux, freqs_ghz, _signals = transposed_spectrum_hdf5
    st = FluxDepState()
    name = LoadService(st).load_spectrum(filepath, spec_type="OneTone")
    entry = st.spectrums[name]
    # dev_values would be the freq axis (Hz, ~5e9), not the flux range
    assert entry.raw["dev_values"].max() > 1e6  # clearly not the flux range


# --- inherit seeding -------------------------------------------------------


def test_inherited_load_marks_alignment_seeded(spectrum_hdf5):
    filepath, *_ = spectrum_hdf5
    st = FluxDepState()
    svc = LoadService(st)
    first = svc.load_spectrum(filepath, spec_type="OneTone")
    assert st.spectrums[first].alignment_seeded is False  # fresh load
    st.set_alignment(first, flux_half=1.0, flux_int=2.0, flux_period=2.0)
    second = svc.load_spectrum(filepath, spec_type="TwoTone", inherit_from=first)
    # inheriting a spectrum's alignment marks the new one as seeded
    assert st.spectrums[second].alignment_seeded is True
    assert st.spectrums[second].flux_half == 1.0


# --- processed reload ------------------------------------------------------


def test_load_processed_roundtrip(spectrum_hdf5, tmp_path):
    from zcu_tools.gui.app.fluxdep.services.alignment import (
        AlignmentService,
        PointsService,
    )
    from zcu_tools.gui.app.fluxdep.services.export import ExportService

    # build + export a processed spectrum
    filepath, *_ = spectrum_hdf5
    st = FluxDepState()
    name = LoadService(st).load_spectrum(filepath, spec_type="OneTone")
    AlignmentService(st).set_alignment(name, flux_half=0.0, flux_int=1.0)
    PointsService(st).set_points(name, np.array([0.0, 2.0]), np.array([5.0, 5.5]))
    out = str(tmp_path / "spectrums.hdf5")
    ExportService(st).export_spectrums(filepath=out)

    # restore into a fresh state
    st2 = FluxDepState()
    names = LoadService(st2).load_processed_spectrums(out)
    assert names == [name]
    entry = st2.spectrums[name]
    assert entry.aligned is True
    assert entry.points_selected is True
    assert entry.spec_type == "OneTone"  # type now persisted (was lost → TwoTone)
    assert entry.flux_period == 2.0
    np.testing.assert_allclose(entry.points["freqs"], [5.0, 5.5])
