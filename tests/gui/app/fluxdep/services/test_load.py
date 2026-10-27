"""Tests for LoadService — load an hdf5 spectrum into FluxDepState."""

from __future__ import annotations

import numpy as np
import pytest
from zcu_tools.datafile import LabberData
from zcu_tools.gui.app.fluxdep.services.load import (
    LoadService,
    transpose_spectrum_data,
)
from zcu_tools.gui.app.fluxdep.state import (
    AlignmentPublication,
    FluxDepState,
    spectrum_version_key,
)
from zcu_tools.gui.expected_error import (
    ExpectedError,
    FailedPreconditionError,
    InvalidInputError,
)


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
    assert entry.points_completed is False
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
    st.set_alignment(
        first,
        alignment=AlignmentPublication(flux_half=1.0, flux_int=2.0, flux_period=2.0),
        new_fluxs=st.spectrums[first].raw["dev_values"].copy(),
        new_point_fluxs=np.empty(0, dtype=np.float64),
    )

    # load again (same file → same basename would clash; use a second file)
    second = svc.load_spectrum(filepath, spec_type="TwoTone", inherit_from=first)
    # same basename → replaced entry, but it inherits first's (now aligned) values
    entry = st.spectrums[second]
    assert (entry.flux_half, entry.flux_int, entry.flux_period) == (1.0, 2.0, 2.0)
    # fluxs derived from inherited alignment: (dev - 1.0)/2.0 + 0.5
    np.testing.assert_allclose(entry.raw["fluxs"], (dev_values - 1.0) / 2.0 + 0.5)


def test_load_rejects_unaligned_unseeded_inheritance_without_publication(spectrum_hdf5):
    filepath, *_ = spectrum_hdf5
    st = FluxDepState()
    svc = LoadService(st)
    source = svc.load_spectrum(filepath, spec_type="OneTone")
    original = st.get_spectrum(source)
    versions = st.version.snapshot()

    with pytest.raises(FailedPreconditionError):
        svc.load_spectrum(filepath, spec_type="TwoTone", inherit_from=source)

    assert st.get_spectrum(source) is original
    assert st.version.snapshot() == versions


def test_load_accepts_seeded_unaligned_inheritance(spectrum_hdf5):
    filepath, *_ = spectrum_hdf5
    st = FluxDepState()
    svc = LoadService(st)
    source = svc.load_spectrum(filepath, spec_type="OneTone")
    st.set_alignment(
        source,
        alignment=AlignmentPublication(flux_half=1.0, flux_int=2.0, flux_period=2.0),
        new_fluxs=st.get_spectrum(source).raw["dev_values"].copy(),
        new_point_fluxs=np.empty(0, dtype=np.float64),
    )
    seeded = svc.load_spectrum(filepath, spec_type="TwoTone", inherit_from=source)
    assert not st.get_spectrum(seeded).aligned
    inherited = svc.load_spectrum(filepath, spec_type="OneTone", inherit_from=seeded)
    entry = st.get_spectrum(inherited)
    assert entry.alignment_seeded
    assert (entry.flux_half, entry.flux_int, entry.flux_period) == (1.0, 2.0, 2.0)


def test_load_spectrum_inherit_from_unknown_raises(spectrum_hdf5):
    filepath, *_ = spectrum_hdf5
    st = FluxDepState()
    with pytest.raises(InvalidInputError) as caught:
        LoadService(st).load_spectrum(
            filepath, spec_type="OneTone", inherit_from="nope"
        )
    assert caught.value.reason_code == "unknown_spectrum"
    assert st.spectrums == {}
    assert st.version.snapshot() == {}


def test_non_2d_native_load_is_nominal_invalid_input(tmp_path):
    path = LabberData(
        ("Signal", "a.u.", np.ones(3, dtype=np.complex128)),
        [("Device", "native", np.arange(3))],
    ).save(str(tmp_path / "trace"))
    st = FluxDepState()
    with pytest.raises(InvalidInputError) as caught:
        LoadService(st).load_spectrum(path, "OneTone")
    assert caught.value.reason_code == "spectrum_not_2d"
    assert st.spectrums == {}
    assert st.version.snapshot() == {}


def test_native_load_io_failure_remains_unexpected(tmp_path):
    st = FluxDepState()
    with pytest.raises(FileNotFoundError) as caught:
        LoadService(st).load_spectrum(str(tmp_path / "missing.hdf5"), "OneTone")
    assert not isinstance(caught.value, ExpectedError)
    assert st.spectrums == {}
    assert st.version.snapshot() == {}


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
    filepath, flux, freqs_ghz, _signals = transposed_spectrum_hdf5
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
    filepath, _flux, _freqs_ghz, _signals = transposed_spectrum_hdf5
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
    st.set_alignment(
        first,
        alignment=AlignmentPublication(flux_half=1.0, flux_int=2.0, flux_period=2.0),
        new_fluxs=st.spectrums[first].raw["dev_values"].copy(),
        new_point_fluxs=np.empty(0, dtype=np.float64),
    )
    second = svc.load_spectrum(filepath, spec_type="TwoTone", inherit_from=first)
    # inheriting a spectrum's alignment marks the new one as seeded
    assert st.spectrums[second].alignment_seeded is True
    assert st.spectrums[second].flux_half == 1.0


# --- processed reload ------------------------------------------------------


def test_processed_projection_uses_affine_offset_without_counting_snapshot_twice(
    tmp_path,
):
    from zcu_tools.analysis.fluxdep.io import dump_spectrums
    from zcu_tools.analysis.fluxdep.models import SpectrumResult

    empty = np.empty(0, dtype=np.float64)
    result = SpectrumResult(
        offset=0.75,
        period=-2.0,
        device="bias",
        unit="A",
        device_snapshot={
            "bias": {"value": 99.0, "unit": "A"},
            "coil": {"value": 0.002, "unit": "A"},
        },
        spectrum={
            "dev_values": np.array([0.0, 1.0]),
            "fluxs": np.array([0.75, 0.25]),
            "freqs": np.array([4.0]),
            "signals": np.array([[1j], [2j]]),
        },
        points={"dev_values": empty, "fluxs": empty, "freqs": empty},
    )
    path = str(tmp_path / "processed.hdf5")
    dump_spectrums(path, {"scan": result})
    state = FluxDepState()
    LoadService(state).load_processed_spectrums(path)
    entry = state.get_spectrum("scan")
    assert (entry.flux_half, entry.flux_int, entry.flux_period) == (0.5, 1.5, -2.0)
    assert (entry.device, entry.unit) == ("bias", "A")
    assert entry.device_snapshot == result.get("device_snapshot")
    np.testing.assert_array_equal(entry.raw["fluxs"], [0.75, 0.25])


@pytest.mark.parametrize("empty", [False, True])
def test_load_processed_roundtrip(spectrum_hdf5, tmp_path, empty):
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
    PointsService(st).set_points(
        name,
        np.empty(0) if empty else np.array([0.0, 2.0]),
        np.empty(0) if empty else np.array([5.0, 5.5]),
    )
    out = str(tmp_path / "spectrums.hdf5")
    ExportService(st).export_spectrums(filepath=out)

    # restore into a fresh state
    st2 = FluxDepState()
    names = LoadService(st2).load_processed_spectrums(out)
    assert names == [name]
    entry = st2.spectrums[name]
    assert entry.aligned is True
    assert entry.points_completed is True
    assert entry.spec_type == "OneTone"  # type now persisted (was lost → TwoTone)
    assert entry.flux_period == 2.0
    np.testing.assert_allclose(entry.points["freqs"], [] if empty else [5.0, 5.5])
    assert entry.point_count == (0 if empty else 2)
    LoadService(st2).load_spectrum(filepath, spec_type="OneTone")
    reloaded = st2.spectrums[name]
    assert not reloaded.points_completed and not reloaded.aligned
    assert reloaded.point_count == 0
