"""Round-trip and fast-fail tests for labber_io.

Tests are grouped by public API surface:
  - save_labber_data / load_labber_data  (uniform-grid, 1-D / 2-D / 3-D)
  - metadata round-trip (comment, tags, project, user, timestamps)
  - save_labber_trace_data               (variable-length / ragged traces)
  - file-extension handling
  - LabberData.save / LabberData.load    (class-level wrappers)
  - error paths (fast-fail validation)
"""

from __future__ import annotations

import numpy as np
import pytest
from zcu_tools.utils.datasaver import (
    Axis,
    LabberData,
    load_labber_data,
    save_labber_data,
    save_labber_trace_data,
)

# ---------------------------------------------------------------------------
# 1-D round-trip
# ---------------------------------------------------------------------------


def test_1d_roundtrip(tmp_path):
    """1-D complex trace: z values, axis name/unit/values all survive the trip."""
    freq = np.linspace(4e9, 5e9, 51)
    z = np.exp(1j * np.linspace(0, np.pi, 51)) * 0.7

    path = str(tmp_path / "scan_1d")
    save_labber_data(path, z=("S21", "arb", z), axes=[("Frequency", "Hz", freq)])

    d = load_labber_data(path)

    assert np.allclose(d.z, z)
    assert d.axes[0].name == "Frequency"
    assert d.axes[0].unit == "Hz"
    assert np.allclose(d.axes[0].values, freq)


# ---------------------------------------------------------------------------
# 2-D round-trip
# ---------------------------------------------------------------------------


def test_2d_roundtrip(tmp_path):
    """2-D: shape (Ny, Nx), inner axis last; x/y alias and unpack all correct."""
    freq = np.linspace(4e9, 5e9, 31)  # x (inner)
    power = np.linspace(-30.0, 0.0, 7)  # y (outer)
    rng = np.random.default_rng(42)
    z2d = rng.standard_normal((len(power), len(freq))) + 1j * rng.standard_normal(
        (len(power), len(freq))
    )

    path = str(tmp_path / "scan_2d")
    save_labber_data(
        path,
        z=("S21", "arb", z2d),
        axes=[("Frequency", "Hz", freq), ("Power", "dBm", power)],
    )

    d = load_labber_data(path)

    assert d.z.shape == (len(power), len(freq))
    assert np.allclose(d.z, z2d)

    # axes in memory: inner first
    assert d.axes[0].name == "Frequency"
    assert d.axes[0].unit == "Hz"
    assert np.allclose(d.axes[0].values, freq)
    assert d.axes[1].name == "Power"
    assert d.axes[1].unit == "dBm"
    assert np.allclose(d.axes[1].values, power)

    # convenience aliases
    assert np.allclose(d.x, freq)
    assert np.allclose(d.y, power)

    # unpack
    z_out, x_out, y_out = d
    assert np.allclose(z_out, z2d)
    assert np.allclose(x_out, freq)
    assert np.allclose(y_out, power)


# ---------------------------------------------------------------------------
# 3-D round-trip
# ---------------------------------------------------------------------------


def test_3d_roundtrip(tmp_path):
    """3-D: shape (Nw, Ny, Nx) three axes, innermost last."""
    freq = np.linspace(4e9, 5e9, 11)  # x
    power = np.linspace(-30.0, 0.0, 5)  # y
    flux = np.linspace(0.0, 1.0, 3)  # w (outermost)
    rng = np.random.default_rng(7)
    z3d = rng.standard_normal(
        (len(flux), len(power), len(freq))
    ) + 1j * rng.standard_normal((len(flux), len(power), len(freq)))

    path = str(tmp_path / "scan_3d")
    save_labber_data(
        path,
        z=("S21", "", z3d),
        axes=[
            ("Frequency", "Hz", freq),
            ("Power", "dBm", power),
            ("Flux", "Phi0", flux),
        ],
    )

    d = load_labber_data(path)

    assert d.z.shape == (len(flux), len(power), len(freq))
    assert np.allclose(d.z, z3d)
    assert np.allclose(d.axes[0].values, freq)
    assert np.allclose(d.axes[1].values, power)
    assert np.allclose(d.axes[2].values, flux)
    assert d.axes[2].name == "Flux"
    assert d.axes[2].unit == "Phi0"


# ---------------------------------------------------------------------------
# Metadata round-trip
# ---------------------------------------------------------------------------


def test_metadata_roundtrip_str_tags(tmp_path):
    """Metadata fields survive: comment, tags (str input), project, user."""
    freq = np.linspace(4e9, 5e9, 11)
    z = np.ones(11, dtype=complex)

    path = str(tmp_path / "meta_str")
    save_labber_data(
        path,
        z=("S21", "", z),
        axes=[("Freq", "Hz", freq)],
        comment="my comment",
        tags="tagA",
        project="proj1",
        user="alice",
    )

    d = load_labber_data(path)
    assert d.comment == "my comment"
    assert d.tags == ["tagA"]
    assert d.project == "proj1"
    assert d.user == "alice"


def test_metadata_roundtrip_list_tags(tmp_path):
    """Tags given as a list of strings round-trip correctly."""
    freq = np.linspace(4e9, 5e9, 11)
    z = np.ones(11, dtype=complex)

    path = str(tmp_path / "meta_list")
    save_labber_data(
        path,
        z=("S21", "", z),
        axes=[("Freq", "Hz", freq)],
        tags=["tagA", "tagB"],
    )

    d = load_labber_data(path)
    assert set(d.tags) == {"tagA", "tagB"}


def test_metadata_timestamps(tmp_path):
    """Per-entry timestamps (epoch s) round-trip to allclose precision."""
    freq = np.linspace(4e9, 5e9, 11)  # x
    power = np.linspace(-30.0, 0.0, 5)  # y  (5 entries)
    z2d = np.ones((len(power), len(freq)), dtype=complex)

    # per-entry timestamps: one per outer entry
    ts = np.array([1_700_000_000.0 + i * 10.0 for i in range(len(power))])

    path = str(tmp_path / "meta_ts")
    save_labber_data(
        path,
        z=("S21", "", z2d),
        axes=[("Freq", "Hz", freq), ("Power", "dBm", power)],
        timestamps=ts,
    )

    d = load_labber_data(path)
    assert d.timestamps is not None
    assert np.allclose(d.timestamps, ts)


# ---------------------------------------------------------------------------
# Ragged trace round-trip
# ---------------------------------------------------------------------------


def test_ragged_trace_roundtrip_with_y(tmp_path):
    """Ragged traces with outer y axis: list of arrays, lengths and values correct."""
    rng = np.random.default_rng(0)
    lengths = [10, 15, 12]
    traces = [rng.standard_normal(n) + 1j * rng.standard_normal(n) for n in lengths]
    xs = [np.linspace(4e9, 5e9, n) for n in lengths]
    power = np.array([-20.0, -10.0, 0.0])

    path = str(tmp_path / "ragged_with_y")
    save_labber_trace_data(
        path,
        z=("S21", "", traces),
        x=("Freq", "Hz", xs),
        y=("Power", "dBm", power),
    )

    d = load_labber_data(path)

    # z must be a list because lengths differ
    assert isinstance(d.z, list)
    assert len(d.z) == len(traces)
    for i, (t_out, t_in) in enumerate(zip(d.z, traces)):
        assert len(t_out) == lengths[i]
        assert np.allclose(t_out, t_in)


def test_ragged_trace_roundtrip_no_y(tmp_path):
    """Ragged traces without outer axis: list returned, values correct."""
    rng = np.random.default_rng(1)
    lengths = [8, 13]
    traces = [rng.standard_normal(n) + 1j * rng.standard_normal(n) for n in lengths]
    xs = [np.linspace(4e9, 5e9, n) for n in lengths]

    path = str(tmp_path / "ragged_no_y")
    save_labber_trace_data(
        path,
        z=("S21", "", traces),
        x=("Freq", "Hz", xs),
    )

    d = load_labber_data(path)
    assert isinstance(d.z, list)
    for t_out, t_in in zip(d.z, traces):
        assert np.allclose(t_out, t_in)


def test_equal_length_traces_stacked_no_outer(tmp_path):
    """Equal-length traces without outer axis: stacked (Nentries, n) ndarray, values correct."""
    rng = np.random.default_rng(2)
    n = 20
    traces = [rng.standard_normal(n) + 1j * rng.standard_normal(n) for _ in range(4)]
    x_shared = np.linspace(4e9, 5e9, n)

    path = str(tmp_path / "equal_traces_no_outer")
    save_labber_trace_data(
        path,
        z=("S21", "", traces),
        x=("Freq", "Hz", x_shared),
    )

    d = load_labber_data(path)
    assert isinstance(d.z, np.ndarray)
    assert d.z.shape == (len(traces), n)
    for i, t in enumerate(traces):
        assert np.allclose(d.z[i], t)


def test_equal_length_traces_stacked_with_outer(tmp_path):
    """Equal-length traces WITH outer axis: returns stacked ndarray, values correct."""
    rng = np.random.default_rng(3)
    n = 20
    traces = [rng.standard_normal(n) + 1j * rng.standard_normal(n) for _ in range(4)]
    x_shared = np.linspace(4e9, 5e9, n)
    power = np.array([-30.0, -20.0, -10.0, 0.0])

    path = str(tmp_path / "equal_traces_outer")
    save_labber_trace_data(
        path,
        z=("S21", "", traces),
        x=("Freq", "Hz", x_shared),
        y=("Power", "dBm", power),
    )

    d = load_labber_data(path)
    # equal-length with outer axis -> stacked ndarray shape (Ny, n)
    assert isinstance(d.z, np.ndarray)
    assert d.z.shape == (len(power), n)
    for i, t in enumerate(traces):
        assert np.allclose(d.z[i], t)


# ---------------------------------------------------------------------------
# File-extension handling
# ---------------------------------------------------------------------------


def test_extension_no_suffix(tmp_path):
    """Saving with no extension writes .hdf5; loading with no extension also works."""
    freq = np.linspace(4e9, 5e9, 11)
    z = np.ones(11, dtype=complex)

    path_no_ext = str(tmp_path / "nosuffix")
    save_labber_data(path_no_ext, z=("S21", "", z), axes=[("Freq", "Hz", freq)])

    # file must be written with .hdf5 extension
    import os

    assert os.path.exists(path_no_ext + ".hdf5")

    # loading without extension also succeeds
    d = load_labber_data(path_no_ext)
    assert np.allclose(d.z, z)


def test_extension_h5_becomes_hdf5(tmp_path):
    """Saving with .h5 extension rewrites it to .hdf5."""
    freq = np.linspace(4e9, 5e9, 11)
    z = np.ones(11, dtype=complex)

    path_h5 = str(tmp_path / "file.h5")
    save_labber_data(path_h5, z=("S21", "", z), axes=[("Freq", "Hz", freq)])

    import os

    assert os.path.exists(str(tmp_path / "file.hdf5"))
    assert not os.path.exists(path_h5)


def test_save_labber_data_rejects_existing_formatted_path(tmp_path):
    """Low-level persistence must not overwrite or suffix an existing target."""
    path = tmp_path / "existing.hdf5"
    path.write_bytes(b"existing")

    with pytest.raises(FileExistsError):
        save_labber_data(
            str(path),
            z=("S21", "", np.ones(1, dtype=complex)),
            axes=[("Freq", "Hz", np.array([1.0]))],
        )

    assert path.read_bytes() == b"existing"
    assert not (tmp_path / "existing_1.hdf5").exists()


def test_save_labber_trace_data_rejects_existing_formatted_path(tmp_path):
    path = tmp_path / "existing_trace.hdf5"
    path.write_bytes(b"existing")

    with pytest.raises(FileExistsError):
        save_labber_trace_data(
            str(path),
            z=("S21", "", [np.ones(2, dtype=complex)]),
            x=("Time", "s", [np.array([0.0, 1.0])]),
        )

    assert path.read_bytes() == b"existing"
    assert not (tmp_path / "existing_trace_1.hdf5").exists()


# ---------------------------------------------------------------------------
# LabberData.save / LabberData.load (class-level wrappers)
# ---------------------------------------------------------------------------


def test_labberdata_class_save_load(tmp_path):
    """LabberData.save / LabberData.load are equivalent to the free functions."""
    freq = np.linspace(4e9, 5e9, 21)
    power = np.linspace(-20.0, 0.0, 4)
    rng = np.random.default_rng(9)
    z2d = rng.standard_normal((len(power), len(freq))) + 1j * rng.standard_normal(
        (len(power), len(freq))
    )

    ld = LabberData(
        ("S21", "arb", z2d),
        axes=[
            Axis("Frequency", "Hz", freq),
            Axis("Power", "dBm", power),
        ],
        comment="class test",
        tags=["cls"],
    )

    path = str(tmp_path / "cls_io")
    ld.save(path)

    d = LabberData.load(path)
    assert np.allclose(d.z, z2d)
    assert d.comment == "class test"
    assert "cls" in d.tags
    assert np.allclose(d.x, freq)
    assert np.allclose(d.y, power)


# ---------------------------------------------------------------------------
# Error paths (fast-fail)
# ---------------------------------------------------------------------------


def test_axes_count_mismatch_raises(tmp_path):
    """len(axes) != z.ndim must raise ValueError."""
    z2d = np.ones((5, 10), dtype=complex)
    freq = np.linspace(4e9, 5e9, 10)

    with pytest.raises(ValueError, match="ndim"):
        # only one axis for 2-D data
        save_labber_data(
            str(tmp_path / "bad_axes"),
            z=("S21", "", z2d),
            axes=[("Freq", "Hz", freq)],
        )


def test_axis_length_mismatch_raises(tmp_path):
    """axis values length != corresponding z dimension must raise ValueError."""
    z2d = np.ones((5, 10), dtype=complex)
    freq_wrong = np.linspace(4e9, 5e9, 8)  # 8 != 10
    power = np.linspace(-30.0, 0.0, 5)

    with pytest.raises(ValueError):
        save_labber_data(
            str(tmp_path / "bad_len"),
            z=("S21", "", z2d),
            axes=[("Freq", "Hz", freq_wrong), ("Power", "dBm", power)],
        )


def test_timestamps_length_mismatch_raises(tmp_path):
    """len(timestamps) != n_entries must raise ValueError."""
    freq = np.linspace(4e9, 5e9, 11)
    power = np.linspace(-30.0, 0.0, 5)
    z2d = np.ones((len(power), len(freq)), dtype=complex)
    ts_wrong = np.arange(3, dtype=float)  # 3 != 5 entries

    with pytest.raises(ValueError, match="timestamps"):
        save_labber_data(
            str(tmp_path / "bad_ts"),
            z=("S21", "", z2d),
            axes=[("Freq", "Hz", freq), ("Power", "dBm", power)],
            timestamps=ts_wrong,
        )
