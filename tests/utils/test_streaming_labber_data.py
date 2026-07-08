"""Streaming grouped Labber writer tests."""

from __future__ import annotations

import h5py
import numpy as np
import pytest
from zcu_tools.utils.datasaver import (
    DatasetRole,
    LabberMetadata,
    StreamingLabberRoleSpec,
    load_grouped_labber_data,
    load_labber_data,
    open_streaming_grouped_labber_data,
    open_streaming_labber_data,
)


def test_streaming_writer_commits_one_2d_row_and_leaves_nan_rows(tmp_path):
    path = str(tmp_path / "streamed")
    flux = np.array([0.0, 0.5, 1.0], dtype=float)
    detune = np.array([-1.0, 0.0, 1.0, 2.0], dtype=float)
    spec = StreamingLabberRoleSpec(
        "signal",
        "Signal",
        "a.u.",
        axes=[("Detune", "MHz", detune), ("Flux device value", "", flux)],
        shape=(len(flux), len(detune)),
    )

    writer = open_streaming_grouped_labber_data(
        path,
        [spec],
        metadata=LabberMetadata(comment="streaming"),
    )
    writer.write_outer_slice("signal", 1, np.array([1.0, 2.0, 3.0, 4.0]))
    writer.flush()
    writer.finalize()
    writer.close()
    writer.close()

    loaded = load_grouped_labber_data(path, required_roles=("signal",))
    payload = loaded.roles[DatasetRole("signal")]
    np.testing.assert_allclose(payload.z[1].real, [1.0, 2.0, 3.0, 4.0])
    assert np.isnan(payload.z[0].real).all()
    assert np.isnan(payload.z[2].real).all()
    assert loaded.metadata.comment == "streaming"

    with h5py.File(path + ".hdf5", "r") as handle:
        assert bool(handle.attrs["zcu_tools.streaming_finalized"]) is True


def test_streaming_single_log_writer_commits_one_2d_row(tmp_path):
    path = str(tmp_path / "single_stream")
    flux = np.array([0.0, 0.5, 1.0], dtype=float)
    detune = np.array([-1.0, 0.0, 1.0, 2.0], dtype=float)
    spec = StreamingLabberRoleSpec(
        "signal",
        "Signal",
        "a.u.",
        axes=[("Detune", "MHz", detune), ("Flux device value", "", flux)],
        shape=(len(flux), len(detune)),
    )

    with open_streaming_labber_data(
        path,
        spec,
        metadata=LabberMetadata(comment="single streaming"),
    ) as writer:
        writer.write_outer_slice(1, np.array([1.0, 2.0, 3.0, 4.0]))
        writer.finalize()

    loaded = load_labber_data(path)
    np.testing.assert_allclose(loaded.z[1].real, [1.0, 2.0, 3.0, 4.0])
    assert np.isnan(loaded.z[0].real).all()
    assert np.isnan(loaded.z[2].real).all()
    assert loaded.metadata.comment == "single streaming"

    with h5py.File(path + ".hdf5", "r") as handle:
        assert "Data" in handle
        assert "zcu_tools.grouped_dataset_version" not in handle.attrs
        assert bool(handle.attrs["zcu_tools.streaming_finalized"]) is True


def test_streaming_writer_commits_one_3d_outer_block(tmp_path):
    path = str(tmp_path / "streamed_3d")
    flux = np.array([0.0, 0.5], dtype=float)
    freq = np.array([10.0, 11.0, 12.0], dtype=float)
    gain = np.array([0.1, 0.2], dtype=float)
    spec = StreamingLabberRoleSpec(
        "signal",
        "Signal",
        "a.u.",
        axes=[
            ("Gain", "a.u.", gain),
            ("Frequency", "MHz", freq),
            ("Flux device value", "", flux),
        ],
        shape=(len(flux), len(freq), len(gain)),
    )
    row = np.arange(len(freq) * len(gain), dtype=float).reshape(len(freq), len(gain))

    with open_streaming_grouped_labber_data(path, [spec]) as writer:
        writer.write_outer_slice("signal", 0, row)
        writer.flush()

    loaded = load_grouped_labber_data(path, required_roles=("signal",))
    payload = loaded.roles[DatasetRole("signal")]
    np.testing.assert_allclose(payload.z[0].real, row)
    assert np.isnan(payload.z[1].real).all()


def test_streaming_writer_rejects_existing_formatted_path(tmp_path):
    path = tmp_path / "existing.hdf5"
    path.write_bytes(b"existing")
    spec = StreamingLabberRoleSpec(
        "signal",
        "Signal",
        "a.u.",
        axes=[("Flux", "", np.array([0.0]))],
        shape=(1,),
    )

    with pytest.raises(FileExistsError):
        open_streaming_grouped_labber_data(str(path), [spec])

    assert path.read_bytes() == b"existing"


def test_streaming_single_log_writer_rejects_existing_formatted_path(tmp_path):
    path = tmp_path / "existing_single.hdf5"
    path.write_bytes(b"existing")
    spec = StreamingLabberRoleSpec(
        "signal",
        "Signal",
        "a.u.",
        axes=[("Flux", "", np.array([0.0]))],
        shape=(1,),
    )

    with pytest.raises(FileExistsError):
        open_streaming_labber_data(str(path), spec)

    assert path.read_bytes() == b"existing"


def test_streaming_grouped_close_marks_closed_when_underlying_close_fails(
    tmp_path, monkeypatch
):
    spec = StreamingLabberRoleSpec(
        "signal",
        "Signal",
        "a.u.",
        axes=[("Flux", "", np.array([0.0]))],
        shape=(1,),
    )
    writer = open_streaming_grouped_labber_data(
        str(tmp_path / "broken_grouped"), [spec]
    )
    original_file = writer._file

    def fail_close() -> None:
        raise RuntimeError("close boom")

    monkeypatch.setattr(original_file, "close", fail_close)
    with pytest.raises(RuntimeError, match="close boom"):
        writer.close()

    writer.close()
    monkeypatch.undo()
    original_file.close()


def test_streaming_single_log_close_marks_closed_when_underlying_close_fails(
    tmp_path, monkeypatch
):
    spec = StreamingLabberRoleSpec(
        "signal",
        "Signal",
        "a.u.",
        axes=[("Flux", "", np.array([0.0]))],
        shape=(1,),
    )
    writer = open_streaming_labber_data(str(tmp_path / "broken_single"), spec)
    original_file = writer._file

    def fail_close() -> None:
        raise RuntimeError("close boom")

    monkeypatch.setattr(original_file, "close", fail_close)
    with pytest.raises(RuntimeError, match="close boom"):
        writer.close()

    writer.close()
    monkeypatch.undo()
    original_file.close()


def test_streaming_writer_validates_role_shape(tmp_path):
    spec = StreamingLabberRoleSpec(
        "fit_freq",
        "Fit frequency",
        "MHz",
        axes=[("Flux", "", np.array([0.0, 1.0]))],
        shape=(2,),
    )
    with open_streaming_grouped_labber_data(str(tmp_path / "scalar"), [spec]) as writer:
        writer.write_outer_slice("fit_freq", 0, 5000.0)
        with pytest.raises(ValueError, match="scalar row"):
            writer.write_outer_slice("fit_freq", 1, np.array([1.0, 2.0]))


def test_streaming_single_log_writer_validates_scalar_shape(tmp_path):
    spec = StreamingLabberRoleSpec(
        "fit_freq",
        "Fit frequency",
        "MHz",
        axes=[("Flux", "", np.array([0.0, 1.0]))],
        shape=(2,),
    )
    with open_streaming_labber_data(str(tmp_path / "single_scalar"), spec) as writer:
        writer.write_outer_slice(0, 5000.0)
        with pytest.raises(ValueError, match="scalar row"):
            writer.write_outer_slice(1, np.array([1.0, 2.0]))
