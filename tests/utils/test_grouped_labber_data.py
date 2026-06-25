"""Grouped Labber dataset persistence tests."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import SupportsInt, cast

import h5py
import numpy as np
import pytest
from zcu_tools.utils.datasaver import (
    Axis,
    DatasetRole,
    LabberData,
    LabberMetadata,
    LabberPayload,
    load_grouped_labber_data,
    load_labber_data,
    save_grouped_labber_data,
)


def _payload_2d() -> LabberPayload:
    freq = np.linspace(4e9, 5e9, 4)
    power = np.linspace(-30.0, 0.0, 3)
    z = np.arange(len(power) * len(freq), dtype=float).reshape(len(power), len(freq))
    return LabberPayload(
        ("Signal", "arb", z + 1j * z),
        axes=[("Frequency", "Hz", freq), ("Power", "dBm", power)],
    )


def _payload_1d() -> LabberPayload:
    time = np.linspace(0.0, 1.0e-6, 5)
    z = np.exp(1j * np.linspace(0.0, np.pi, len(time)))
    return LabberPayload(("Reference", "arb", z), axes=[("Time", "s", time)])


def test_grouped_roundtrip_with_metadata_and_attrs(tmp_path):
    path = tmp_path / "grouped"
    metadata = LabberMetadata(
        comment="grouped result",
        tags=["adr_0027"],
        project="proj",
        user="alice",
        creation_time=1_700_000_000.0,
    )

    written = save_grouped_labber_data(
        str(path),
        {"signal": _payload_2d(), "reference": _payload_1d()},
        metadata=metadata,
    )

    assert written == str(path) + ".hdf5"
    with h5py.File(written, "r") as f:
        version_attr = cast(SupportsInt, f.attrs["zcu_tools.grouped_dataset_version"])
        roles_attr = cast(Iterable[str], f.attrs["zcu_tools.dataset_roles"])
        assert int(version_attr) == 1
        assert list(roles_attr) == ["signal", "reference"]
        assert f.attrs["zcu_tools.dataset_role"] == "signal"
        assert f["Log_2"].attrs["zcu_tools.dataset_role"] == "reference"

    loaded = load_grouped_labber_data(
        str(path), required_roles=("signal", DatasetRole("reference"))
    )

    assert loaded.metadata.comment == "grouped result"
    assert loaded.metadata.tags == ["adr_0027"]
    assert loaded.metadata.project == "proj"
    assert loaded.metadata.user == "alice"
    assert loaded.metadata.creation_time == 1_700_000_000.0
    assert set(loaded.roles) == {DatasetRole("signal"), DatasetRole("reference")}


def test_grouped_roundtrip_allows_heterogeneous_axes_and_shapes(tmp_path):
    path = tmp_path / "heterogeneous"
    signal = _payload_2d()
    reference = _payload_1d()

    save_grouped_labber_data(
        str(path), {"signal": signal, "reference_trace": reference}
    )

    loaded = load_grouped_labber_data(
        str(path), required_roles=("signal", "reference_trace")
    )

    signal_out = loaded.roles[DatasetRole("signal")]
    reference_out = loaded.roles[DatasetRole("reference_trace")]
    assert signal_out.z.shape == (3, 4)
    assert reference_out.z.shape == (5,)
    assert np.allclose(signal_out.z, signal.z)
    assert np.allclose(reference_out.z, reference.z)
    assert [axis.name for axis in signal_out.axes] == ["Frequency", "Power"]
    assert [axis.name for axis in reference_out.axes] == ["Time"]


def test_grouped_strict_required_roles_missing_and_unknown_raise(tmp_path):
    path = tmp_path / "strict"
    save_grouped_labber_data(
        str(path), {"signal": _payload_2d(), "reference": _payload_1d()}
    )

    with pytest.raises(ValueError, match="missing required dataset role"):
        load_grouped_labber_data(
            str(path), required_roles=("signal", "reference", "calibration")
        )

    with pytest.raises(ValueError, match="unknown dataset role"):
        load_grouped_labber_data(str(path), required_roles=("signal",))


def test_grouped_diagnostic_load_returns_all_roles(tmp_path):
    path = tmp_path / "diagnostic"
    save_grouped_labber_data(
        str(path), {"signal": _payload_2d(), "reference": _payload_1d()}
    )

    loaded = load_grouped_labber_data(str(path))

    assert set(loaded.roles) == {DatasetRole("signal"), DatasetRole("reference")}


def test_grouped_invalid_and_duplicate_roles_raise(tmp_path):
    with pytest.raises(ValueError, match="lowercase snake_case"):
        save_grouped_labber_data(str(tmp_path / "invalid"), {"BadRole": _payload_2d()})

    path = save_grouped_labber_data(
        str(tmp_path / "duplicate"),
        {"signal": _payload_2d(), "reference": _payload_1d()},
    )
    with h5py.File(path, "a") as f:
        f.attrs["zcu_tools.dataset_roles"] = np.array(
            ["signal", "signal"], dtype=h5py.string_dtype("utf-8")
        )

    with pytest.raises(ValueError, match="duplicate dataset role"):
        load_grouped_labber_data(path)


def test_single_labber_loader_fast_fails_on_grouped_file(tmp_path):
    path = tmp_path / "grouped_fast_fail"
    save_grouped_labber_data(str(path), {"signal": _payload_2d()})

    with pytest.raises(ValueError, match="load_grouped_labber_data"):
        load_labber_data(str(path))


def test_grouped_save_uses_exact_formatted_path_without_suffixing(tmp_path):
    existing_suffix = tmp_path / "run_1.hdf5"
    existing_suffix.write_bytes(b"existing")

    written = save_grouped_labber_data(
        str(tmp_path / "run.h5"), {"signal": _payload_2d()}
    )

    assert written == str(tmp_path / "run.hdf5")
    assert (tmp_path / "run.hdf5").exists()
    assert not (tmp_path / "run.h5").exists()
    assert existing_suffix.read_bytes() == b"existing"


def test_grouped_save_rejects_existing_formatted_path(tmp_path):
    path = tmp_path / "grouped.hdf5"
    path.write_bytes(b"existing")

    with pytest.raises(FileExistsError):
        save_grouped_labber_data(str(path), {"signal": _payload_2d()})

    assert path.read_bytes() == b"existing"
    assert not (tmp_path / "grouped_1.hdf5").exists()


def test_grouped_role_value_cannot_be_labber_data(tmp_path):
    data = LabberData(
        ("Signal", "arb", np.ones(3, dtype=complex)),
        axes=[Axis("Frequency", "Hz", np.arange(3, dtype=float))],
    )

    with pytest.raises(TypeError, match="LabberPayload"):
        invalid_roles = cast(
            Mapping[str | DatasetRole, LabberPayload], {"signal": data}
        )
        save_grouped_labber_data(str(tmp_path / "bad_value"), invalid_roles)
