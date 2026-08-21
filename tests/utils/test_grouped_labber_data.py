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


def _payload_2d_reference() -> LabberPayload:
    payload = _payload_2d()
    return LabberPayload(("Reference", "arb", payload.z + 1.0), axes=payload.axes)


def test_grouped_roundtrip_with_metadata_and_attrs(tmp_path):
    path = tmp_path / "grouped"
    metadata = LabberMetadata(
        comment="grouped result",
        tags=["adr_0027"],
        project="proj",
        user="alice",
        creation_time=1_700_000_000.0,
    )

    signal = _payload_2d()
    reference = LabberPayload(("Reference", "arb", signal.z + 1.0), axes=signal.axes)
    written = save_grouped_labber_data(
        str(path),
        {"signal": signal, "reference": reference},
        metadata=metadata,
    )

    assert written == str(path) + ".hdf5"
    with h5py.File(written, "r") as f:
        version_attr = cast(SupportsInt, f.attrs["zcu_tools.grouped_dataset_version"])
        roles_attr = cast(Iterable[str], f.attrs["zcu_tools.dataset_roles"])
        channels_attr = cast(Iterable[str], f.attrs["zcu_tools.dataset_role_channels"])
        assert int(version_attr) == 2
        assert list(roles_attr) == ["signal", "reference"]
        assert list(channels_attr) == ["Signal", "Reference"]
        assert "zcu_tools.dataset_role" not in f.attrs
        assert not any(str(name).startswith("Log_") for name in f)
        log_list = f.get("Log list")
        assert isinstance(log_list, h5py.Dataset)
        assert [row["channel_name"] for row in log_list[()]] == [
            b"Signal",
            b"Reference",
        ]

    loaded = load_grouped_labber_data(
        str(path), required_roles=("signal", DatasetRole("reference"))
    )

    assert loaded.metadata.comment == "grouped result"
    assert loaded.metadata.tags == ["adr_0027"]
    assert loaded.metadata.project == "proj"
    assert loaded.metadata.user == "alice"
    assert loaded.metadata.creation_time == 1_700_000_000.0
    assert list(loaded.roles) == [DatasetRole("signal"), DatasetRole("reference")]
    assert np.array_equal(loaded.roles[DatasetRole("signal")].z, _payload_2d().z)
    assert np.array_equal(
        loaded.roles[DatasetRole("reference")].z, _payload_2d().z + 1.0
    )


def test_grouped_v2_rejects_heterogeneous_axes_before_creating_file(tmp_path):
    path = tmp_path / "heterogeneous"

    with pytest.raises(ValueError, match="common grid"):
        save_grouped_labber_data(
            str(path), {"signal": _payload_2d(), "reference": _payload_1d()}
        )

    assert not (tmp_path / "heterogeneous.hdf5").exists()


@pytest.mark.parametrize(
    ("case", "match"),
    [
        ("zero_axis", "at least one step axis"),
        ("shape", "shape"),
        ("axis_values", "common grid"),
        ("axis_label", "common grid"),
        ("axis_unit", "common grid"),
        ("timestamp_cardinality", "timestamps"),
        ("timestamp_equality", "timestamps"),
        ("timestamp_presence", "timestamps"),
        ("empty_channel", "non-empty"),
        ("duplicate_channel", "unique"),
        ("axis_channel_collision", "unique"),
        ("ragged", "ragged|vector"),
        ("vector", "axes"),
        ("object", "numeric"),
    ],
)
def test_grouped_v2_validates_complete_contract_before_file_creation(
    tmp_path, case: str, match: str
) -> None:
    axis = np.arange(3, dtype=float)
    base = LabberPayload(("Signal", "a.u.", np.arange(3.0)), [("X", "s", axis)])
    roles: Mapping[str, LabberPayload]
    if case == "zero_axis":
        roles = {"signal": LabberPayload(("Signal", "a.u.", np.arange(3.0)), [])}
    elif case == "shape":
        roles = {
            "signal": base,
            "reference": LabberPayload(
                ("Reference", "a.u.", np.arange(4.0)), [("X", "s", axis)]
            ),
        }
    elif case == "axis_values":
        roles = {
            "signal": base,
            "reference": LabberPayload(
                ("Reference", "a.u.", np.arange(3.0)),
                [("X", "s", axis + 1.0)],
            ),
        }
    elif case == "axis_label":
        roles = {
            "signal": base,
            "reference": LabberPayload(
                ("Reference", "a.u.", np.arange(3.0)),
                [("Other X", "s", axis)],
            ),
        }
    elif case == "axis_unit":
        roles = {
            "signal": base,
            "reference": LabberPayload(
                ("Reference", "a.u.", np.arange(3.0)),
                [("X", "ms", axis)],
            ),
        }
    elif case == "timestamp_cardinality":
        roles = {
            "signal": LabberPayload(
                base.data, base.axes, timestamps=np.array([1.0, 2.0])
            )
        }
    elif case == "timestamp_equality":
        y = np.arange(2, dtype=float)
        shape = (2, 3)
        roles = {
            "signal": LabberPayload(
                ("Signal", "a.u.", np.ones(shape)),
                [("X", "s", axis), ("Y", "V", y)],
                timestamps=np.array([1.0, 2.0]),
            ),
            "reference": LabberPayload(
                ("Reference", "a.u.", np.ones(shape)),
                [("X", "s", axis), ("Y", "V", y)],
                timestamps=np.array([1.0, 3.0]),
            ),
        }
    elif case == "timestamp_presence":
        roles = {
            "signal": base,
            "reference": LabberPayload(
                ("Reference", "a.u.", np.arange(3.0)),
                base.axes,
                timestamps=np.array([1.0]),
            ),
        }
    elif case == "empty_channel":
        roles = {"signal": LabberPayload(("", "a.u.", np.arange(3.0)), base.axes)}
    elif case == "duplicate_channel":
        roles = {
            "signal": base,
            "reference": LabberPayload(("Signal", "a.u.", np.arange(3.0)), base.axes),
        }
    elif case == "axis_channel_collision":
        roles = {"signal": LabberPayload(("X", "a.u.", np.arange(3.0)), base.axes)}
    elif case == "ragged":
        roles = {
            "signal": LabberPayload(
                ("Signal", "a.u.", [np.array([1.0]), np.array([2.0, 3.0])]),
                [("X", "s", np.arange(2, dtype=float))],
            )
        }
    elif case == "vector":
        roles = {
            "signal": LabberPayload(
                ("Signal", "a.u.", np.ones((3, 2))),
                [("X", "s", np.arange(3, dtype=float))],
            )
        }
    else:
        roles = {
            "signal": LabberPayload(
                ("Signal", "a.u.", np.array([object()], dtype=object)),
                [("X", "s", np.array([0.0]))],
            )
        }

    path = tmp_path / case
    with pytest.raises(ValueError, match=match):
        save_grouped_labber_data(str(path), roles)
    assert not path.with_suffix(".hdf5").exists()


def test_grouped_v2_accepts_flat_numeric_list_values(tmp_path):
    path = save_grouped_labber_data(
        str(tmp_path / "flat_list"),
        {
            "signal": LabberPayload(
                ("Signal", "a.u.", [1.0, 2.0, 3.0]),
                [("X", "s", [0.0, 1.0, 2.0])],
            )
        },
    )

    loaded = load_grouped_labber_data(path, required_roles=("signal",))
    np.testing.assert_array_equal(
        loaded.roles[DatasetRole("signal")].z, [1.0, 2.0, 3.0]
    )


def test_grouped_v2_rejects_unmarked_v1_with_manual_migration_command(tmp_path):
    path = save_grouped_labber_data(
        str(tmp_path / "legacy_v1"), {"signal": _payload_2d()}
    )
    with h5py.File(path, "a") as f:
        f.attrs["zcu_tools.grouped_dataset_version"] = 1
        del f.attrs["zcu_tools.dataset_role_channels"]

    with pytest.raises(ValueError) as exc_info:
        load_grouped_labber_data(path)

    assert (
        str(exc_info.value)
        == """grouped dataset version 1 requires manual migration:
.venv/bin/python script/migrate_experiment_data.py \\
  --experiment grouped/v1 \\
  --input INPUT.hdf5 \\
  --output OUTPUT.hdf5"""
    )


@pytest.mark.parametrize(
    ("grouped_version", "streaming_version", "match"),
    [
        (2.5, None, "invalid grouped dataset version"),
        (1, 1.5, "invalid streaming grouped dataset version"),
    ],
)
def test_grouped_loader_rejects_fractional_version_markers(
    tmp_path, grouped_version: float, streaming_version: float | None, match: str
) -> None:
    path = save_grouped_labber_data(
        str(tmp_path / f"bad_version_{grouped_version}_{streaming_version}"),
        {"signal": _payload_2d()},
    )
    with h5py.File(path, "a") as f:
        f.attrs["zcu_tools.grouped_dataset_version"] = grouped_version
        if streaming_version is not None:
            f.attrs["zcu_tools.streaming_grouped_dataset_version"] = streaming_version

    with pytest.raises(ValueError, match=match):
        load_grouped_labber_data(path)


def test_grouped_v2_loader_validates_explicit_role_channel_mapping(tmp_path):
    roles = {"signal": _payload_2d(), "reference": _payload_2d_reference()}
    short_path = save_grouped_labber_data(str(tmp_path / "short_mapping"), roles)
    with h5py.File(short_path, "a") as f:
        f.attrs["zcu_tools.dataset_role_channels"] = np.array(
            ["Signal"], dtype=h5py.string_dtype("utf-8")
        )
    with pytest.raises(ValueError, match="mapping lengths"):
        load_grouped_labber_data(short_path)

    wrong_path = save_grouped_labber_data(str(tmp_path / "wrong_mapping"), roles)
    with h5py.File(wrong_path, "a") as f:
        f.attrs["zcu_tools.dataset_role_channels"] = np.array(
            ["Signal", "Not Reference"], dtype=h5py.string_dtype("utf-8")
        )
    with pytest.raises(ValueError, match="actual Labber channels"):
        load_grouped_labber_data(wrong_path)


def test_grouped_v2_loader_rejects_corrupt_step_bookkeeping(tmp_path):
    dimensions_path = save_grouped_labber_data(
        str(tmp_path / "bad_dimensions"), {"signal": _payload_2d()}
    )
    with h5py.File(dimensions_path, "a") as f:
        f["Data"].attrs["Step dimensions"] = np.array([4, 4], dtype=np.int64)
    with pytest.raises(ValueError, match="Step dimensions"):
        load_grouped_labber_data(dimensions_path)

    coordinates_path = save_grouped_labber_data(
        str(tmp_path / "bad_coordinates"), {"signal": _payload_2d()}
    )
    with h5py.File(coordinates_path, "a") as f:
        data_group = f.get("Data")
        assert isinstance(data_group, h5py.Group)
        data = data_group.get("Data")
        assert isinstance(data, h5py.Dataset)
        data[0, 1, 1] += 1.0
    with pytest.raises(ValueError, match="step-coordinate"):
        load_grouped_labber_data(coordinates_path)

    step_list_path = save_grouped_labber_data(
        str(tmp_path / "bad_step_list"), {"signal": _payload_2d()}
    )
    with h5py.File(step_list_path, "a") as f:
        step_list_dataset = f.get("Step list")
        assert isinstance(step_list_dataset, h5py.Dataset)
        step_list = step_list_dataset[()]
        step_list[0]["channel_name"] = "Wrong axis"
        del f["Step list"]
        f.create_dataset("Step list", data=step_list)
    with pytest.raises(ValueError, match="step-channel bookkeeping"):
        load_grouped_labber_data(step_list_path)


def test_grouped_strict_required_roles_missing_and_unknown_raise(tmp_path):
    path = tmp_path / "strict"
    save_grouped_labber_data(
        str(path), {"signal": _payload_2d(), "reference": _payload_2d_reference()}
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
        str(path), {"signal": _payload_2d(), "reference": _payload_2d_reference()}
    )

    loaded = load_grouped_labber_data(str(path))

    assert set(loaded.roles) == {DatasetRole("signal"), DatasetRole("reference")}


def test_grouped_invalid_and_duplicate_roles_raise(tmp_path):
    with pytest.raises(ValueError, match="lowercase snake_case"):
        save_grouped_labber_data(str(tmp_path / "invalid"), {"BadRole": _payload_2d()})

    path = save_grouped_labber_data(
        str(tmp_path / "duplicate"),
        {"signal": _payload_2d(), "reference": _payload_2d_reference()},
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
