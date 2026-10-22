"""Migration-only Labber reading through the public offline contract."""

from pathlib import Path

import h5py as h5
import numpy as np
import pytest
from zcu_tools.datafile import (
    LabberMetadata,
    LabberPayload,
    load_labber_logs,
    save_grouped_labber_data,
    save_labber_data,
)

from migrations.legacy_labber import (
    LegacyAxisSchema,
    LegacyVariableSchema,
    load_legacy_labber_payload,
)


def _schema(
    variable: str, *, axis: str = "time", signal: str = "voltage"
) -> LegacyVariableSchema:
    return LegacyVariableSchema(
        variable=variable,
        axes=(LegacyAxisSchema(name=axis, unit="s", dtype=np.dtype("float64")),),
        signal_name=signal,
        signal_unit="V",
        signal_dtype=np.dtype("float64"),
    )


def _dataset(file: h5.File, name: str) -> h5.Dataset:
    node = file[name]
    if not isinstance(node, h5.Dataset):
        raise TypeError(name)
    return node


@pytest.mark.parametrize("representation", ["single", "grouped"])
def test_explicit_variable_identity_and_reader_metadata_are_preserved(
    tmp_path: Path, representation: str
) -> None:
    source = tmp_path / "unrelated_filename.hdf5"
    payload = LabberPayload(
        ("voltage", "V", np.array([2.0, 3.0])), [("time", "s", np.array([0.0, 1.0]))]
    )
    metadata = LabberMetadata(tags=["synthetic"], project="test_project")
    if representation == "single":
        save_labber_data(
            str(source), z=payload.data, axes=payload.axes, metadata=metadata
        )
    else:
        save_grouped_labber_data(str(source), {"measured": payload}, metadata=metadata)
    before = source.read_bytes()
    loaded, actual_metadata = load_legacy_labber_payload(
        source, schema=(_schema("measured"),)
    )
    assert actual_metadata.tags == ["synthetic"]
    assert actual_metadata.project == "test_project"
    assert tuple(loaded.vars) == ("measured",)
    np.testing.assert_array_equal(loaded.vars["measured"].values, [2.0, 3.0])
    assert loaded.vars["measured"].values.dtype == np.dtype("float64")
    assert source.read_bytes() == before


def test_unmarked_file_does_not_guess_multiple_variable_identities(
    tmp_path: Path,
) -> None:
    source = tmp_path / "signal.hdf5"
    save_labber_data(
        str(source),
        z=("voltage", "V", np.array([1.0])),
        axes=[("time", "s", np.array([0.0]))],
    )
    with pytest.raises(ValueError, match="cannot identify multiple"):
        load_legacy_labber_payload(source, schema=(_schema("one"), _schema("two")))


@pytest.fixture
def unmarked_logs(tmp_path: Path) -> tuple[Path, tuple[LegacyVariableSchema, ...]]:
    source = tmp_path / "old.hdf5"
    schemas = []
    for index, (name, axis, values, signal) in enumerate(
        [
            ("first", "time", [0.0, 1.0], [1.0, 2.0]),
            ("second", "frequency", [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]),
        ]
    ):
        path = source if index == 0 else tmp_path / "additional.hdf5"
        save_labber_data(
            str(path),
            z=(name, "V", np.asarray(signal)),
            axes=[(axis, "s", np.asarray(values))],
            timestamps=np.array([1759276800.0 + index]),
            metadata=LabberMetadata(tags=["offline"], creation_time=1759276800.0),
        )
        if index:
            with h5.File(source, "r+") as root, h5.File(path, "r") as extra:
                group = root.create_group("Log_2")
                for key in ("Data", "Channels", "Log list"):
                    extra.copy(key, group)
        schemas.append(_schema(name, axis=axis, signal=name))
    return source, tuple(schemas)


def test_unmarked_logs_follow_explicit_schema_order_with_independent_grids(
    unmarked_logs: tuple[Path, tuple[LegacyVariableSchema, ...]], tmp_path: Path
) -> None:
    source, schema = unmarked_logs
    original = source.read_bytes()
    loaded, metadata = load_legacy_labber_payload(source, schema=schema)
    assert metadata.tags == ["offline"]
    np.testing.assert_array_equal(loaded.vars["first"].values, [1.0, 2.0])
    np.testing.assert_array_equal(loaded.vars["second"].values, [6.0, 7.0, 8.0])
    np.testing.assert_array_equal(loaded.vars["second"].axes[0].values, [3.0, 4.0, 5.0])
    from zcu_tools.datafile import save

    destination = tmp_path / "converted.h5"
    save(destination, loaded)
    with h5.File(destination, "r") as file:
        original_tree = file["groups/labber"]
        assert isinstance(original_tree, h5.Group)
        # Original timestamps remain available in the opaque Labber subtree.
        detached = tmp_path / "detached.hdf5"
        with h5.File(detached, "w") as copy:
            for name in original_tree:
                original_tree.copy(name, copy)
            copy.attrs.update(original_tree.attrs)
    np.testing.assert_array_equal(
        load_labber_logs(str(detached))[1].timestamps, [1759276801.0]
    )
    assert source.read_bytes() == original


@pytest.mark.parametrize("mismatch", ["count", "signal", "axis", "unit", "shape"])
def test_unmarked_log_schema_mismatch_is_located(
    unmarked_logs: tuple[Path, tuple[LegacyVariableSchema, ...]], mismatch: str
) -> None:
    source, schema = unmarked_logs
    if mismatch == "count":
        schema = schema[:1]
    else:
        with h5.File(source, "r+") as file:
            if mismatch == "shape":
                file["Log_2/Data"].attrs["Step dimensions"] = np.array([99])
            elif mismatch == "signal":
                dataset = _dataset(file, "Log_2/Log list")
                rows = dataset[()]
                rows[0]["channel_name"] = "wrong"
                dataset[...] = rows
            else:
                dataset = _dataset(file, "Log_2/Channels")
                rows = dataset[()]
                rows[0]["name" if mismatch == "axis" else "unitPhys"] = "wrong"
                dataset[...] = rows
    with pytest.raises(ValueError, match="log count|/second|Log_2"):
        load_legacy_labber_payload(source, schema=schema)


def test_declared_units_are_not_guessed_or_converted(tmp_path: Path) -> None:
    source = tmp_path / "signal.hdf5"
    save_labber_data(
        str(source),
        z=("voltage", "mV", np.array([1.0])),
        axes=[("time", "s", np.array([0.0]))],
    )
    with pytest.raises(ValueError, match="unit"):
        load_legacy_labber_payload(source, schema=(_schema("one"),))
