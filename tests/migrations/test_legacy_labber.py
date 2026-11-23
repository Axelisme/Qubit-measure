"""Migration-only Labber reading through the public offline contract."""

from dataclasses import dataclass
from pathlib import Path

import h5py as h5
import numpy as np
import pytest
from zcu_tools.cfg_model import ConfigBase
from zcu_tools.datafile import (
    GroupedLabberData,
    LabberData,
    LabberMetadata,
    LabberPayload,
    format_ext,
    load_labber_logs,
)
from zcu_tools.datafile.grouped import write_grouped_labber_data_file
from zcu_tools.experiment import (
    GroupedAxesSpec,
    GroupedLoadData,
    VariableAxisSpec,
    VariableSpec,
    VariableZSpec,
)

from migrations.experiment_declaration import declare_migration_experiment
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
        LabberData(payload.data, payload.axes, metadata=metadata).save(str(source))
    else:
        write_grouped_labber_data_file(
            format_ext(str(source)),
            GroupedLabberData({"measured": payload}, metadata=metadata),
        )
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


class _EchoCfg(ConfigBase):
    pass


@dataclass(frozen=True)
class _EchoResult:
    times: np.ndarray
    signal: np.ndarray
    iq_moments: np.ndarray | None = None


def test_declared_historical_echo_schema_reads_unmarked_log_with_native_moments(
    tmp_path: Path,
) -> None:
    time = VariableAxisSpec(label="Time", unit="s", field_name="times", scale=1e-6)

    def restore(data: GroupedLoadData[_EchoCfg]) -> _EchoResult:
        signal = data.variable("data")
        return _EchoResult(signal.axes[0], signal.z)

    spec = GroupedAxesSpec(
        variables=(
            VariableSpec(
                "data",
                (time,),
                VariableZSpec(
                    "signal", label="Signal", unit="a.u.", dtype=np.complex128
                ),
            ),
            VariableSpec(
                "iq_moments",
                (VariableAxisSpec(label="Moment", unit="", generated="arange"), time),
                VariableZSpec("iq_moments", label="Moments", unit="a.u."),
                required=False,
            ),
        ),
        result_type=_EchoResult,
        cfg_type=_EchoCfg,
        result_builder=restore,
        tag="synthetic/echo",
    )
    history = LegacyVariableSchema(
        variable="data",
        axes=(LegacyAxisSchema(name="Time", unit="s", dtype=np.dtype("float64")),),
        signal_name="Signal",
        signal_unit="a.u.",
        signal_dtype=np.dtype("complex128"),
    )
    declaration = declare_migration_experiment(spec, schemas=(history,))
    source = tmp_path / "echo.hdf5"
    LabberData(
        ("Signal", "a.u.", np.array([1 + 2j, 3 + 4j])),
        [("Time", "s", np.array([0.0, 1e-6]))],
    ).save(str(source))

    assert [item.variable for item in declaration.schemas] == ["data"]
    loaded, _ = load_legacy_labber_payload(source, schema=declaration.schemas)
    np.testing.assert_array_equal(loaded.vars["data"].values, [1 + 2j, 3 + 4j])


def test_unmarked_file_does_not_guess_multiple_variable_identities(
    tmp_path: Path,
) -> None:
    source = tmp_path / "signal.hdf5"
    LabberData(
        ("voltage", "V", np.array([1.0])), [("time", "s", np.array([0.0]))]
    ).save(str(source))
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
        LabberData(
            (name, "V", np.asarray(signal)),
            [(axis, "s", np.asarray(values))],
            timestamps=np.array([1759276800.0 + index]),
            metadata=LabberMetadata(tags=["offline"], creation_time=1759276800.0),
        ).save(str(path))
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
    LabberData(
        ("voltage", "mV", np.array([1.0])), [("time", "s", np.array([0.0]))]
    ).save(str(source))
    with pytest.raises(ValueError, match="unit"):
        load_legacy_labber_payload(source, schema=(_schema("one"),))
