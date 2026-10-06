"""Typed native persistence contracts with single and heterogeneous grouped specs."""

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import ClassVar, Literal

import h5py as h5
import numpy as np
import pytest
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator
from zcu_tools.datafile import (
    DataVariable,
    decode_labber_comment,
    load_grouped_labber_data,
    load_labber_data,
    load_run_data,
    save_run_data,
    validate_experiment_payload,
    write_labber,
)
from zcu_tools.experiment import (
    AxesSpec,
    ExpCfgModel,
    GroupedAxesSpec,
    GroupedLoadData,
    PersistableExperiment,
    PreparedRunData,
    RunRecord,
    VariableAxisSpec,
    VariableSpec,
    VariableZSpec,
    ZSpec,
    load_run,
    load_run_with_metadata,
    native_schemas,
    prepare_run_data,
    save_run,
)
from zcu_tools.experiment.axes_spec import Axis

from tests._native_support import hdf_dataset, hdf_json_text, native_metadata


class NestedCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    count: int = 3

    @model_validator(mode="before")
    @classmethod
    def reject_forbidden_raw_key(cls, data: object) -> object:
        if isinstance(data, dict) and "forbidden_future" in data:
            raise ValueError("before validator observed forbidden_future")
        return data


class NativeCfg(ExpCfgModel):
    model_config = ConfigDict(extra="forbid")
    nested: NestedCfg = NestedCfg()
    name: str = "historical"


@dataclass
class SingleResult:
    frequency: np.ndarray
    signal: np.ndarray


SINGLE = AxesSpec(
    axes=(Axis("frequency", "Frequency", "Hz", scale=1e6),),
    z=ZSpec("signal", "Signal", "V", scale=1e-3),
    result_type=SingleResult,
    cfg_type=NativeCfg,
    tag="test/single",
    data_variable=DataVariable("resonator"),
    cfg_schema_version="3.2",
)


class SingleExperiment(PersistableExperiment[SingleResult, NativeCfg]):
    AXES_SPEC: ClassVar = SINGLE


@dataclass
class GroupedResult:
    frequency: np.ndarray
    signal: np.ndarray
    delay: np.ndarray
    decay: np.ndarray


def build_grouped(data: GroupedLoadData[NativeCfg]) -> GroupedResult:
    """Rebuild the fake Result using each variable's memory-unit arrays."""
    resonator, decay = data.variable("resonator"), data.variable("decay")
    return GroupedResult(resonator.axes[0], resonator.z, decay.axes[0], decay.z)


GROUPED = GroupedAxesSpec(
    variables=(
        VariableSpec(
            "resonator",
            (VariableAxisSpec("Frequency", "Hz", "frequency", scale=1e6),),
            VariableZSpec("signal", "Signal", "V", scale=1e-3, dtype=np.complex128),
        ),
        VariableSpec(
            "decay",
            (VariableAxisSpec("Delay", "s", "delay", scale=1e-6),),
            VariableZSpec("decay", "Amplitude", "", dtype=np.float64),
        ),
    ),
    result_type=GroupedResult,
    cfg_type=NativeCfg,
    tag="test/grouped",
    result_builder=build_grouped,
    cfg_schema_version="3.2",
)


class GroupedExperiment(PersistableExperiment[GroupedResult, NativeCfg]):
    AXES_SPEC: ClassVar = GROUPED


@pytest.mark.parametrize("scale", [0.0, float("nan"), float("inf")])
def test_single_signal_scale_rejects_invalid_conversion(scale: float) -> None:
    with pytest.raises(ValueError, match="scale"):
        ZSpec(field_name="signal", label="signal", unit="V", scale=scale)


def test_single_instance_native_round_trip_keeps_cfg_snapshot_and_memory_units(
    tmp_path: Path,
) -> None:
    experiment = SingleExperiment()
    result = SingleResult(np.array([4000.0, 5000.0]), np.array([2 + 3j, 4 - 5j]))
    record = RunRecord(cfg=NativeCfg(name="original"), result=result)
    metadata = native_metadata(SINGLE.tag)
    path = tmp_path / "unrelated-name.h5"
    experiment.save_run(record, path, metadata=metadata)
    loaded, snapshot = experiment.load_run(path)
    assert loaded.cfg == record.cfg
    assert snapshot == metadata.snapshot
    np.testing.assert_array_equal(loaded.result.frequency, result.frequency)
    np.testing.assert_allclose(loaded.result.signal, result.signal, rtol=1e-15)
    stored = load_run_data(path)
    validate_experiment_payload(stored.payload, schema=native_schemas(SINGLE))
    assert stored.cfg.cfg_type == "NativeCfg"
    assert stored.cfg.schema_version == "3.2"
    assert tuple(stored.payload.variables) == (DataVariable("resonator"),)
    wire = stored.payload.variables[DataVariable("resonator")]
    np.testing.assert_array_equal(wire.axes[0].values, result.frequency * 1e6)
    np.testing.assert_allclose(wire.z, result.signal * 1e-3)
    assert stored.extensions is None


def test_grouped_instance_uses_shared_native_entry_for_different_grids(
    tmp_path: Path,
) -> None:
    result = GroupedResult(
        np.array([4000.0, 5000.0]),
        np.array([2 + 3j, 4 - 5j]),
        np.array([1.0, 2.0, 3.0]),
        np.array([0.5, 0.3, 0.1]),
    )
    record = RunRecord(cfg=NativeCfg(), result=result)
    path = tmp_path / "grouped.h5"
    experiment = GroupedExperiment()
    experiment.save_run(record, path, metadata=native_metadata(GROUPED.tag))
    loaded, snapshot = experiment.load_run(path)
    assert loaded.cfg == record.cfg
    assert snapshot == native_metadata(GROUPED.tag).snapshot
    for field in ("frequency", "signal", "delay", "decay"):
        np.testing.assert_allclose(
            getattr(loaded.result, field), getattr(result, field), rtol=1e-15
        )
    stored = load_run_data(path)
    validate_experiment_payload(stored.payload, schema=native_schemas(GROUPED))
    assert stored.payload.representation == "grouped"
    assert tuple(stored.payload.variables) == GROUPED.required_variables
    np.testing.assert_allclose(
        stored.payload.variables[DataVariable("decay")].axes[0].values,
        result.delay * 1e-6,
    )


def test_single_member_grouped_is_not_coerced_to_single(tmp_path: Path) -> None:
    spec = GroupedAxesSpec(
        variables=(GROUPED.variables[0],),
        result_type=SingleResult,
        cfg_type=NativeCfg,
        tag="test/one-group",
        result_builder=lambda data: SingleResult(
            data.variable("resonator").axes[0], data.variable("resonator").z
        ),
    )
    record = RunRecord(
        cfg=NativeCfg(),
        result=SingleResult(np.array([4000.0, 5000.0]), np.array([2 + 3j, 4 - 5j])),
    )
    path = tmp_path / "one-group.h5"
    save_run(record, path, spec=spec, metadata=native_metadata(spec.tag))
    loaded, _ = load_run(path, spec=spec)
    assert load_run_data(path).payload.representation == "grouped"
    np.testing.assert_allclose(loaded.result.signal, record.result.signal)


@pytest.mark.parametrize("cfg", [None, NativeCfg()])
def test_save_rejects_missing_cfg_or_wrong_tag_without_creating_file(
    tmp_path: Path, cfg: NativeCfg | None
) -> None:
    record = RunRecord(cfg=cfg, result=SingleResult(np.array([4000.0]), np.array([1j])))
    with pytest.raises(ValueError, match="cfg|experiment"):
        save_run(
            record,
            tmp_path / "invalid.h5",
            spec=SINGLE,
            metadata=native_metadata("wrong/tag"),
        )
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("version", ["", "1", "1.2.3", "one.two", "-1.0"])
def test_specs_reject_invalid_cfg_schema_versions_at_declaration(version: str) -> None:
    with pytest.raises(ValueError, match="version"):
        replace(SINGLE, cfg_schema_version=version)
    with pytest.raises(ValueError, match="version"):
        replace(GROUPED, cfg_schema_version=version)


@pytest.mark.parametrize(
    "attribute,value",
    [
        ("experiment", "unknown/tag"),
        ("cfg_type", "UnknownCfg"),
        ("cfg_schema_version", "4.0"),
    ],
)
def test_typed_load_rejects_wrong_identity_and_cfg_major(
    tmp_path: Path, attribute: str, value: str
) -> None:
    path = tmp_path / "wrong.h5"
    record = RunRecord(
        cfg=NativeCfg(), result=SingleResult(np.array([4000.0]), np.array([1j]))
    )
    save_run(record, path, spec=SINGLE, metadata=native_metadata(SINGLE.tag))
    with h5.File(path, "r+") as file:
        location = "/" if attribute == "experiment" else "/cfg"
        file[location].attrs[attribute] = value
    with pytest.raises(ValueError, match=attribute) as error:
        load_run(path, spec=SINGLE)
    assert str(path) in str(error.value)
    assert location in str(error.value)


@pytest.mark.parametrize("forbidden", [False, True])
def test_same_major_nested_unknown_cfg_is_projected_but_before_validator_sees_raw(
    tmp_path: Path, forbidden: bool
) -> None:
    path = tmp_path / "future.h5"
    record = RunRecord(
        cfg=NativeCfg(), result=SingleResult(np.array([4000.0]), np.array([1j]))
    )
    save_run(record, path, spec=SINGLE, metadata=native_metadata(SINGLE.tag))
    with h5.File(path, "r+") as file:
        file.attrs["format_version"] = "2.7"
        file["cfg"].attrs["cfg_schema_version"] = "3.9"
        raw = json.loads(hdf_json_text(file, "cfg"))
        raw["future"] = {"unknown": [1, None]}
        raw["nested"]["forbidden_future" if forbidden else "future"] = 9
        hdf_dataset(file, "cfg")[()] = json.dumps(raw)
    assert load_run_data(path).cfg.values == raw
    if forbidden:
        with pytest.raises(ValidationError, match="before validator observed"):
            load_run(path, spec=SINGLE)
    else:
        loaded, snapshot = load_run(path, spec=SINGLE)
        assert loaded.cfg == record.cfg
        assert snapshot == native_metadata(SINGLE.tag).snapshot


def test_shared_base_grouped_labber_entry_remains_separate_from_native(
    tmp_path: Path,
) -> None:
    spec = GroupedAxesSpec(
        variables=(GROUPED.variables[0],),
        result_type=SingleResult,
        cfg_type=NativeCfg,
        tag="test/one-group",
        result_builder=lambda data: SingleResult(
            data.variable("resonator").axes[0], data.variable("resonator").z
        ),
    )

    class OneGroupedExperiment(PersistableExperiment[SingleResult, NativeCfg]):
        AXES_SPEC: ClassVar = spec

    record = RunRecord(
        cfg=NativeCfg(),
        result=SingleResult(np.array([4000.0, 5000.0]), np.array([2 + 3j, 4 - 5j])),
    )
    experiment = OneGroupedExperiment()
    path = tmp_path / "grouped.hdf5"
    experiment.save(record, path)
    loaded = experiment.load(path)
    assert loaded.cfg == record.cfg
    np.testing.assert_array_equal(loaded.result.frequency, record.result.frequency)
    np.testing.assert_allclose(loaded.result.signal, record.result.signal)
    assert tuple(load_grouped_labber_data(str(path)).variables) == (
        DataVariable("resonator"),
    )


def test_prepare_single_data_feeds_both_formats_with_same_cfg_and_comment(
    tmp_path: Path,
) -> None:
    record = RunRecord(
        cfg=NativeCfg(name="captured"),
        result=SingleResult(np.array([4000.0, 5000.0]), np.array([2 + 3j, 4 - 5j])),
    )
    comment = "量測結果\nKeep the original text."
    prepared = prepare_run_data(record, spec=SINGLE, comment=comment)
    assert prepared.payload.representation == "single"
    assert prepared.payload.metadata.comment == comment
    assert prepared.payload.metadata.tags == [SINGLE.tag]
    assert prepared.cfg.cfg_type == "NativeCfg"
    assert prepared.cfg.schema_version == SINGLE.cfg_schema_version
    assert record.cfg is not None
    assert prepared.cfg.values == record.cfg.model_dump(mode="json")
    signal = prepared.payload.variables[DataVariable("resonator")]
    np.testing.assert_array_equal(signal.axes[0].values, record.result.frequency * 1e6)
    np.testing.assert_allclose(signal.z, record.result.signal * 1e-3)
    assert list(tmp_path.iterdir()) == []
    native, labber = tmp_path / "native.h5", tmp_path / "labber.h5"
    save_run_data(
        native, prepared.payload, native_metadata(SINGLE.tag), cfg=prepared.cfg
    )
    write_labber(labber, prepared.payload, cfg=prepared.cfg)
    stored = load_run_data(native)
    decoded = decode_labber_comment(load_labber_data(str(labber)).metadata.comment)
    assert stored.cfg == prepared.cfg
    assert stored.payload.metadata.comment == comment
    assert decoded.cfg == prepared.cfg.values
    assert decoded.comment == comment
    record.cfg.name = "later"
    record.cfg.nested.count = 99
    assert prepared.cfg.values["name"] == "captured"
    nested = prepared.cfg.values["nested"]
    assert isinstance(nested, dict)
    assert nested["count"] == 3


def test_prepared_envelope_isolates_mappings_without_copying_arrays() -> None:
    record = RunRecord(
        cfg=NativeCfg(), result=SingleResult(np.array([4000.0]), np.array([1j]))
    )
    original = prepare_run_data(record, spec=SINGLE, comment="original")
    variables = dict(original.payload.variables)
    payload = replace(original.payload, variables=variables)
    prepared = PreparedRunData(payload=payload, cfg=original.cfg)
    variables.clear()
    payload.metadata.tags.append("later")
    payload.metadata.comment = "later"
    original.cfg.values["name"] = "later"
    nested = original.cfg.values["nested"]
    assert isinstance(nested, dict)
    nested["count"] = 99
    assert tuple(prepared.payload.variables) == (DataVariable("resonator"),)
    assert prepared.payload.metadata.tags == [SINGLE.tag]
    assert prepared.payload.metadata.comment == "original"
    assert prepared.cfg.values["name"] == "historical"
    prepared_nested = prepared.cfg.values["nested"]
    assert isinstance(prepared_nested, dict)
    assert prepared_nested["count"] == 3
    assert (
        prepared.payload.variables[DataVariable("resonator")].z
        is original.payload.variables[DataVariable("resonator")].z
    )


def test_prepare_grouped_data_keeps_heterogeneous_grids_and_validator() -> None:
    record = RunRecord(
        cfg=NativeCfg(),
        result=GroupedResult(
            np.array([4000.0, 5000.0]),
            np.array([1j, 2j]),
            np.array([1.0, 2.0, 3.0]),
            np.array([0.5, 0.3, 0.1]),
        ),
    )
    prepared = prepare_run_data(record, spec=GROUPED, comment="grouped")
    assert prepared.payload.representation == "grouped"
    assert tuple(prepared.payload.variables) == GROUPED.required_variables
    decay = prepared.payload.variables[DataVariable("decay")]
    np.testing.assert_allclose(decay.axes[0].values, record.result.delay * 1e-6)
    np.testing.assert_allclose(decay.z, record.result.decay)

    def reject_result(result: GroupedResult) -> None:
        raise ValueError("rejected result")

    with pytest.raises(ValueError, match="rejected result"):
        prepare_run_data(
            record, spec=replace(GROUPED, result_validator=reject_result), comment=""
        )


@pytest.mark.parametrize(
    ("cfg", "result", "error", "message"),
    [
        (None, SingleResult(np.array([4000.0]), np.array([1j])), ValueError, "cfg"),
        (
            ExpCfgModel(),
            SingleResult(np.array([4000.0]), np.array([1j])),
            TypeError,
            "cfg",
        ),
        (
            NativeCfg(),
            GroupedResult(
                np.array([1.0]), np.array([1j]), np.array([1.0]), np.array([1.0])
            ),
            TypeError,
            "Result",
        ),
        (
            NativeCfg(),
            SingleResult(np.array([4000.0, 5000.0]), np.array([1j])),
            ValueError,
            "shape|axis|length",
        ),
    ],
)
def test_prepare_rejects_invalid_record(
    cfg: NativeCfg | None,
    result: SingleResult,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        prepare_run_data(RunRecord(cfg=cfg, result=result), spec=SINGLE, comment="")


@pytest.mark.parametrize("comment", [None, 3, b"bytes"])
def test_prepare_rejects_non_text_comment(comment: str) -> None:
    record = RunRecord(
        cfg=NativeCfg(), result=SingleResult(np.array([4000.0]), np.array([1j]))
    )
    with pytest.raises(ValidationError, match="string"):
        prepare_run_data(record, spec=SINGLE, comment=comment)


def test_prepare_scalar_native_data_does_not_add_labber_restrictions(
    tmp_path: Path,
) -> None:
    @dataclass
    class ScalarResult:
        signal: np.ndarray

    spec = AxesSpec(
        axes=(),
        z=SINGLE.z,
        result_type=ScalarResult,
        cfg_type=NativeCfg,
        tag="test/scalar",
    )
    record = RunRecord(cfg=NativeCfg(), result=ScalarResult(np.array(1j)))
    prepared = prepare_run_data(record, spec=spec, comment="scalar")
    native, labber = tmp_path / "scalar.h5", tmp_path / "no-axis-labber.h5"
    save_run_data(native, prepared.payload, native_metadata(spec.tag), cfg=prepared.cfg)
    loaded, metadata = load_run_with_metadata(native, spec=spec)
    np.testing.assert_allclose(loaded.result.signal, record.result.signal)
    assert metadata == native_metadata(spec.tag)
    with pytest.raises(ValueError, match="step axis"):
        write_labber(labber, prepared.payload, cfg=prepared.cfg)
    assert not labber.exists()


@pytest.mark.parametrize("completion", ["complete", "partial", "stopped"])
def test_full_metadata_load_preserves_historical_evidence(
    tmp_path: Path,
    completion: Literal["complete", "partial", "stopped"],
) -> None:
    record = RunRecord(
        cfg=NativeCfg(), result=SingleResult(np.array([4000.0]), np.array([1j]))
    )
    metadata = replace(native_metadata(SINGLE.tag), completion=completion)
    path = tmp_path / "historical.h5"
    save_run(record, path, spec=SINGLE, metadata=metadata)
    loaded, restored = load_run_with_metadata(path, spec=SINGLE)
    old_loaded, snapshot = load_run(path, spec=SINGLE)
    assert restored == metadata
    assert snapshot == restored.snapshot
    assert loaded.cfg == old_loaded.cfg == record.cfg
    np.testing.assert_allclose(loaded.result.signal, old_loaded.result.signal)
    with pytest.raises(ValueError, match="experiment"):
        load_run_with_metadata(path, spec=replace(SINGLE, tag="wrong"))


def test_public_persistence_declaration_supports_single_grouped_and_missing() -> None:
    assert SingleExperiment().persistence_spec() is SINGLE
    assert GroupedExperiment().persistence_spec() is GROUPED
    missing = PersistableExperiment[SingleResult, NativeCfg]()
    with pytest.raises(NotImplementedError, match="AXES_SPEC"):
        missing.persistence_spec()


def test_typed_load_selects_declared_variables_and_ignores_extra_variable(
    tmp_path: Path,
) -> None:
    path = tmp_path / "extra.h5"
    record = RunRecord(
        cfg=NativeCfg(), result=SingleResult(np.array([4000.0]), np.array([1j]))
    )
    save_run(record, path, spec=SINGLE, metadata=native_metadata(SINGLE.tag))
    stored = load_run_data(path)
    variables = dict(stored.payload.variables)
    variables[DataVariable("extra")] = next(iter(variables.values()))
    payload = replace(stored.payload, variables=variables, representation="grouped")
    save_run_data(path, payload, stored.metadata, cfg=stored.cfg, replace=True)
    loaded, _ = load_run(path, spec=SINGLE)
    np.testing.assert_array_equal(loaded.result.signal, record.result.signal)
    missing = replace(
        stored.payload,
        variables={DataVariable("extra"): next(iter(variables.values()))},
    )
    save_run_data(path, missing, stored.metadata, cfg=stored.cfg, replace=True)
    with pytest.raises(ValueError, match="resonator"):
        load_run(path, spec=SINGLE)
