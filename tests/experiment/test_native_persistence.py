"""Typed native mapping and data-first configuration restoration contracts."""

import json
import warnings
from copy import deepcopy
from dataclasses import dataclass, replace
from pathlib import Path
from typing import ClassVar, Literal

import numpy as np
import pytest
from pydantic import BaseModel, ConfigDict, ValidationError, model_validator
from zcu_tools.cfg_model import ConfigBase
from zcu_tools.datafile import (
    Axis as DiskAxis,
)
from zcu_tools.datafile import (
    Container,
    DataVariable,
    encode_labber_comment,
    load,
    load_labber_data,
    save,
    write_labber,
)
from zcu_tools.experiment import (
    AxesSpec,
    Axis,
    CfgUnavailableWarning,
    Experiment,
    GroupedAxesSpec,
    GroupedLoadData,
    PreparedRunData,
    RunRecord,
    VariableAxisSpec,
    VariableSpec,
    VariableZSpec,
    ZSpec,
    load_run,
    load_run_with_metadata,
    prepare_run_data,
    restore_run,
    save_run,
)

from tests._native_support import native_metadata


class NestedCfg(BaseModel):
    model_config = ConfigDict(extra="forbid")
    count: int = 3

    @model_validator(mode="before")
    @classmethod
    def reject_forbidden_raw_key(cls, data: object) -> object:
        if isinstance(data, dict) and "forbidden_future" in data:
            raise ValueError("before validator observed forbidden_future")
        return data


class NativeCfg(ConfigBase):
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


def test_axes_namespace_maps_memory_units_to_native_values():
    spec = AxesSpec(
        axes=(AxesSpec.Axis("frequency", "Frequency", "Hz", scale=1e6),),
        z=AxesSpec.Z("signal", "Signal", "V", scale=1e-3),
        result_type=SingleResult,
        cfg_type=NativeCfg,
        tag="test/namespace",
    )
    record = RunRecord(
        cfg=NativeCfg(),
        result=SingleResult(
            frequency=np.array([100.0, 200.0]),
            signal=np.array([1.0 + 2.0j, 3.0 + 4.0j]),
        ),
    )
    prepared = prepare_run_data(record, spec=spec, comment="namespace")
    variable = prepared.container.vars["data"]
    np.testing.assert_array_equal(variable.axes[0].values, [100e6, 200e6])
    np.testing.assert_allclose(variable.values, [0.001 + 0.002j, 0.003 + 0.004j])


class SingleExperiment(Experiment[SingleResult, NativeCfg]):
    AXES_SPEC: ClassVar[AxesSpec[SingleResult, NativeCfg]] = SINGLE


@dataclass
class GroupedResult:
    frequency: np.ndarray
    signal: np.ndarray
    delay: np.ndarray
    decay: np.ndarray


def build_grouped(data: GroupedLoadData[NativeCfg]) -> GroupedResult:
    """Build each field from its declared variable's independent grid."""
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


class GroupedExperiment(Experiment[GroupedResult, NativeCfg]):
    AXES_SPEC: ClassVar[GroupedAxesSpec[GroupedResult, NativeCfg]] = GROUPED


def single_record() -> RunRecord[NativeCfg, SingleResult]:
    """Return fixed asymmetric complex values in memory units."""
    return RunRecord(
        NativeCfg(),
        SingleResult(np.array([4000.0, 5000.0]), np.array([2 + 3j, 4 - 5j])),
        metadata=native_metadata(SINGLE.tag),
    )


def cfg_update(container: Container, **changes: object) -> Container:
    """Return an isolated container with explicit cfg document defects."""
    docs = deepcopy(dict(container.docs))
    for key, value in changes.items():
        if key == "cfg_type" or key == "schema_version":
            assert isinstance(value, str)
            docs["cfg"][key] = value
        elif key == "values":
            assert isinstance(value, dict)
            docs["cfg"][key] = value
        else:
            raise ValueError(key)
    return replace(container, docs=docs)


@pytest.mark.parametrize("scale", [0.0, float("nan"), float("inf")])
def test_single_signal_scale_rejects_invalid_conversion(scale: float) -> None:
    with pytest.raises(ValueError, match="scale"):
        ZSpec("signal", "signal", "V", scale=scale)


def test_single_instance_native_round_trip_keeps_cfg_snapshot_and_memory_units(
    tmp_path: Path,
) -> None:
    record = single_record()
    path = tmp_path / "unrelated-name.h5"
    experiment = SingleExperiment()
    save_run(record, path, spec=experiment.AXES_SPEC)
    loaded, snapshot = load_run(path, spec=experiment.AXES_SPEC)
    assert loaded.cfg == record.cfg
    assert loaded.metadata == record.metadata
    assert record.metadata is not None
    assert snapshot == record.metadata.snapshot
    np.testing.assert_array_equal(loaded.result.frequency, record.result.frequency)
    np.testing.assert_allclose(loaded.result.signal, record.result.signal)
    stored = load(path)
    assert stored.docs["cfg"] == {
        "values": {"nested": {"count": 3}, "name": "historical"},
        "cfg_type": "NativeCfg",
        "schema_version": "3.2",
    }
    assert tuple(stored.vars) == ("resonator",)
    np.testing.assert_array_equal(stored.vars["resonator"].axes[0].values, [4e9, 5e9])
    np.testing.assert_allclose(
        stored.vars["resonator"].values, [0.002 + 0.003j, 0.004 - 0.005j]
    )


def test_grouped_instance_uses_shared_native_entry_for_different_grids(
    tmp_path: Path,
) -> None:
    result = GroupedResult(
        np.array([4000.0, 5000.0]),
        np.array([2 + 3j, 4 - 5j]),
        np.array([1.0, 2.0, 3.0]),
        np.array([0.5, 0.3, 0.1]),
    )
    record = RunRecord(NativeCfg(), result, metadata=native_metadata(GROUPED.tag))
    path = tmp_path / "grouped.h5"
    experiment = GroupedExperiment()
    save_run(record, path, spec=experiment.AXES_SPEC)
    loaded = load_run_with_metadata(path, spec=experiment.AXES_SPEC)
    assert loaded.cfg == record.cfg
    for field in ("frequency", "signal", "delay", "decay"):
        np.testing.assert_allclose(
            getattr(loaded.result, field), getattr(result, field)
        )
    stored = load(path)
    assert set(stored.vars) == {"resonator", "decay"}
    np.testing.assert_allclose(stored.vars["decay"].axes[0].values, [1e-6, 2e-6, 3e-6])


@pytest.mark.parametrize("cfg", [None, NativeCfg()])
def test_save_rejects_missing_cfg_or_wrong_tag_without_creating_file(
    tmp_path: Path, cfg: NativeCfg | None
) -> None:
    record = replace(single_record(), cfg=cfg, metadata=native_metadata("wrong/tag"))
    with pytest.raises(ValueError, match="cfg|experiment"):
        save_run(record, tmp_path / "invalid.h5", spec=SINGLE)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("version", ["", "1", "1.2.3", "one.two", "-1.0"])
def test_specs_reject_invalid_cfg_schema_versions_at_declaration(version: str) -> None:
    with pytest.raises(ValueError, match="version"):
        replace(SINGLE, cfg_schema_version=version)
    with pytest.raises(ValueError, match="version"):
        replace(GROUPED, cfg_schema_version=version)


@pytest.mark.parametrize("forbidden", [False, True])
def test_same_major_nested_unknown_cfg_is_projected_but_before_validator_sees_raw(
    tmp_path: Path, forbidden: bool
) -> None:
    record = single_record()
    prepared = prepare_run_data(record, spec=SINGLE, comment="")
    assert record.metadata is not None
    container = prepared.to_container(record.metadata)
    raw = {
        "nested": {"count": 3, "forbidden_future" if forbidden else "future": 9},
        "name": "historical",
        "future": {"unknown": [1, None]},
    }
    container = cfg_update(container, values=raw, schema_version="3.9")
    path = tmp_path / "future.h5"
    save(path, replace(container, attrs=replace(container.attrs, format_version="3.7")))
    if forbidden:
        with pytest.warns(CfgUnavailableWarning, match="before validator observed"):
            loaded = load_run_with_metadata(path, spec=SINGLE)
        assert loaded.cfg is None
    else:
        loaded = load_run_with_metadata(path, spec=SINGLE)
        assert loaded.cfg == record.cfg
    assert load(path).docs["cfg"]["values"] == raw
    np.testing.assert_allclose(loaded.result.signal, record.result.signal)


def test_prepare_single_data_feeds_both_formats_with_same_cfg_and_comment(
    tmp_path: Path,
) -> None:
    record = single_record()
    comment = "量測結果\nKeep the original text."
    prepared = prepare_run_data(record, spec=SINGLE, comment=comment)
    assert prepared.labber_metadata.comment == comment
    assert prepared.labber_metadata.tags == [SINGLE.tag]
    assert prepared.cfg.cfg_type == "NativeCfg"
    assert prepared.cfg.schema_version == "3.2"
    assert record.cfg is not None and record.metadata is not None
    assert prepared.cfg.values == record.cfg.model_dump(mode="json")
    assert list(tmp_path.iterdir()) == []
    native, labber = tmp_path / "native.h5", tmp_path / "labber.h5"
    save(native, prepared.to_container(record.metadata))
    write_labber(
        labber,
        prepared.container,
        metadata=replace(
            prepared.labber_metadata,
            comment=encode_labber_comment(prepared.cfg.values, comment),
        ),
    )
    decoded = json.loads(load_labber_data(str(labber)).metadata.comment)
    assert load(native).docs["cfg"]["values"] == decoded["cfg"] == prepared.cfg.values
    assert decoded["comment"] == comment
    record.cfg.nested.count = 99
    assert prepared.cfg.values["nested"] == {"count": 3}


def test_prepared_envelope_isolates_mappings_without_copying_arrays() -> None:
    original = prepare_run_data(single_record(), spec=SINGLE, comment="original")
    variables = dict(original.container.vars)
    prepared = PreparedRunData(
        replace(original.container, vars=variables),
        original.cfg,
        original.labber_metadata,
    )
    variables.clear()
    original.labber_metadata.tags.append("later")
    original.cfg.values["name"] = "later"
    assert tuple(prepared.container.vars) == ("resonator",)
    assert prepared.labber_metadata.tags == [SINGLE.tag]
    assert prepared.cfg.values["name"] == "historical"
    assert (
        prepared.container.vars["resonator"].values
        is original.container.vars["resonator"].values
    )


@pytest.mark.parametrize("defect", ["cfg", "shape", "validator"])
def test_prepare_rejects_invalid_record(defect: str) -> None:
    record = single_record()
    if defect == "cfg":
        record = replace(record, cfg=None)
    elif defect == "shape":
        record.result.signal = np.array([1j])
    else:

        def reject(result: SingleResult) -> None:
            raise ValueError("rejected result")

        spec = GroupedAxesSpec(
            (GROUPED.variables[0],),
            SingleResult,
            NativeCfg,
            SINGLE.tag,
            lambda data: SingleResult(
                data.variable("resonator").axes[0], data.variable("resonator").z
            ),
            result_validator=reject,
        )
        with pytest.raises(ValueError, match="rejected result"):
            prepare_run_data(record, spec=spec, comment="")
        return
    with pytest.raises((ValueError, TypeError)):
        prepare_run_data(record, spec=SINGLE, comment="")


@pytest.mark.parametrize("cfg", [ConfigBase()])
def test_prepare_rejects_wrong_cfg_type(cfg: NativeCfg) -> None:
    with pytest.raises(TypeError, match="cfg"):
        prepare_run_data(replace(single_record(), cfg=cfg), spec=SINGLE, comment="")


@pytest.mark.parametrize("result", [object()])
def test_prepare_rejects_wrong_result_type(result: SingleResult) -> None:
    with pytest.raises(TypeError, match="Result"):
        prepare_run_data(RunRecord(NativeCfg(), result), spec=SINGLE, comment="")


@pytest.mark.parametrize("comment", [None, 3, b"bytes"])
def test_prepare_rejects_non_text_comment(comment: str) -> None:
    with pytest.raises(ValidationError, match="string"):
        prepare_run_data(single_record(), spec=SINGLE, comment=comment)


def test_prepare_scalar_native_data_does_not_add_labber_restrictions(
    tmp_path: Path,
) -> None:
    @dataclass
    class ScalarResult:
        signal: np.ndarray

    spec = AxesSpec((), SINGLE.z, ScalarResult, NativeCfg, "test/scalar")
    record = RunRecord(
        NativeCfg(), ScalarResult(np.array(1j)), metadata=native_metadata(spec.tag)
    )
    path = tmp_path / "scalar.h5"
    save_run(record, path, spec=spec)
    loaded = load_run_with_metadata(path, spec=spec)
    np.testing.assert_allclose(loaded.result.signal, 1j)
    prepared = prepare_run_data(record, spec=spec, comment="scalar")
    with pytest.raises(ValueError, match="step axis"):
        write_labber(tmp_path / "no-axis.h5", prepared.container)


@pytest.mark.parametrize("started_at", [None, "2026-10-06T01:00:00Z"])
@pytest.mark.parametrize("completion", ["complete", "partial", "stopped", "unknown"])
def test_full_metadata_load_preserves_historical_evidence(
    tmp_path: Path,
    completion: Literal["complete", "partial", "stopped", "unknown"],
    started_at: str | None,
) -> None:
    metadata = native_metadata(SINGLE.tag).model_copy(
        update={"completion": completion, "started_at": started_at}
    )
    record = replace(single_record(), metadata=metadata)
    path = tmp_path / "historical.h5"
    save_run(record, path, spec=SINGLE)
    loaded = load_run_with_metadata(path, spec=SINGLE)
    old_loaded, snapshot = load_run(path, spec=SINGLE)
    assert loaded.metadata == metadata
    assert snapshot == metadata.snapshot
    assert loaded.cfg == old_loaded.cfg == record.cfg
    with pytest.raises(ValueError, match="experiment") as caught:
        load_run_with_metadata(path, spec=replace(SINGLE, tag="wrong"))
    assert str(path) in str(caught.value)


@pytest.fixture(params=[False, True], ids=["single", "grouped"])
def restorable_run(
    request: pytest.FixtureRequest,
) -> tuple[
    Container,
    AxesSpec[SingleResult, NativeCfg] | GroupedAxesSpec[SingleResult, NativeCfg],
]:
    spec = SINGLE
    if request.param:
        spec = GroupedAxesSpec(
            (GROUPED.variables[0],),
            SingleResult,
            NativeCfg,
            SINGLE.tag,
            lambda data: SingleResult(
                data.variable("resonator").axes[0], data.variable("resonator").z
            ),
            cfg_schema_version=SINGLE.cfg_schema_version,
        )
    record = single_record()
    assert record.metadata is not None
    prepared = prepare_run_data(record, spec=spec, comment="historical")
    return prepared.to_container(record.metadata), spec


@pytest.mark.parametrize("issue", [None, "cfg_type", "schema", "nested.count"])
def test_restore_keeps_data_and_raw_history_when_typed_cfg_is_unavailable(
    restorable_run: tuple[
        Container,
        AxesSpec[SingleResult, NativeCfg] | GroupedAxesSpec[SingleResult, NativeCfg],
    ],
    issue: str | None,
) -> None:
    stored, spec = restorable_run
    if issue == "cfg_type":
        stored = cfg_update(stored, cfg_type="RetiredCfg")
    elif issue == "schema":
        stored = cfg_update(stored, schema_version="4.0")
    elif issue == "nested.count":
        stored = cfg_update(stored, values={"nested": {"count": "bad"}})
    else:
        stored = cfg_update(
            stored, schema_version="3.9", values={"future": {"unknown": [1, None]}}
        )
    original_docs = deepcopy(stored.docs)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        restored = restore_run(stored, spec=spec)
    np.testing.assert_array_equal(restored.record.result.frequency, [4000.0, 5000.0])
    np.testing.assert_allclose(restored.record.result.signal, [2 + 3j, 4 - 5j])
    assert stored.docs == original_docs
    if issue is None:
        assert restored.record.cfg == NativeCfg()
        assert restored.cfg_issue is None and caught == []
    else:
        assert restored.record.cfg is None
        assert restored.cfg_issue is not None and issue in restored.cfg_issue
        assert len(caught) == 1 and caught[0].category is CfgUnavailableWarning
        assert str(caught[0].message) == restored.cfg_issue
        with pytest.raises(ValueError, match="cfg"):
            prepare_run_data(restored.record, spec=spec, comment="")


@pytest.mark.parametrize(
    "defect", ["tag", "variable", "axis", "unit", "dtype", "shape"]
)
def test_restore_rejects_bad_data_before_downgrading_cfg(
    restorable_run: tuple[
        Container,
        AxesSpec[SingleResult, NativeCfg] | GroupedAxesSpec[SingleResult, NativeCfg],
    ],
    defect: str,
) -> None:
    stored, spec = restorable_run
    stored = cfg_update(stored, cfg_type="RetiredCfg")
    wire = stored.vars["resonator"]
    if defect == "tag":
        docs = deepcopy(dict(stored.docs))
        docs["metadata"]["experiment"] = "wrong"
        stored = replace(stored, docs=docs)
    elif defect == "variable":
        stored = replace(stored, vars={})
    else:
        if defect == "axis":
            wire = replace(wire, axes=(DiskAxis("Wrong", "Hz", wire.axes[0].values),))
        elif defect == "unit":
            wire = replace(wire, unit="A")
        elif defect == "dtype":
            wire = replace(wire, values=np.array(["bad", "data"]))
        else:
            wire = replace(wire, values=np.array([1j]))
        stored = replace(stored, vars={"resonator": wire})
    with (
        warnings.catch_warnings(record=True) as caught,
        pytest.raises((ValueError, TypeError)),
    ):
        restore_run(stored, spec=spec)
    assert caught == []


def test_typed_load_selects_declared_variables_and_ignores_extra_variable(
    tmp_path: Path,
) -> None:
    path = tmp_path / "extra.h5"
    save_run(single_record(), path, spec=SINGLE)
    stored = load(path)
    variables = dict(stored.vars)
    variables["extra"] = variables["resonator"]
    save(path, replace(stored, vars=variables), replace=True)
    loaded, _ = load_run(path, spec=SINGLE)
    np.testing.assert_allclose(loaded.result.signal, single_record().result.signal)
    save(path, replace(stored, vars={"extra": variables["extra"]}), replace=True)
    with pytest.raises(ValueError, match="resonator"):
        load_run(path, spec=SINGLE)


class UnexpectedValidatorCfg(ConfigBase):
    @model_validator(mode="before")
    @classmethod
    def fail_unexpectedly(cls, _data: object) -> object:
        raise RuntimeError("unexpected typed cfg validator failure")


@pytest.mark.parametrize("rejection", ["builder", "type", "validator", "none"])
def test_result_rejection_precedes_typed_cfg_validation(
    restorable_run: tuple[
        Container,
        AxesSpec[SingleResult, NativeCfg] | GroupedAxesSpec[SingleResult, NativeCfg],
    ],
    rejection: str,
) -> None:
    stored, _ = restorable_run
    stored = cfg_update(stored, cfg_type="UnexpectedValidatorCfg")

    def build(data: GroupedLoadData[UnexpectedValidatorCfg]) -> SingleResult:
        if rejection == "builder":
            raise ValueError("builder rejected data")
        return SingleResult(
            data.variable("resonator").axes[0], data.variable("resonator").z
        )

    def validate(result: SingleResult) -> None:
        if rejection == "validator":
            raise ValueError("validator rejected data")

    @dataclass
    class OtherResult(SingleResult):
        pass

    spec = GroupedAxesSpec(
        (GROUPED.variables[0],),
        OtherResult if rejection == "type" else SingleResult,
        UnexpectedValidatorCfg,
        SINGLE.tag,
        build,
        result_validator=validate,
        cfg_schema_version="3.2",
    )
    error, match = (
        (RuntimeError, "unexpected typed cfg")
        if rejection == "none"
        else (
            (TypeError, "builder must return")
            if rejection == "type"
            else (ValueError, "rejected data")
        )
    )
    with pytest.raises(error, match=match):
        restore_run(stored, spec=spec)


@pytest.mark.parametrize("issue", ["cfg_type", "schema", "nested.count"])
def test_native_wrappers_share_cfg_fallback_and_keep_history(
    tmp_path: Path,
    restorable_run: tuple[
        Container,
        AxesSpec[SingleResult, NativeCfg] | GroupedAxesSpec[SingleResult, NativeCfg],
    ],
    issue: str,
) -> None:
    stored, spec = restorable_run
    if issue == "cfg_type":
        stored = cfg_update(stored, cfg_type="RetiredCfg")
    elif issue == "schema":
        stored = cfg_update(stored, schema_version="4.0")
    else:
        stored = cfg_update(stored, values={"nested": {"count": "bad"}})
    path = tmp_path / "history.h5"
    save(path, stored)
    with pytest.warns(CfgUnavailableWarning, match=issue):
        record = load_run_with_metadata(path, spec=spec)
    with pytest.warns(CfgUnavailableWarning, match=issue):
        old_record, snapshot = load_run(path, spec=spec)
    assert record.metadata is not None
    assert record.metadata == old_record.metadata
    assert record.metadata.model_dump(mode="json") == stored.docs["metadata"]
    assert snapshot == record.metadata.snapshot
    assert record.cfg is old_record.cfg is None
    np.testing.assert_allclose(record.result.signal, [2 + 3j, 4 - 5j])
    np.testing.assert_array_equal(record.result.signal, old_record.result.signal)
    assert load(path).docs == stored.docs


def test_restore_run_returns_the_record_and_cfg_diagnostic(
    restorable_run: tuple[
        Container,
        AxesSpec[SingleResult, NativeCfg] | GroupedAxesSpec[SingleResult, NativeCfg],
    ],
) -> None:
    stored, spec = restorable_run

    restored = restore_run(stored, spec=spec)
    assert restored.cfg_issue is None
    assert restored.record.cfg == NativeCfg()
    assert restored.record.metadata is not None
    assert restored.record.metadata.model_dump(mode="json") == stored.docs["metadata"]
    np.testing.assert_array_equal(restored.record.result.frequency, [4000.0, 5000.0])
    np.testing.assert_allclose(restored.record.result.signal, [2 + 3j, 4 - 5j])


def test_single_result_rejection_precedes_typed_cfg_validation(
    restorable_run: tuple[
        Container,
        AxesSpec[SingleResult, NativeCfg] | GroupedAxesSpec[SingleResult, NativeCfg],
    ],
) -> None:
    stored, _ = restorable_run
    stored = cfg_update(stored, cfg_type="UnexpectedValidatorCfg")

    @dataclass
    class RejectedResult(SingleResult):
        def __post_init__(self) -> None:
            raise ValueError("single constructor rejected data")

    spec = AxesSpec(
        SINGLE.axes,
        SINGLE.z,
        RejectedResult,
        UnexpectedValidatorCfg,
        SINGLE.tag,
        data_variable=SINGLE.data_variable,
        cfg_schema_version="3.2",
    )
    with pytest.raises(ValueError, match="single constructor rejected data"):
        restore_run(stored, spec=spec)


@pytest.mark.parametrize("defect", ["missing", "values", "envelope"])
def test_native_load_rejects_malformed_cfg_instead_of_falling_back(
    tmp_path: Path, defect: str
) -> None:
    path = tmp_path / "malformed.h5"
    save_run(single_record(), path, spec=SINGLE)
    stored = load(path)
    docs = deepcopy(dict(stored.docs))
    if defect == "missing":
        del docs["cfg"]
    elif defect == "values":
        docs["cfg"]["values"] = []
    else:
        docs["cfg"]["schema_version"] = "invalid"
    save(path, replace(stored, docs=docs), replace=True)
    with pytest.raises(ValueError, match="cfg") as caught:
        load_run(path, spec=SINGLE)
    assert str(path) in str(caught.value)
