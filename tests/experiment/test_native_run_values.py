"""Fixed run values at the typed Native save and load boundary."""

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from pydantic import JsonValue
from zcu_tools.cfg_model import ConfigBase
from zcu_tools.datafile import DataVariable, JsonObject, load, read_doc, save
from zcu_tools.experiment import (
    AxesSpec,
    Axis,
    CfgUnavailableWarning,
    GroupedAxesSpec,
    GroupedLoadData,
    RunRecord,
    RunValueAxis,
    RunValueSpec,
    VariableAxisSpec,
    VariableSpec,
    VariableZSpec,
    ZSpec,
    load_run_with_metadata,
    save_labber,
    save_run,
)

from tests._native_support import native_metadata


class SyntheticCfg(ConfigBase):
    pass


@dataclass(frozen=True)
class AlignedResult:
    time: NDArray[np.float64]
    tone: NDArray[np.int64]
    signal: NDArray[np.float64]
    offsets: NDArray[np.float64]


def aligned_result(data: GroupedLoadData[SyntheticCfg]) -> AlignedResult:
    channel = data.variable("signal")
    time, tone = channel.axes
    return AlignedResult(
        time, tone, channel.z, data.run_values.vector("offsets", dtype=np.float64)
    )


ALIGNED = GroupedAxesSpec(
    variables=(
        VariableSpec(
            "signal",
            (
                VariableAxisSpec(field_name="time", label="Time", unit="s"),
                VariableAxisSpec(
                    field_name="tone", label="Tone", unit="", dtype=np.int64
                ),
            ),
            VariableZSpec("signal", label="Signal", unit="V"),
        ),
    ),
    result_type=AlignedResult,
    cfg_type=SyntheticCfg,
    tag="test/aligned",
    result_builder=aligned_result,
    run_values=(
        RunValueSpec(
            field_name="offsets",
            wire_name="applied_offsets",
            unit="Hz",
            scale=1e6,
            dtype=np.float64,
            allow_unknown=True,
            axis=RunValueAxis(variable=DataVariable("signal"), label="Tone"),
        ),
    ),
)


@pytest.fixture
def aligned_path(tmp_path: Path) -> Path:
    path = tmp_path / "aligned.h5"
    result = AlignedResult(
        np.array([1.0, 2.0, 3.0]),
        np.array([4, 1]),
        np.ones((2, 3)),
        np.array([0.12, np.nan]),
    )
    record = RunRecord(
        SyntheticCfg(), result, native_metadata(ALIGNED.tag).model_dump(mode="json")
    )
    save_run(record, path, spec=ALIGNED)
    return path


def test_native_run_value_vector_keeps_tone_order_units_and_unknown(
    aligned_path: Path, tmp_path: Path
) -> None:
    loaded = load_run_with_metadata(aligned_path, spec=ALIGNED)
    np.testing.assert_array_equal(loaded.result.tone, [4, 1])
    np.testing.assert_array_equal(loaded.result.offsets, [0.12, np.nan])
    assert loaded.result.offsets.dtype == np.float64

    resaved = tmp_path / "resaved.h5"
    save_run(loaded, resaved, spec=ALIGNED)
    assert read_doc(resaved, "run_values") == {
        "version": 1,
        "values": {"applied_offsets": {"unit": "Hz", "value": [120000.0, None]}},
    }


@dataclass(frozen=True)
class ScalarResult:
    x: NDArray[np.float64]
    signal: NDArray[np.float64]
    fixed: float | None = None


SCALAR = AxesSpec(
    axes=(Axis("x", label="Time", unit="s"),),
    z=ZSpec("signal", "Signal", "V", dtype=np.float64),
    result_type=ScalarResult,
    cfg_type=SyntheticCfg,
    tag="test/fixed",
    run_values=(RunValueSpec(field_name="fixed", wire_name="start", unit="s"),),
)
NULLABLE = replace(
    SCALAR, run_values=(replace(SCALAR.run_values[0], allow_unknown=True),)
)


@pytest.fixture
def scalar_path(tmp_path: Path) -> Path:
    path = tmp_path / "fixed.h5"
    record = RunRecord(
        SyntheticCfg(),
        ScalarResult(np.array([1.0, 2.0]), np.array([3.0, 4.0]), 0.25),
        native_metadata(SCALAR.tag).model_dump(mode="json"),
    )
    save_run(record, path, spec=SCALAR)
    return path


def replace_document(path: Path, document: JsonObject | None) -> None:
    stored = load(path)
    docs = {name: value for name, value in stored.docs.items() if name != "run_values"}
    if document is not None:
        docs["run_values"] = document
    save(path, replace(stored, docs=docs), replace=True)


@pytest.mark.parametrize("document", [None, {"version": 1, "values": {}}])
def test_missing_document_and_entry_share_the_declared_unknown_policy(
    scalar_path: Path, aligned_path: Path, document: JsonObject | None
) -> None:
    replace_document(scalar_path, document)
    replace_document(aligned_path, document)

    assert load_run_with_metadata(scalar_path, spec=NULLABLE).result.fixed is None
    vector = load_run_with_metadata(aligned_path, spec=ALIGNED).result.offsets
    np.testing.assert_array_equal(vector, [np.nan, np.nan])
    with pytest.raises(
        ValueError, match=r"fixed\.h5: /docs/run_values/values/start: required.*missing"
    ):
        load_run_with_metadata(scalar_path, spec=SCALAR)


@pytest.mark.parametrize("version", [True, 1.0, "1", 2, None])
def test_invalid_version_is_not_a_missing_document(
    scalar_path: Path, version: JsonValue
) -> None:
    document: JsonObject = {"values": {}}
    if version is not None:
        document["version"] = version
    replace_document(scalar_path, document)

    with pytest.raises(ValueError, match=r"fixed\.h5: /docs/run_values/version"):
        load_run_with_metadata(scalar_path, spec=NULLABLE)


@pytest.mark.parametrize(
    ("document", "suffix"),
    [
        ({"version": 1, "values": []}, "values"),
        ({"version": 1, "values": {"start": None}}, "values/start"),
        ({"version": 1, "values": {"start": {"value": 0.25}}}, "values/start/unit"),
        (
            {"version": 1, "values": {"start": {"unit": "us", "value": 0.25}}},
            "values/start/unit",
        ),
        ({"version": 1, "values": {"start": {"unit": "s"}}}, "values/start/value"),
        (
            {"version": 1, "values": {"start": {"unit": "s", "value": True}}},
            "values/start/value",
        ),
        (
            {"version": 1, "values": {"start": {"unit": "s", "value": "0.25"}}},
            "values/start/value",
        ),
        (
            {"version": 1, "values": {"start": {"unit": "s", "value": [0.25]}}},
            "values/start/value",
        ),
    ],
)
def test_malformed_declared_scalar_is_not_unknown(
    scalar_path: Path, document: JsonObject, suffix: str
) -> None:
    replace_document(scalar_path, document)
    with pytest.raises(ValueError, match=rf"fixed\.h5: /docs/run_values/{suffix}:"):
        load_run_with_metadata(scalar_path, spec=NULLABLE)


@pytest.mark.parametrize("value", [None, [0.25], [[0.25], [None]], [False, None]])
def test_malformed_vector_is_not_replaced_with_unknown(
    aligned_path: Path, value: JsonValue
) -> None:
    replace_document(
        aligned_path,
        {"version": 1, "values": {"applied_offsets": {"unit": "Hz", "value": value}}},
    )
    with pytest.raises(
        ValueError, match=r"/docs/run_values/values/applied_offsets/value"
    ):
        load_run_with_metadata(aligned_path, spec=ALIGNED)


def test_typed_resave_writes_known_entries_only(scalar_path: Path) -> None:
    replace_document(
        scalar_path,
        {
            "version": 1,
            "values": {
                "start": {"unit": "s", "value": None, "origin": 4},
                "later_value": "unknown to this spec",
            },
            "notes": "forward content",
        },
    )
    with pytest.raises(ValueError, match="/docs/run_values/values/start/value"):
        load_run_with_metadata(scalar_path, spec=SCALAR)
    record = load_run_with_metadata(scalar_path, spec=NULLABLE)
    assert record.result.fixed is None

    save_run(record, scalar_path, spec=NULLABLE, replace=True)
    assert read_doc(scalar_path, "run_values") == {
        "version": 1,
        "values": {"start": {"unit": "s", "value": None}},
    }


def test_cfg_incompatibility_does_not_hide_fixed_values_but_blocks_typed_save(
    scalar_path: Path, tmp_path: Path
) -> None:
    incompatible = replace(SCALAR, cfg_schema_version="2.0")
    with pytest.warns(CfgUnavailableWarning, match="schema major"):
        record = load_run_with_metadata(scalar_path, spec=incompatible)
    assert record.cfg is None
    assert record.result.fixed == 0.25

    destination = tmp_path / "cannot-resave.h5"
    with pytest.raises(ValueError, match="cannot save without configuration"):
        save_run(record, destination, spec=incompatible)
    assert not destination.exists()


def test_bad_fixed_value_blocks_native_but_not_labber(
    scalar_path: Path, tmp_path: Path
) -> None:
    record = load_run_with_metadata(scalar_path, spec=SCALAR)
    bad = replace(record, result=replace(record.result, fixed=float("inf")))
    native = tmp_path / "invalid.h5"
    with pytest.raises(ValueError, match="/docs/run_values/values/start/value"):
        save_run(bad, native, spec=SCALAR)
    assert not native.exists()

    labber = tmp_path / "allowed.hdf5"
    save_labber(bad, labber, spec=SCALAR)
    assert labber.is_file()
