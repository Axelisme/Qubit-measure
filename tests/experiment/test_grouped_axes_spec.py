from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from zcu_tools.cfg_model import ConfigBase
from zcu_tools.datafile import (
    DataVariable,
    load_grouped_labber_data,
)
from zcu_tools.experiment import (
    MHZ_TO_HZ,
    GroupedAxesSpec,
    GroupedLoadData,
    VariableAxisSpec,
    VariableSpec,
    VariableZSpec,
    save_labber,
)
from zcu_tools.experiment.records import RunRecord

from tests.experiment._grouped_layout_support import (
    assert_mixed_disk_values,
    mixed_record,
    mixed_spec,
)


class _GroupedCfg(ConfigBase):
    name: str = "grouped"


@dataclass(frozen=True)
class _GroupedResult:
    params: np.ndarray
    scores: np.ndarray


def _validate_result(result: _GroupedResult) -> None:
    params = np.asarray(result.params, dtype=np.float64)
    scores = np.asarray(result.scores, dtype=np.float64)
    if params.ndim != 2 or params.shape[1] != 2:
        raise ValueError(f"grouped params must have shape (N, 2), got {params.shape}")
    if scores.shape != (params.shape[0],):
        raise ValueError(
            f"grouped scores shape {scores.shape} != expected {(params.shape[0],)}"
        )


def _build_result(data: GroupedLoadData[_GroupedCfg]) -> _GroupedResult:
    params = np.column_stack(
        [
            data.variable("freq").z,
            data.variable("gain").z,
        ]
    ).astype(np.float64)
    scores = data.variable("score").z.astype(np.float64)
    _validate_result(_GroupedResult(params=params, scores=scores))
    return _GroupedResult(
        params=params,
        scores=scores,
    )


_ITERATION_AXIS = (
    VariableAxisSpec.generated_arange("Iteration", "a.u.", dtype=np.int64),
)
_GROUPED_SPEC = GroupedAxesSpec(
    variables=(
        VariableSpec(
            variable="freq",
            axes=_ITERATION_AXIS,
            z=VariableZSpec(
                field_name="params",
                label="Frequency",
                unit="Hz",
                scale=MHZ_TO_HZ,
                dtype=np.float64,
                index=0,
                index_axis=1,
            ),
        ),
        VariableSpec(
            variable="gain",
            axes=_ITERATION_AXIS,
            z=VariableZSpec(
                field_name="params",
                label="Gain",
                unit="a.u.",
                dtype=np.float64,
                index=1,
                index_axis=1,
            ),
        ),
        VariableSpec(
            variable="score",
            axes=_ITERATION_AXIS,
            z=VariableZSpec(
                field_name="scores",
                label="Score",
                unit="a.u.",
                dtype=np.float64,
            ),
        ),
    ),
    result_type=_GroupedResult,
    cfg_type=_GroupedCfg,
    tag="test/grouped_axes_spec",
    result_builder=_build_result,
    result_validator=_validate_result,
)


def test_grouped_axes_spec_exports_typed_result(tmp_path: Path) -> None:
    params = np.array(
        [
            [6123.0, 0.1],
            [6124.0, 0.2],
            [6125.0, 0.3],
        ],
        dtype=np.float64,
    )
    scores = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    source = RunRecord(
        cfg=_GroupedCfg(name="roundtrip"),
        result=_GroupedResult(params=params, scores=scores),
    )
    written = tmp_path / "grouped.hdf5"
    save_labber(source, written, comment="note", spec=_GROUPED_SPEC)

    grouped = load_grouped_labber_data(
        str(written), required_variables=_GROUPED_SPEC.required_variables
    )
    assert list(grouped.variables) == [
        DataVariable("freq"),
        DataVariable("gain"),
        DataVariable("score"),
    ]
    np.testing.assert_allclose(
        grouped.variables[DataVariable("freq")].z, params[:, 0] * MHZ_TO_HZ
    )
    np.testing.assert_allclose(grouped.variables[DataVariable("gain")].z, params[:, 1])
    np.testing.assert_allclose(grouped.variables[DataVariable("score")].z, scores)


def test_grouped_save_keeps_explicit_source_and_refuses_overwrite(
    tmp_path: Path,
) -> None:
    first = RunRecord(
        cfg=_GroupedCfg(name="first"),
        result=_GroupedResult(
            params=np.array([[6100.0, 0.2]]),
            scores=np.array([1.0]),
        ),
    )
    second = RunRecord(
        cfg=_GroupedCfg(name="second"),
        result=_GroupedResult(
            params=np.array([[6200.0, 0.3]]),
            scores=np.array([2.0]),
        ),
    )
    second_path = tmp_path / "second.hdf5"
    first_path = tmp_path / "first.hdf5"
    save_labber(second, second_path, spec=_GROUPED_SPEC)
    save_labber(first, first_path, spec=_GROUPED_SPEC)

    with pytest.raises(FileExistsError):
        save_labber(second, first_path, spec=_GROUPED_SPEC)

    grouped = load_grouped_labber_data(str(first_path))
    np.testing.assert_array_equal(
        grouped.variables[DataVariable("freq")].z, first.result.params[:, 0] * MHZ_TO_HZ
    )
    np.testing.assert_array_equal(
        grouped.variables[DataVariable("gain")].z, first.result.params[:, 1]
    )
    np.testing.assert_array_equal(
        grouped.variables[DataVariable("score")].z, first.result.scores
    )


@pytest.mark.parametrize(
    "independent", [True, False], ids=["independent", "default-common"]
)
def test_grouped_save_obeys_declared_layout_for_mixed_rank_variables(
    tmp_path: Path, independent: bool
) -> None:
    spec = mixed_spec(independent=independent)
    destination = tmp_path / "mixed.hdf5"
    source = mixed_record()
    if not independent:
        with pytest.raises(ValueError, match="common grid"):
            save_labber(source, destination, spec=spec)
        assert not destination.exists()
        return

    save_labber(source, destination, comment="mixed rank export", spec=spec)
    loaded = load_grouped_labber_data(
        str(destination), required_variables=spec.required_variables
    )
    assert_mixed_disk_values(loaded.variables)
    with pytest.raises(FileExistsError):
        save_labber(source, destination, spec=spec)


def test_grouped_save_requires_cfg(tmp_path: Path) -> None:
    source: RunRecord[_GroupedCfg, _GroupedResult] = RunRecord(
        cfg=None,
        result=_GroupedResult(
            params=np.array([[6100.0, 0.2]]),
            scores=np.array([1.0]),
        ),
    )
    destination = tmp_path / "missing-cfg.hdf5"
    with pytest.raises(ValueError, match="without configuration"):
        save_labber(source, destination, spec=_GROUPED_SPEC)
    assert not destination.exists()


def test_grouped_axes_spec_declaration_rejects_missing_result_field() -> None:
    with pytest.raises(ValueError, match="field_name"):
        GroupedAxesSpec(
            variables=(
                VariableSpec(
                    variable="bad",
                    axes=_ITERATION_AXIS,
                    z=VariableZSpec(
                        field_name="missing",
                        label="Missing",
                        unit="a.u.",
                    ),
                ),
            ),
            result_type=_GroupedResult,
            cfg_type=_GroupedCfg,
            tag="test/bad",
            result_builder=_build_result,
        )
