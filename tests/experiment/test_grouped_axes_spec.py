from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest
from zcu_tools.datafile import (
    DatasetRole,
    LabberPayload,
    load_grouped_labber_data,
    save_grouped_labber_data,
)
from zcu_tools.experiment import (
    MHZ_TO_HZ,
    GroupedAxesSpec,
    GroupedLoadData,
    RoleAxisSpec,
    RoleSpec,
    RoleZSpec,
)
from zcu_tools.experiment.cfg_model import ExpCfgModel
from zcu_tools.experiment.records import RunRecord


class _GroupedCfg(ExpCfgModel):
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
            data.role("freq").z,
            data.role("gain").z,
        ]
    ).astype(np.float64)
    scores = data.role("score").z.astype(np.float64)
    _validate_result(_GroupedResult(params=params, scores=scores))
    return _GroupedResult(
        params=params,
        scores=scores,
    )


_ITERATION_AXIS = (RoleAxisSpec.generated_arange("Iteration", "a.u.", dtype=np.int64),)
_GROUPED_SPEC = GroupedAxesSpec(
    roles=(
        RoleSpec(
            role="freq",
            axes=_ITERATION_AXIS,
            z=RoleZSpec(
                field_name="params",
                label="Frequency",
                unit="Hz",
                scale=MHZ_TO_HZ,
                dtype=np.float64,
                index=0,
                index_axis=1,
            ),
        ),
        RoleSpec(
            role="gain",
            axes=_ITERATION_AXIS,
            z=RoleZSpec(
                field_name="params",
                label="Gain",
                unit="a.u.",
                dtype=np.float64,
                index=1,
                index_axis=1,
            ),
        ),
        RoleSpec(
            role="score",
            axes=_ITERATION_AXIS,
            z=RoleZSpec(
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


def test_grouped_axes_spec_saves_and_loads_typed_result(tmp_path: Path) -> None:
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
    _GROUPED_SPEC.save(source, written, comment="note")

    grouped = load_grouped_labber_data(
        str(written), required_roles=_GROUPED_SPEC.required_roles
    )
    assert list(grouped.roles) == [
        DatasetRole("freq"),
        DatasetRole("gain"),
        DatasetRole("score"),
    ]
    np.testing.assert_allclose(
        grouped.roles[DatasetRole("freq")].z, params[:, 0] * MHZ_TO_HZ
    )
    np.testing.assert_allclose(grouped.roles[DatasetRole("gain")].z, params[:, 1])
    np.testing.assert_allclose(grouped.roles[DatasetRole("score")].z, scores)

    loaded = _GROUPED_SPEC.load(written)
    np.testing.assert_allclose(loaded.result.params, params)
    np.testing.assert_allclose(loaded.result.scores, scores)
    assert loaded.cfg is not None
    assert loaded.cfg.name == "roundtrip"


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
    _GROUPED_SPEC.save(second, second_path)
    _GROUPED_SPEC.load(second_path)
    _GROUPED_SPEC.save(first, first_path)

    with pytest.raises(FileExistsError):
        _GROUPED_SPEC.save(second, first_path)

    loaded = _GROUPED_SPEC.load(first_path)
    assert loaded.cfg == first.cfg
    np.testing.assert_array_equal(loaded.result.params, first.result.params)
    np.testing.assert_array_equal(loaded.result.scores, first.result.scores)


def test_grouped_save_requires_cfg(tmp_path: Path) -> None:
    source: RunRecord[_GroupedCfg, _GroupedResult] = RunRecord(
        cfg=None,
        result=_GroupedResult(
            params=np.array([[6100.0, 0.2]]),
            scores=np.array([1.0]),
        ),
    )
    destination = tmp_path / "missing-cfg.hdf5"
    with pytest.raises(ValueError, match="without cfg"):
        _GROUPED_SPEC.save(source, destination)
    assert not destination.exists()


@pytest.mark.parametrize("comment", ["", "not a cfg comment"])
def test_grouped_load_without_cfg_retains_data(tmp_path: Path, comment: str) -> None:
    result = _GroupedResult(
        params=np.array([[6100.0, 0.2]]),
        scores=np.array([1.0]),
    )
    path = _GROUPED_SPEC.save_grouped_result(
        str(tmp_path / "raw.hdf5"),
        result,
        comment=comment,
    )
    loaded = _GROUPED_SPEC.load(Path(path))
    assert loaded.cfg is None
    np.testing.assert_array_equal(loaded.result.params, result.params)
    np.testing.assert_array_equal(loaded.result.scores, result.scores)


def test_grouped_load_invalid_cfg_warns_and_retains_data(tmp_path: Path) -> None:
    result = _GroupedResult(
        params=np.array([[6100.0, 0.2]]),
        scores=np.array([1.0]),
    )
    path = _GROUPED_SPEC.save_grouped_result(
        str(tmp_path / "invalid-cfg.hdf5"),
        result,
        comment='{"cfg": {"name": []}}',
    )
    with pytest.warns(UserWarning):
        loaded = _GROUPED_SPEC.load(Path(path))
    assert loaded.cfg is None
    np.testing.assert_array_equal(loaded.result.params, result.params)
    np.testing.assert_array_equal(loaded.result.scores, result.scores)


def test_grouped_axes_spec_load_rejects_missing_required_role(tmp_path: Path) -> None:
    payload = LabberPayload(
        ("Score", "a.u.", np.ones(2, dtype=np.float64)),
        axes=[("Iteration", "a.u.", np.arange(2, dtype=np.int64))],
    )
    path = save_grouped_labber_data(str(tmp_path / "partial"), {"score": payload})

    with pytest.raises(ValueError, match="missing required dataset role"):
        _GROUPED_SPEC.load(Path(path))


def test_grouped_axes_spec_declaration_rejects_missing_result_field() -> None:
    with pytest.raises(ValueError, match="field_name"):
        GroupedAxesSpec(
            roles=(
                RoleSpec(
                    role="bad",
                    axes=_ITERATION_AXIS,
                    z=RoleZSpec(
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
