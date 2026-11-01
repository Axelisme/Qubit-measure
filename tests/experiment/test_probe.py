"""Native roundtrip and scaled-field comparison contracts."""

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pytest
from numpy.typing import NDArray
from zcu_tools.experiment import (
    AxesSpec,
    Axis,
    GroupedAxesSpec,
    VariableAxisSpec,
    VariableSpec,
    VariableZSpec,
    ZSpec,
)
from zcu_tools.experiment.probe import compare_roundtrip, save_stage
from zcu_tools.experiment.records import RunRecord

from tests._native_support import native_metadata
from tests.experiment._saving_support import SavingCfg, SavingCore, SavingResult


def test_save_accepts_single_scale_native_roundoff(tmp_path: Path):
    spec = replace(
        SavingCore.AXES_SPEC,
        axes=(Axis("frequency", "coordinate", "s", scale=1e-6),),
    )
    record = RunRecord(
        SavingCfg(),
        SavingResult(
            frequency=np.array([0.122, 0.244, 0.366]),
            signal=np.array([1 + 2j, 3 - 4j, 5 + 6j]),
        ),
        native_metadata(spec.tag).model_dump(mode="json"),
    )
    outcome = save_stage(record, spec, tmp_path)
    assert outcome.roundtrip_equal is True
    assert outcome.cfg_issue is None
    assert outcome.record.result.frequency.shape == record.result.frequency.shape
    assert outcome.record.result.frequency.dtype == record.result.frequency.dtype
    assert outcome.record.result.signal.shape == record.result.signal.shape
    assert outcome.record.result.signal.dtype == record.result.signal.dtype


def test_save_reports_grouped_scale_native_roundoff_as_equal(tmp_path: Path):
    spec = GroupedAxesSpec(
        variables=(
            VariableSpec(
                "response",
                axes=(
                    VariableAxisSpec(
                        label="coordinate", unit="s", field_name="frequency", scale=1e-6
                    ),
                ),
                z=VariableZSpec(
                    "signal", label="signal", unit="V", dtype=np.complex128
                ),
            ),
        ),
        result_type=SavingResult,
        cfg_type=SavingCfg,
        tag="test/grouped_probe",
        result_builder=lambda data: SavingResult(
            frequency=data.variable("response").axes[0],
            signal=data.variable("response").z,
        ),
    )
    record = RunRecord(
        SavingCfg(),
        SavingResult(
            frequency=np.array([0.122, 0.244, 0.366]),
            signal=np.array([1 + 2j, 3 - 4j, 5 + 6j]),
        ),
        native_metadata(spec.tag).model_dump(mode="json"),
    )
    outcome = save_stage(record, spec, tmp_path)
    assert outcome.roundtrip_equal is True
    assert outcome.cfg_issue is None
    assert outcome.record.result.frequency.shape == record.result.frequency.shape
    assert outcome.record.result.frequency.dtype == record.result.frequency.dtype
    assert outcome.record.result.signal.shape == record.result.signal.shape
    assert outcome.record.result.signal.dtype == record.result.signal.dtype


@dataclass
class _ComparisonResult:
    coordinate: NDArray[np.generic]
    signal: NDArray[np.generic]
    index: NDArray[np.int64]


@pytest.fixture
def single_comparison_spec() -> AxesSpec[_ComparisonResult, SavingCfg]:
    return AxesSpec(
        axes=(Axis("coordinate", "coordinate", "s", scale=1e-6),),
        z=ZSpec("signal", "signal", "V", scale=1e-3),
        result_type=_ComparisonResult,
        cfg_type=SavingCfg,
        tag="test/comparison",
    )


_ComparisonSpec = (
    AxesSpec[_ComparisonResult, SavingCfg]
    | GroupedAxesSpec[_ComparisonResult, SavingCfg]
)


@pytest.fixture(params=["single", "grouped"])
def comparison_spec(
    request: pytest.FixtureRequest,
    single_comparison_spec: AxesSpec[_ComparisonResult, SavingCfg],
    grouped_comparison_spec: GroupedAxesSpec[_ComparisonResult, SavingCfg],
) -> _ComparisonSpec:
    return (
        single_comparison_spec if request.param == "single" else grouped_comparison_spec
    )


@pytest.fixture
def grouped_comparison_spec() -> GroupedAxesSpec[_ComparisonResult, SavingCfg]:
    return GroupedAxesSpec(
        variables=(
            VariableSpec(
                "response",
                axes=(
                    VariableAxisSpec(
                        label="coordinate",
                        unit="s",
                        field_name="coordinate",
                        scale=1e-6,
                    ),
                ),
                z=VariableZSpec("signal", label="signal", unit="V", scale=1e-3),
            ),
            VariableSpec(
                "coordinate_copy",
                axes=(VariableAxisSpec.generated_arange(label="sample", unit=""),),
                z=VariableZSpec("coordinate", label="coordinate", unit="s", scale=1e-6),
            ),
        ),
        result_type=_ComparisonResult,
        cfg_type=SavingCfg,
        tag="test/grouped_comparison",
        result_builder=lambda data: _ComparisonResult(
            coordinate=data.variable("response").axes[0],
            signal=data.variable("response").z,
            index=np.array([0, 1], dtype=np.int64),
        ),
    )


def test_grouped_scaled_fields_accept_bounded_roundoff(
    grouped_comparison_spec: GroupedAxesSpec[_ComparisonResult, SavingCfg],
    comparison_result: _ComparisonResult,
):
    restored = replace(
        comparison_result,
        coordinate=np.array(
            [1 + 2 * np.finfo(np.float64).eps, np.nan, np.inf, -np.inf, 0]
        ),
        signal=np.array(
            [
                1 + 2j + 2 * np.finfo(np.float64).eps,
                complex(np.nan, 3),
                complex(4, np.nan),
            ]
        ),
    )
    assert compare_roundtrip(comparison_result, restored, spec=grouped_comparison_spec)


@dataclass
class _ColumnResult:
    params: NDArray[np.generic]


@pytest.fixture
def mixed_scale_spec() -> GroupedAxesSpec[_ColumnResult, SavingCfg]:
    return GroupedAxesSpec(
        variables=tuple(
            VariableSpec(
                f"column_{column}",
                axes=(VariableAxisSpec(label="sample", unit="", values=(0, 1)),),
                z=VariableZSpec(
                    "params",
                    label=f"parameter_{column}",
                    unit="",
                    scale=scale,
                    index=column,
                ),
            )
            for column, scale in enumerate((1e-6, 1.0, 1e6))
        ),
        result_type=_ColumnResult,
        cfg_type=SavingCfg,
        tag="test/mixed-scale",
        result_builder=lambda data: _ColumnResult(
            np.column_stack(
                [data.variable(f"column_{column}").z for column in range(3)]
            )
        ),
    )


@pytest.mark.parametrize("bad_column", [None, 0, 1, 2])
def test_mixed_scale_columns_compare_and_warn_only_for_corruption(
    mixed_scale_spec: GroupedAxesSpec[_ColumnResult, SavingCfg],
    caplog: pytest.LogCaptureFixture,
    bad_column: int | None,
):
    original = _ColumnResult(np.ones((2, 3)))
    restored = _ColumnResult(
        np.array([[1 + 2 * np.finfo(float).eps, 1, 1 - np.finfo(float).eps]] * 2)
    )
    if bad_column is not None:
        restored.params[0, bad_column] = 2
    assert compare_roundtrip(original, restored, spec=mixed_scale_spec) is (
        bad_column is None
    )
    warnings = [
        record
        for record in caplog.records
        if record.name == "zcu_tools.experiment.probe" and record.levelname == "WARNING"
    ]
    assert len(warnings) == (0 if bad_column is None else 1)
    if bad_column is not None:
        assert "params" in warnings[0].message
        assert "index_axis=-1" in warnings[0].message
        assert f"index={bad_column}" in warnings[0].message
        assert "value" in warnings[0].message.lower()


def test_save_roundtrips_mixed_scale_grouped_columns(
    mixed_scale_spec: GroupedAxesSpec[_ColumnResult, SavingCfg],
    tmp_path: Path,
):
    record = RunRecord(
        SavingCfg(),
        _ColumnResult(np.full((2, 3), 0.122)),
        native_metadata(mixed_scale_spec.tag).model_dump(mode="json"),
    )
    outcome = save_stage(record, mixed_scale_spec, tmp_path)
    assert outcome.roundtrip_equal is True
    assert outcome.cfg_issue is None
    assert outcome.record.result.params.shape == record.result.params.shape
    assert outcome.record.result.params.dtype == record.result.params.dtype


@pytest.mark.parametrize("index_axis", [-1, 0])
def test_mixed_scale_identity_column_remains_exact_and_reports_all_bad_columns(
    mixed_scale_spec: GroupedAxesSpec[_ColumnResult, SavingCfg],
    caplog: pytest.LogCaptureFixture,
    index_axis: int,
):
    spec = replace(
        mixed_scale_spec,
        variables=tuple(
            replace(variable, z=replace(variable.z, index_axis=index_axis))
            for variable in mixed_scale_spec.variables
        ),
    )
    original = _ColumnResult(np.ones((3, 3)))
    restored = _ColumnResult(np.full((3, 3), 1 + 3 * np.finfo(float).eps))
    assert compare_roundtrip(original, restored, spec=spec) is False
    warnings = [
        record.message
        for record in caplog.records
        if record.name == "zcu_tools.experiment.probe"
    ]
    assert len(warnings) == 3
    for column, message in enumerate(warnings):
        assert f"index_axis={index_axis}" in message
        assert f"index={column}" in message
    caplog.clear()
    restored = _ColumnResult(np.full((3, 3), 1 + np.finfo(float).eps))
    assert compare_roundtrip(original, restored, spec=spec) is False
    assert len(caplog.records) == 1
    assert "index=1" in caplog.records[0].message
    caplog.clear()
    if index_axis == 0:
        restored.params[1, :] = 1
    else:
        restored.params[:, 1] = 1
    assert compare_roundtrip(original, restored, spec=spec) is True
    assert not caplog.records


@pytest.mark.parametrize("reverse", [False, True])
def test_mixed_scale_overlapping_declarations_must_all_match(
    mixed_scale_spec: GroupedAxesSpec[_ColumnResult, SavingCfg],
    reverse: bool,
):
    scaled = mixed_scale_spec.variables[0]
    exact = replace(scaled, variable="exact", z=replace(scaled.z, index=None, scale=1))
    variables = (scaled, exact) if not reverse else (exact, scaled)
    spec = replace(mixed_scale_spec, variables=variables)
    original = _ColumnResult(np.ones((2, 3)))
    restored = _ColumnResult(original.params.copy())
    restored.params[0, 0] = np.nextafter(1.0, 2.0)
    assert compare_roundtrip(original, original, spec=spec) is True
    assert compare_roundtrip(original, restored, spec=spec) is False


def test_mixed_scale_undeclared_columns_remain_exact(
    mixed_scale_spec: GroupedAxesSpec[_ColumnResult, SavingCfg],
):
    spec = replace(mixed_scale_spec, variables=mixed_scale_spec.variables[:1])
    original = _ColumnResult(np.ones((2, 3)))
    restored = _ColumnResult(original.params.copy())
    restored.params[0, 2] = np.nextafter(1.0, 2.0)
    assert compare_roundtrip(original, restored, spec=spec) is False


@pytest.mark.parametrize("scale", [1.0, np.inf, np.nan])
def test_grouped_identity_and_invalid_scales_remain_exact(
    grouped_comparison_spec: GroupedAxesSpec[_ComparisonResult, SavingCfg],
    comparison_result: _ComparisonResult,
    scale: float,
):
    response = grouped_comparison_spec.variables[0]
    spec = replace(
        grouped_comparison_spec,
        variables=(replace(response, z=replace(response.z, scale=scale)),),
    )
    signal = comparison_result.signal.copy()
    signal[0] += np.finfo(np.float64).eps
    assert not compare_roundtrip(
        comparison_result, replace(comparison_result, signal=signal), spec=spec
    )


@pytest.mark.parametrize("generated", [False, True], ids=["fixed", "generated"])
def test_grouped_axes_without_result_fields_do_not_relax_undeclared_arrays(
    grouped_comparison_spec: GroupedAxesSpec[_ComparisonResult, SavingCfg],
    comparison_result: _ComparisonResult,
    generated: bool,
):
    response = grouped_comparison_spec.variables[0]
    axis = (
        replace(VariableAxisSpec.generated_arange(label="sample", unit=""), scale=1e-6)
        if generated
        else VariableAxisSpec(label="sample", unit="", values=(0, 1, 2), scale=1e-6)
    )
    spec = replace(
        grouped_comparison_spec, variables=(replace(response, axes=(axis,)),)
    )
    coordinate = comparison_result.coordinate.copy()
    coordinate[0] = np.nextafter(1.0, 2.0)
    assert not compare_roundtrip(
        comparison_result, replace(comparison_result, coordinate=coordinate), spec=spec
    )


@pytest.fixture
def comparison_result() -> _ComparisonResult:
    return _ComparisonResult(
        coordinate=np.array([1.0, np.nan, np.inf, -np.inf, 0.0]),
        signal=np.array([1 + 2j, complex(np.nan, 3), complex(4, np.nan)]),
        index=np.array([0, 1], dtype=np.int64),
    )


@pytest.mark.parametrize("dtype", [np.float32, np.float64, np.complex64, np.complex128])
@pytest.mark.parametrize("steps, expected", [(2, True), (3, False)])
def test_scaled_comparison_enforces_relative_bound_at_storage_precision(
    comparison_spec: _ComparisonSpec,
    comparison_result: _ComparisonResult,
    dtype: type[np.floating] | type[np.complexfloating],
    steps: int,
    expected: bool,
):
    original = replace(comparison_result, signal=np.array([1.0], dtype=dtype))
    restored = replace(
        original, signal=np.array([1 + steps * np.finfo(dtype).eps], dtype=dtype)
    )
    assert compare_roundtrip(original, restored, spec=comparison_spec) is expected


@pytest.mark.parametrize(
    "coordinate",
    [
        np.array([1 + 3 * np.finfo(np.float64).eps, np.nan, np.inf, -np.inf, 0]),
        np.array([np.nan, 1.0, np.inf, -np.inf, 0]),
        np.array([1.0, np.nan, -np.inf, -np.inf, 0]),
        np.array([1.0, np.nan, 1e300, -np.inf, 0]),
        np.array([1.0, np.nan, np.inf, -np.inf, np.nextafter(0.0, 1.0)]),
        np.array([[1.0, np.nan, np.inf, -np.inf, 0]]),
        np.array([1.0, np.nan, np.inf, -np.inf, 0], dtype=np.float32),
    ],
    ids=[
        "corruption",
        "nan-position",
        "infinity-sign",
        "finite-infinity",
        "zero",
        "shape",
        "dtype",
    ],
)
def test_scaled_comparison_rejects_axis_corruption_and_layout_changes(
    comparison_spec: _ComparisonSpec,
    comparison_result: _ComparisonResult,
    coordinate: NDArray[np.generic],
):
    restored = replace(comparison_result, coordinate=coordinate)
    assert compare_roundtrip(comparison_result, restored, spec=comparison_spec) is False


@pytest.mark.parametrize(
    "changed",
    [complex(np.nan, 3 + 3 * np.finfo(np.float64).eps * 3), complex(3, np.nan)],
    ids=["finite-component-corruption", "nan-component-position"],
)
def test_scaled_complex_nan_does_not_hide_other_component_changes(
    comparison_spec: _ComparisonSpec,
    comparison_result: _ComparisonResult,
    changed: complex,
):
    signal = comparison_result.signal.copy()
    signal[1] = changed
    restored = replace(comparison_result, signal=signal)
    assert compare_roundtrip(comparison_result, restored, spec=comparison_spec) is False


def test_scaled_comparison_accepts_matching_special_components(
    comparison_spec: _ComparisonSpec,
    comparison_result: _ComparisonResult,
):
    assert compare_roundtrip(comparison_result, comparison_result, spec=comparison_spec)


@pytest.mark.parametrize("field", ["coordinate", "signal", "index"])
def test_unscaled_and_undeclared_fields_remain_exact(
    comparison_spec: _ComparisonSpec,
    comparison_result: _ComparisonResult,
    field: str,
):
    if isinstance(comparison_spec, AxesSpec):
        spec = replace(
            comparison_spec,
            axes=(replace(comparison_spec.axes[0], scale=1),),
            z=replace(comparison_spec.z, scale=1),
        )
    else:
        spec = replace(
            comparison_spec,
            variables=tuple(
                replace(
                    variable,
                    axes=tuple(replace(axis, scale=1) for axis in variable.axes),
                    z=replace(variable.z, scale=1),
                )
                for variable in comparison_spec.variables
            ),
        )
    changed = getattr(comparison_result, field).copy()
    changed[0] += np.finfo(np.float64).eps if field != "index" else 1
    restored = replace(comparison_result, **{field: changed})
    assert compare_roundtrip(comparison_result, restored, spec=spec) is False


def test_mixed_scale_empty_undeclared_field_still_requires_matching_dtype(
    comparison_spec: _ComparisonSpec,
    comparison_result: _ComparisonResult,
):
    original = replace(comparison_result, index=np.array([], dtype=np.int64))
    restored = replace(original, index=np.array([], dtype=np.int32))
    assert compare_roundtrip(original, restored, spec=comparison_spec) is False


def test_scaled_integer_axis_remains_exact(
    comparison_spec: _ComparisonSpec,
    comparison_result: _ComparisonResult,
):
    original = replace(comparison_result, coordinate=np.array([2**60], dtype=np.int64))
    restored = replace(original, coordinate=np.array([2**60 + 1], dtype=np.int64))
    assert compare_roundtrip(original, restored, spec=comparison_spec) is False


@pytest.mark.parametrize("scale", [0.0, np.inf, np.nan])
def test_invalid_axis_scale_does_not_enable_tolerance(
    single_comparison_spec: AxesSpec[_ComparisonResult, SavingCfg],
    comparison_result: _ComparisonResult,
    scale: float,
):
    spec = replace(
        single_comparison_spec,
        axes=(replace(single_comparison_spec.axes[0], scale=scale),),
    )
    coordinate = comparison_result.coordinate.copy()
    coordinate[0] = np.nextafter(1.0, 2.0)
    restored = replace(comparison_result, coordinate=coordinate)
    assert compare_roundtrip(comparison_result, restored, spec=spec) is False


def test_subnormal_difference_gets_no_absolute_allowance(
    comparison_spec: _ComparisonSpec,
    comparison_result: _ComparisonResult,
):
    smallest = np.nextafter(0.0, 1.0)
    original = replace(comparison_result, coordinate=np.array([smallest]))
    restored = replace(original, coordinate=np.array([2 * smallest]))
    assert compare_roundtrip(original, restored, spec=comparison_spec) is False


@pytest.mark.parametrize(
    "signal, expected",
    [
        (complex(np.inf, 1.0), True),
        (complex(-np.inf, 1.0), False),
        (complex(np.inf, np.nextafter(1.0, 2.0)), True),
        (complex(np.inf, 2.0), False),
    ],
)
def test_complex_infinity_does_not_hide_finite_component_changes(
    comparison_spec: _ComparisonSpec,
    comparison_result: _ComparisonResult,
    signal: complex,
    expected: bool,
):
    original = replace(comparison_result, signal=np.array([complex(np.inf, 1)]))
    restored = replace(original, signal=np.array([signal]))
    assert compare_roundtrip(original, restored, spec=comparison_spec) is expected


@dataclass
class _RunOnlyScalarResult:
    """Coordinate and signal arrays plus a scalar estimate only run produces."""

    coordinate: NDArray[np.float64]
    estimate: float | None
    signal: NDArray[np.complex128]


@pytest.fixture
def run_only_scalar_spec() -> AxesSpec[_RunOnlyScalarResult, SavingCfg]:
    return AxesSpec(
        axes=(Axis("coordinate", "coordinate", "s"),),
        z=ZSpec("signal", "signal", "V"),
        result_type=_RunOnlyScalarResult,
        cfg_type=SavingCfg,
        tag="test/run_only_scalar",
    )


def test_scalar_field_restored_as_none_reports_dtype_and_keeps_comparing(
    run_only_scalar_spec: AxesSpec[_RunOnlyScalarResult, SavingCfg],
    caplog: pytest.LogCaptureFixture,
):
    original = _RunOnlyScalarResult(
        coordinate=np.array([1.0, 2.0]),
        estimate=0.5,
        signal=np.array([1 + 2j, 3 - 4j]),
    )
    restored = replace(original, estimate=None, signal=np.array([1 + 2j, 5 - 4j]))
    assert compare_roundtrip(original, restored, spec=run_only_scalar_spec) is False
    warnings = [
        record.message
        for record in caplog.records
        if record.name == "zcu_tools.experiment.probe" and record.levelname == "WARNING"
    ]
    assert len(warnings) == 2
    estimate_warning, signal_warning = warnings
    assert "field=estimate" in estimate_warning
    assert "dtype float64 != object" in estimate_warning
    assert "field=signal" in signal_warning
    assert "value" in signal_warning.lower()


def test_scalar_field_none_on_both_sides_is_equal(
    run_only_scalar_spec: AxesSpec[_RunOnlyScalarResult, SavingCfg],
    caplog: pytest.LogCaptureFixture,
):
    original = _RunOnlyScalarResult(
        coordinate=np.array([1.0, 2.0]),
        estimate=None,
        signal=np.array([1 + 2j, 3 - 4j]),
    )
    restored = replace(original, signal=original.signal.copy())
    assert compare_roundtrip(original, restored, spec=run_only_scalar_spec) is True
    assert not [
        record
        for record in caplog.records
        if record.name == "zcu_tools.experiment.probe"
    ]
