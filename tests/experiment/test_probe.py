"""Offline probe contracts through public orchestration and in-process entry."""

import json
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import pytest
from matplotlib import pyplot as plt
from numpy.typing import NDArray
from zcu_tools.experiment import (
    AxesSpec,
    Axis,
    Experiment,
    GroupedAxesSpec,
    VariableAxisSpec,
    VariableSpec,
    VariableZSpec,
    ZSpec,
    native_persistence,
)
from zcu_tools.experiment.context import RunContext
from zcu_tools.experiment.probe import compare_roundtrip, main, probe
from zcu_tools.experiment.records import RunRecord
from zcu_tools.experiment.simulation import white_noise
from zcu_tools.plotting.plots import Plots

from tests.experiment._probe_support import ProbeExperiment
from tests.experiment._saving_support import SavingCfg, SavingCore, SavingResult


@pytest.fixture(autouse=True)
def restore_rc_params():
    with matplotlib.rc_context():
        yield


def test_probe_applies_style_before_run_and_preserves_native_overrides(tmp_path: Path):
    observed: list[float | str] = []

    class StyledExperiment(ProbeExperiment):
        def run(self, config: SavingCfg, *, context: RunContext) -> SavingResult:
            _, axes = context.plots.subplots("style-default")
            observed.append(axes.xaxis.label.get_fontsize())
            axes.set_xlabel("Frequency", fontsize=23)
            assert axes.xaxis.label.get_fontsize() == 23
            return super().run(config, context=context)

        def analyze(
            self,
            source: RunRecord[SavingCfg, SavingResult],
            options: Any,
            *,
            plots: Plots,
        ) -> dict[str, object]:
            _, axes = plots.subplots("style-default")
            observed.append(axes.xaxis.label.get_fontsize())
            with plt.style.context({"axes.labelsize": 31}):
                _, custom = plots.subplots("style-override")
                assert custom.xaxis.label.get_fontsize() == 31
            return super().analyze(source, options, plots=plots)

    environment = white_noise(seed=17)
    try:
        with matplotlib.rc_context({"axes.labelsize": 37}):
            report = probe(StyledExperiment(), SavingCfg(), environment, tmp_path)
            assert report.failed_stage is None
            assert report.exception is None
            assert len(observed) == 2
            assert all(size != 37 for size in observed)
    finally:
        environment.devices.close_all_devices()


def test_probe_roundtrips_result_and_exports_live_and_analysis_pngs(tmp_path: Path):
    environment = white_noise(seed=17)
    try:
        report = probe(ProbeExperiment(), SavingCfg(gain=2), environment, tmp_path)
        assert report.failed_stage is None
        assert report.exception is None
        assert report.environment == environment.description
        assert report.fields["frequency"].shape == (3,)
        assert report.fields["frequency"].dtype == "float64"
        assert report.fields["signal"].shape == (3,)
        assert report.fields["signal"].dtype == "complex128"
        assert report.restored_fields == report.fields
        assert report.roundtrip_equal is True
        assert report.analysis_summary is not None
        assert "'gain': 2.0" in report.analysis_summary
        assert report.native_path.is_file()
        assert len(report.png_paths) == 2
        assert {path.name for path in report.png_paths} == {
            "live-signal%2Ftrace.png",
            "analysis-fit.png",
        }
        assert all(
            path.read_bytes().startswith(b"\x89PNG\r\n\x1a\n")
            for path in report.png_paths
        )
        assert all(
            report.timings[stage] >= 0
            for stage in ("run", "save", "restore", "analyze")
        )
    finally:
        environment.devices.close_all_devices()


def test_probe_accepts_single_scale_native_roundoff(tmp_path: Path):
    class ScaledExperiment(ProbeExperiment):
        AXES_SPEC = replace(
            SavingCore.AXES_SPEC,
            axes=(Axis("frequency", "coordinate", "s", scale=1e-6),),
        )

        def run(self, config: SavingCfg, *, context: RunContext) -> SavingResult:
            return SavingResult(
                frequency=np.array([0.122, 0.244, 0.366]),
                signal=np.array([1 + 2j, 3 - 4j, 5 + 6j]),
            )

    environment = white_noise(seed=17)
    try:
        report = probe(ScaledExperiment(), SavingCfg(), environment, tmp_path)
        assert report.failed_stage is None
        assert report.exception is None
        assert report.restored_fields == report.fields
        assert report.roundtrip_equal is True
    finally:
        environment.devices.close_all_devices()


def test_probe_reports_grouped_scale_native_roundoff_as_equal(tmp_path: Path):
    class GroupedScaledExperiment(Experiment[SavingResult, SavingCfg]):
        AXES_SPEC = GroupedAxesSpec(
            variables=(
                VariableSpec(
                    "response",
                    axes=(
                        VariableAxisSpec("coordinate", "s", "frequency", scale=1e-6),
                    ),
                    z=VariableZSpec("signal", "signal", "V", dtype=np.complex128),
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

        def run(self, config: SavingCfg, *, context: RunContext) -> SavingResult:
            return SavingResult(
                frequency=np.array([0.122, 0.244, 0.366]),
                signal=np.array([1 + 2j, 3 - 4j, 5 + 6j]),
            )

        def analyze(
            self,
            source: RunRecord[SavingCfg, SavingResult],
            options: Any,
            *,
            plots: Plots,
        ) -> float:
            return float(source.result.frequency[-1])

    environment = white_noise(seed=17)
    try:
        report = probe(GroupedScaledExperiment(), SavingCfg(), environment, tmp_path)
        assert report.failed_stage is None
        assert report.exception is None
        assert report.restored_fields == report.fields
        assert report.roundtrip_equal is True
        assert report.analysis_summary is not None
    finally:
        environment.devices.close_all_devices()


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
                axes=(VariableAxisSpec("coordinate", "s", "coordinate", scale=1e-6),),
                z=VariableZSpec("signal", "signal", "V", scale=1e-3),
            ),
            VariableSpec(
                "coordinate_copy",
                axes=(VariableAxisSpec.generated_arange("sample", ""),),
                z=VariableZSpec("coordinate", "coordinate", "s", scale=1e-6),
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


@pytest.mark.parametrize("conflicting_scale", [1.0, 1e3, -1e-6])
def test_grouped_comparison_rejects_conflicting_scales_for_one_field(
    grouped_comparison_spec: GroupedAxesSpec[_ComparisonResult, SavingCfg],
    comparison_result: _ComparisonResult,
    conflicting_scale: float,
):
    response, coordinate_copy = grouped_comparison_spec.variables
    spec = replace(
        grouped_comparison_spec,
        variables=(
            response,
            replace(
                coordinate_copy, z=replace(coordinate_copy.z, scale=conflicting_scale)
            ),
        ),
    )
    with pytest.raises(ValueError, match="coordinate") as caught:
        compare_roundtrip(comparison_result, comparison_result, spec=spec)
    message = str(caught.value)
    assert str(1e-6) in message
    assert str(conflicting_scale) in message


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
        replace(VariableAxisSpec.generated_arange("sample", ""), scale=1e-6)
        if generated
        else VariableAxisSpec("sample", "", values=(0, 1, 2), scale=1e-6)
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


@pytest.mark.parametrize("fail_analysis", [False, True], ids=["success", "failure"])
def test_probe_isolates_nested_caller_options_even_when_analysis_fails(
    tmp_path: Path, fail_analysis: bool
):
    error = RuntimeError("analysis failed after mutating options")

    class MutatingExperiment(ProbeExperiment):
        def analyze(
            self,
            source: RunRecord[SavingCfg, SavingResult],
            options: Any,
            *,
            plots: Plots,
        ) -> dict[str, object]:
            options["window"]["samples"].append(7)
            if fail_analysis:
                raise error
            return {"sample_count": len(options["window"]["samples"])}

    options = {"window": {"samples": [2, 4]}}
    environment = white_noise(seed=17)
    try:
        report = probe(
            MutatingExperiment(), SavingCfg(), environment, tmp_path, options=options
        )
        if fail_analysis:
            assert report.failed_stage == "analyze"
            assert report.exception is error
            assert report.analysis_summary is None
        else:
            assert report.failed_stage is None
            assert report.exception is None
            assert report.analysis_summary == "{'sample_count': 3}"
        assert options == {"window": {"samples": [2, 4]}}
    finally:
        environment.devices.close_all_devices()


@pytest.mark.parametrize("cfg_format", ["json", "yaml"])
def test_main_validates_cfg_and_produces_probe_report_in_process(
    tmp_path: Path, capsys: pytest.CaptureFixture[str], cfg_format: str
):
    cfg = tmp_path / f"cfg.{cfg_format}"
    cfg.write_text('{"gain": 3}' if cfg_format == "json" else "gain: 3\n")
    options = tmp_path / "options.json"
    options.write_text('{"window": 7}')
    report = main(
        [
            "tests.experiment._probe_support:ProbeExperiment",
            "--cfg",
            str(cfg),
            "--environment",
            "white_noise",
            "--seed",
            "17",
            "--output-dir",
            str(tmp_path / "output"),
            "--options",
            str(options),
        ]
    )
    assert report.failed_stage is None
    assert report.roundtrip_equal is True
    assert report.fields["signal"].shape == (3,)
    assert report.analysis_summary is not None
    assert "'gain': 3.0" in report.analysis_summary
    assert "'window': 7" in report.analysis_summary
    assert len(report.png_paths) == 2
    output = json.loads(capsys.readouterr().out)
    assert output["fields"]["signal"]["shape"] == [3]
    assert output["exception"] is None


@pytest.mark.parametrize("stage", ["run", "save", "restore", "analyze"])
def test_probe_reports_failed_stage_and_retains_original_exception(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, stage: str
):
    error = RuntimeError(f"injected {stage} failure")

    class FailingExperiment(ProbeExperiment):
        def run(self, config: SavingCfg, *, context: RunContext) -> SavingResult:
            if stage == "run":
                raise error
            return super().run(config, context=context)

        def analyze(
            self,
            source: RunRecord[SavingCfg, SavingResult],
            options: Any,
            *,
            plots: Plots,
        ) -> dict[str, object]:
            if stage == "analyze":
                raise error
            return super().analyze(source, options, plots=plots)

    def fail(*_args: object, **_kwargs: object) -> Any:
        raise error

    if stage in ("save", "restore"):
        monkeypatch.setattr(native_persistence, f"{stage}_run", fail)
    environment = white_noise(seed=17)
    try:
        report = probe(FailingExperiment(), SavingCfg(), environment, tmp_path)
        assert report.failed_stage == stage
        assert report.exception is error
        assert report.timings[stage] >= 0
        assert report.analysis_summary is None
        if stage != "run":
            assert any(path.name.startswith("live-") for path in report.png_paths)
    finally:
        environment.devices.close_all_devices()
