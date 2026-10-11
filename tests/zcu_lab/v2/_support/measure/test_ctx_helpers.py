"""Public fixed-parameter frequency/flux range policies."""

from __future__ import annotations

from collections.abc import Callable, Generator
from pathlib import Path

import pytest
from zcu_tools.format_version import YamlMap, YamlValue
from zcu_tools.gui.cfg import EvalValue, SweepValue
from zcu_tools.gui.session.context_types import (
    ContextSourceFault,
    ContextSourceUnavailableError,
    SavedPoint,
)
from zcu_tools.gui.session.expression import ExpressionError, evaluate_scalar_expr
from zcu_tools.gui.session.parameters import ParameterContext, RoleBinding
from zcu_tools.gui.session.types import SessionEnv
from zcu_tools.resources.context.library import LibrarySnapshot, ModuleLibraryDocument
from zcu_tools.resources.entry import (
    ComponentSchema,
    RoleResolutionError,
    RoleSpec,
    component_registry,
)
from zcu_tools.resources.entry.schema import PointDocument

from zcu_lab.v2._support.measure.ctx_helpers import (
    optional_parameter_float,
    proper_flux_range,
    proper_qub_freq_range,
    proper_res_freq_range,
)
from zcu_lab.v2._support.measure.seeds import res_freq_range


class RangeSensor(ComponentSchema):
    freq: YamlValue = None
    kappa: YamlValue = None


class RangeDriver(RangeSensor):
    flux_half: YamlValue = None
    flux_int: YamlValue = None


RangeFactory = Callable[[dict[str, YamlMap]], ParameterContext]


@pytest.fixture
def range_factory() -> Generator[RangeFactory]:
    before = component_registry.kinds()
    component_registry.register("range/sensor", RangeSensor)
    component_registry.register("range/drive", RangeDriver)

    def make(overrides: dict[str, YamlMap]) -> ParameterContext:
        components: YamlMap = {
            "R1": {"kind": "range/sensor", **overrides.get("R1", {})},
            "Q1": {"kind": "range/drive", **overrides.get("Q1", {})},
        }
        point = PointDocument.model_validate(
            {
                "format": "zcu.parameter-container",
                "format_version": "1.0",
                "general": {"created_at": "2026-10-04T00:00:00Z"},
                "components": components,
            }
        )
        return ParameterContext(
            point=point,
            library=LibrarySnapshot(
                ModuleLibraryDocument(format="zcu.module-library", format_version="1.0")
            ),
            target=SavedPoint(
                entry_id="00000000-0000-0000-0000-000000000001", point="working"
            ),
            roles=RoleBinding(
                "Q1",
                {
                    "qubit": RoleSpec("range/drive"),
                    "resonator": RoleSpec("range/sensor"),
                },
                {"qubit": "Q1", "resonator": "R1"},
            ),
        )

    try:
        yield make
    finally:
        component_registry.unregister("range/drive")
        component_registry.unregister("range/sensor")
        assert component_registry.kinds() == before


@pytest.mark.parametrize("value", [None, 2, 2.5])
def test_optional_parameter_float_reads_null_or_finite_working_value(
    range_factory: RangeFactory, value: YamlValue
) -> None:
    capture = range_factory({"R1": {"freq": value}})
    actual = optional_parameter_float(capture, "resonator.freq")
    assert actual == value
    if value is not None:
        assert isinstance(actual, float)
    assert optional_parameter_float(capture, "resonator.unset") is None


@pytest.mark.parametrize("value", [True, "2.5", [2.5], {"value": 2.5}])
def test_optional_parameter_float_rejects_present_non_real_values(
    range_factory: RangeFactory, value: YamlValue
) -> None:
    capture = range_factory({"R1": {"freq": value}})
    with pytest.raises(TypeError, match="real scalar"):
        optional_parameter_float(capture, "resonator.freq")


@pytest.mark.parametrize("value", [float("inf"), float("-inf"), float("nan")])
def test_optional_parameter_float_rejects_present_nonfinite_values(
    range_factory: RangeFactory, value: float
) -> None:
    capture = range_factory({"R1": {"freq": value}})
    with pytest.raises(ValueError, match="finite"):
        optional_parameter_float(capture, "resonator.freq")


def test_optional_parameter_float_keeps_capture_and_propagates_source_fault(
    range_factory: RangeFactory,
) -> None:
    capture = range_factory({"R1": {"freq": 2.5}})
    other = range_factory({"R1": {"freq": 7.5}})
    assert optional_parameter_float(other, "resonator.freq") == 7.5
    assert optional_parameter_float(capture, "resonator.freq") == 2.5
    unavailable = ParameterContext(
        point=capture.document(),
        library=capture.library,
        target=capture.target,
        roles=capture.roles,
        source_faults=(
            ContextSourceFault(
                document="point",
                path=Path("/contract/point.yaml"),
                error_type="ValueError",
                message="broken source",
            ),
        ),
    )
    with pytest.raises(ContextSourceUnavailableError):
        optional_parameter_float(unavailable, "resonator.freq")


def test_res_freq_range_uses_eval_value_when_parameters_present(
    range_factory: RangeFactory,
) -> None:
    capture = range_factory({"R1": {"freq": 5500.0, "kappa": 10.0}})
    sweep = proper_res_freq_range(capture, 301)
    assert isinstance(sweep, SweepValue)
    assert isinstance(sweep.start, EvalValue)
    assert isinstance(sweep.stop, EvalValue)
    assert sweep.start.expr == "$resonator.freq - 1.5 * $resonator.kappa"
    assert sweep.start.resolved is None
    assert sweep.stop.expr == "$resonator.freq + 1.5 * $resonator.kappa"
    assert sweep.stop.resolved is None
    assert sweep.expts == 301
    assert evaluate_scalar_expr(sweep.start.expr, capture) == 5485.0
    assert evaluate_scalar_expr(sweep.stop.expr, capture) == 5515.0


def test_res_freq_range_falls_back_to_scalar_without_calibration(
    range_factory: RangeFactory,
) -> None:
    capture = range_factory({})
    sweep = proper_res_freq_range(capture, 101)
    assert sweep.expts == 101
    for edge in (sweep.start, sweep.stop):
        assert isinstance(edge, EvalValue)
        assert edge.resolved is None
        with pytest.raises(ExpressionError) as raised:
            evaluate_scalar_expr(edge.expr, capture)
        assert str(raised.value) == (
            "Invalid parameter path '$resonator.freq': missing parameter"
        )


def test_freq_range_span_factor_one_keeps_unit_span(
    range_factory: RangeFactory,
) -> None:
    capture = range_factory({"R1": {"freq": 6000.0, "kappa": 20.0}})
    sweep = proper_res_freq_range(capture, 101, span_factor=1.0)
    assert isinstance(sweep.start, EvalValue)
    assert sweep.start.expr == "$resonator.freq - 1.0 * $resonator.kappa"
    assert sweep.start.resolved is None
    assert isinstance(sweep.stop, EvalValue)
    assert sweep.stop.expr == "$resonator.freq + 1.0 * $resonator.kappa"
    assert sweep.stop.resolved is None
    assert sweep.expts == 101
    assert evaluate_scalar_expr(sweep.start.expr, capture) == 5980.0
    assert evaluate_scalar_expr(sweep.stop.expr, capture) == 6020.0


def test_qub_freq_range_uses_qubit_parameter_paths(range_factory: RangeFactory) -> None:
    capture = range_factory({"Q1": {"freq": 4200.0, "kappa": 5.0}})
    sweep = proper_qub_freq_range(capture, 201, span_factor=2.0)
    assert isinstance(sweep.start, EvalValue)
    assert sweep.start.expr == "$qubit.freq - 2.0 * $qubit.kappa"
    assert sweep.start.resolved is None
    assert sweep.expts == 201
    assert isinstance(sweep.stop, EvalValue)
    assert evaluate_scalar_expr(sweep.start.expr, capture) == 4190.0
    assert evaluate_scalar_expr(sweep.stop.expr, capture) == 4210.0


@pytest.mark.parametrize("calibration", [{"freq": 5500.0}, {"kappa": 10.0}])
def test_resonator_partial_calibration_rejects_missing_companion_at_resolution(
    range_factory: RangeFactory, calibration: YamlMap
) -> None:
    original = range_factory({"R1": calibration})
    doubled = range_factory(
        {
            "R1": {
                key: value * 2
                for key, value in calibration.items()
                if isinstance(value, float)
            }
        }
    )
    sweep = proper_res_freq_range(original, 21)
    assert isinstance(sweep.start, EvalValue)
    assert isinstance(sweep.stop, EvalValue)
    assert sweep.expts == 21
    missing = "kappa" if "freq" in calibration else "freq"
    for capture in (original, doubled):
        for edge in (sweep.start, sweep.stop):
            assert edge.resolved is None
            with pytest.raises(ExpressionError) as raised:
                evaluate_scalar_expr(edge.expr, capture)
            assert str(raised.value) == (
                f"Invalid parameter path '$resonator.{missing}': missing parameter"
            )
    if "freq" in calibration:
        assert evaluate_scalar_expr("$resonator.freq", original) == 5500.0
        assert evaluate_scalar_expr("$resonator.freq", doubled) == 11000.0
    else:
        assert evaluate_scalar_expr("$resonator.kappa", original) == 10.0
        assert evaluate_scalar_expr("$resonator.kappa", doubled) == 20.0


def test_flux_range_extrapolates_past_calibrated_points(
    range_factory: RangeFactory,
) -> None:
    capture = range_factory({"Q1": {"flux_half": 1e-3, "flux_int": 3e-3}})
    sweep = proper_flux_range(capture, 101)
    assert isinstance(sweep.start, EvalValue)
    assert isinstance(sweep.stop, EvalValue)
    assert sweep.start.expr == "1.1 * $qubit.flux_int - 0.1 * $qubit.flux_half"
    assert sweep.start.resolved is None
    assert sweep.stop.expr == "1.1 * $qubit.flux_half - 0.1 * $qubit.flux_int"
    assert sweep.expts == 101
    assert evaluate_scalar_expr(sweep.start.expr, capture) == pytest.approx(3.2e-3)
    assert evaluate_scalar_expr(sweep.stop.expr, capture) == pytest.approx(0.8e-3)


def test_flux_range_falls_back_when_calibration_absent(
    range_factory: RangeFactory,
) -> None:
    capture = range_factory({})
    sweep = proper_flux_range(capture, 101)
    assert sweep.expts == 101
    for edge, field in ((sweep.start, "flux_int"), (sweep.stop, "flux_half")):
        assert isinstance(edge, EvalValue)
        assert edge.resolved is None
        with pytest.raises(ExpressionError) as raised:
            evaluate_scalar_expr(edge.expr, capture)
        assert str(raised.value) == (
            f"Invalid parameter path '$qubit.{field}': missing parameter"
        )


@pytest.mark.parametrize(
    "calibration", [{"flux_half": 1e-3}, {"flux_int": None, "flux_half": 1e-3}]
)
def test_partial_or_null_flux_calibration_uses_original_fallback(
    range_factory: RangeFactory, calibration: YamlMap
) -> None:
    capture = range_factory({"Q1": calibration})
    sweep = proper_flux_range(capture, 11)
    assert sweep.expts == 11
    for edge in (sweep.start, sweep.stop):
        assert isinstance(edge, EvalValue)
        assert edge.resolved is None
        with pytest.raises(ExpressionError) as raised:
            evaluate_scalar_expr(edge.expr, capture)
        expected = (
            "Parameter path '$qubit.flux_int' is not numeric"
            if "flux_int" in calibration
            else "Invalid parameter path '$qubit.flux_int': missing parameter"
        )
        assert str(raised.value) == expected


@pytest.mark.parametrize("value", [True, "5500", [], {}, float("inf"), float("nan")])
@pytest.mark.parametrize("field", ["freq", "kappa"])
def test_present_invalid_resonator_scalar_does_not_use_fallback(
    range_factory: RangeFactory, field: str, value: YamlValue
) -> None:
    capture = range_factory({"R1": {field: value}})
    sweep = proper_res_freq_range(capture, 11)
    assert sweep.expts == 11
    reason = "must be finite" if isinstance(value, float) else "is not numeric"
    for edge in (sweep.start, sweep.stop):
        assert isinstance(edge, EvalValue)
        assert edge.resolved is None
        with pytest.raises(ExpressionError) as raised:
            evaluate_scalar_expr(edge.expr, capture)
        expected = (
            "Invalid parameter path '$resonator.freq': missing parameter"
            if field == "kappa"
            else f"Parameter path '$resonator.freq' {reason}"
        )
        assert str(raised.value) == expected
    if field == "kappa":
        # The missing companion must not masquerade as rejection of bad kappa.
        with pytest.raises(ExpressionError) as raised:
            evaluate_scalar_expr("$resonator.kappa", capture)
        assert str(raised.value) == f"Parameter path '$resonator.kappa' {reason}"


@pytest.mark.parametrize("field", ["flux_half", "flux_int"])
def test_present_invalid_flux_scalar_does_not_use_fallback(
    range_factory: RangeFactory, field: str
) -> None:
    capture = range_factory({"Q1": {field: "invalid"}})
    sweep = proper_flux_range(capture, 11)
    assert sweep.expts == 11
    for edge, first_field in ((sweep.start, "flux_int"), (sweep.stop, "flux_half")):
        assert isinstance(edge, EvalValue)
        assert edge.resolved is None
        with pytest.raises(ExpressionError) as raised:
            evaluate_scalar_expr(edge.expr, capture)
        expected = (
            f"Parameter path '$qubit.{first_field}' is not numeric"
            if field == first_field
            else f"Invalid parameter path '$qubit.{first_field}': missing parameter"
        )
        assert str(raised.value) == expected


def test_range_seed_uses_the_fixed_parameter_capture(
    range_factory: RangeFactory,
) -> None:
    context = range_factory({"R1": {"freq": 5500.0, "kappa": 10.0}})
    environment = SessionEnv(soc=None, soccfg=None, parameters=context)
    sweep = res_freq_range(expts=21).resolve(environment)
    assert isinstance(sweep.start, EvalValue)
    assert sweep.start.expr == "$resonator.freq - 1.5 * $resonator.kappa"
    assert sweep.start.resolved is None
    assert sweep.expts == 21
    assert evaluate_scalar_expr(sweep.start.expr, environment.parameters) == 5485.0


def test_unknown_role_does_not_become_uncalibrated_fallback(
    range_factory: RangeFactory,
) -> None:
    context = range_factory({})
    direct = ParameterContext(
        point=context.document(),
        library=context.library,
        target=context.target,
        roles=None,
    )
    sweep = proper_res_freq_range(direct, 11)
    assert sweep.expts == 11
    for edge in (sweep.start, sweep.stop):
        assert isinstance(edge, EvalValue)
        assert edge.resolved is None
        with pytest.raises(ExpressionError) as raised:
            evaluate_scalar_expr(edge.expr, direct)
        assert str(raised.value) == (
            "Invalid parameter path '$resonator.freq': Role 'resonator' requires "
            "kind None; focus=None: Alias is not bound"
        )
        assert isinstance(raised.value.__cause__, RoleResolutionError)
