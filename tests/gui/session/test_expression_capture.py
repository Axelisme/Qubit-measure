"""Scalar expressions evaluated against fixed parameter captures."""

from collections.abc import Generator

import pytest
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_types import SavedPoint
from zcu_tools.gui.session.expression import (
    ExpressionError,
    evaluate_numeric_expr,
    evaluate_scalar_expr,
)
from zcu_tools.gui.session.types import ParameterContext, RoleBinding
from zcu_tools.resources.context.library import LibrarySnapshot, ModuleLibraryDocument
from zcu_tools.resources.entry import ComponentSchema, RoleSpec, component_registry
from zcu_tools.resources.entry.schema import PointDocument


class Driver(ComponentSchema):
    rate: float
    wiring: dict[str, list[int]]
    flux: dict[str, dict[str, float]]
    flux_dev: str
    index: int


@pytest.fixture
def parameters() -> Generator[ParameterContext]:
    before = component_registry.kinds()
    component_registry.register("lowering/driver", Driver)
    try:
        point = PointDocument.model_validate(
            {
                "format": "zcu.parameter-container",
                "format_version": "2.0",
                "schema_version": 1,
                "general": {
                    "created_at": "2026-10-08T00:00:00Z",
                    "ext": {"samples": [{"__complex__": [3, 4]}]},
                },
                "components": {
                    "Q1": {
                        "kind": "lowering/driver",
                        "rate": 4300.0,
                        "wiring": {"ro_ch": [2, 5, 7]},
                        "flux": {"period": {"yoko1": 4.0, "yoko2": 8.0}},
                        "flux_dev": "yoko1",
                        "index": 1,
                    }
                },
            }
        )
        library = LibrarySnapshot(
            ModuleLibraryDocument(format="zcu.module-library", format_version="2.0")
        )
        yield ParameterContext(
            point=point,
            library=library,
            target=SavedPoint(entry_name="entry", point="working"),
            roles=RoleBinding(
                None,
                {"qubit": RoleSpec("lowering/driver")},
                {},
                aliases=(AliasRow("qubit", "Q1"), AliasRow("mux", "Q1")),
            ),
        )
    finally:
        component_registry.unregister("lowering/driver")
        assert component_registry.kinds() == before


def test_role_path_expression_uses_fixed_capture(parameters: ParameterContext) -> None:
    assert evaluate_scalar_expr("$qubit.rate * 2", parameters) == 8600.0
    assert evaluate_scalar_expr("Q1.rate + 100", parameters) == 4400.0


@pytest.mark.parametrize("target_type", [int, float])
@pytest.mark.parametrize("bounds", [(4400.0, 4500.0), (4300.0, 4400.0)])
def test_eval_ref_resolves_captured_paths_and_inclusive_bounds(
    parameters: ParameterContext, target_type: type, bounds: tuple[float, float]
) -> None:
    from zcu_tools.gui.session import expression

    ref = expression.EvalRef("Q1.rate + 100", target_type, *bounds)
    result = expression.resolve_eval_ref(ref, parameters)
    assert result == 4400
    assert type(result) is target_type


@pytest.mark.parametrize(
    ("expr", "target_type", "message"),
    [
        ("-1", float, "value -1.0 out of range"),
        ("11", int, "value 11 out of range"),
        ("1.5", int, "not an integer"),
        ("Q1.rate", float, "requires a context"),
        ("sqrt(-1)", float, "math domain error"),
    ],
)
def test_eval_ref_rejects_invalid_or_out_of_range_values(
    expr: str, target_type: type, message: str
) -> None:
    from zcu_tools.gui.session import expression

    with pytest.raises(ExpressionError, match=message):
        expression.resolve_eval_ref(expression.EvalRef(expr, target_type, 0, 10), None)


def test_eval_ref_source_failure_propagates_but_trial_returns_diagnostic(
    parameters: ParameterContext, caplog: pytest.LogCaptureFixture
) -> None:
    from zcu_tools.gui.session import expression
    from zcu_tools.gui.session.predictor_trial import PredictorTrial
    from zcu_tools.gui.session.services.predictor import PredictorService

    failure = OSError("alias source offline")
    capture = parameters.bind_aliases(
        (AliasRow("qubit", "Q1"),), {}, alias_error=failure
    )
    ref = expression.EvalRef("$qubit.rate", float, 0, 10000)
    with pytest.raises(OSError, match="alias source offline") as raised:
        expression.resolve_eval_ref(ref, capture)
    assert raised.value is failure
    trial = PredictorTrial(PredictorService())
    resolved = trial.resolve_calibration_frequency_mhz(ref, lambda: capture)
    assert resolved.frequency_mhz is None
    assert resolved.error == "alias source offline"
    assert not caplog.records


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("min(3, 2)", 2),
        ("max(3, 2)", 3),
        ("min(4.5, -2, 3, 0)", -2),
        ("max(4.5, -2, 3, 0)", 4.5),
        ("2 * max(min(8, 3 + 2), abs(-4), 1) - 1", 9),
        ("min($qubit.rate / 1000, 4)", 4),
    ],
)
def test_minmax_compare_multiple_numeric_arguments(
    parameters: ParameterContext, expression: str, expected: float
) -> None:
    assert evaluate_scalar_expr(expression, parameters) == expected
    assert evaluate_numeric_expr(expression, parameters) == float(expected)


@pytest.mark.parametrize(
    "expression",
    ["min()", "max()", "min(1)", "max(1)", "min(1j, 2j)", "max(1, 2j)"],
)
def test_minmax_reject_missing_arguments_and_unordered_numbers(expression: str) -> None:
    with pytest.raises(ExpressionError):
        evaluate_scalar_expr(expression, None)


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("$mux.wiring.ro_ch[0]", 2),
        ("$mux.wiring.ro_ch[-1]", 7),
        ("$mux.wiring.ro_ch[1 + 1]", 7),
        ("$mux.wiring.ro_ch[$qubit.index]", 5),
        ("$mux.wiring.ro_ch[min(2, 1)] * 2", 10),
    ],
)
def test_list_subscripts_read_integer_literals_and_expressions(
    parameters: ParameterContext, expression: str, expected: int
) -> None:
    assert evaluate_scalar_expr(expression, parameters) == expected
    assert evaluate_numeric_expr(expression, parameters) == float(expected)


@pytest.mark.parametrize(
    "expression",
    [
        "$mux.wiring.ro_ch[True]",
        "$mux.wiring.ro_ch[1.0]",
        "$mux.wiring.ro_ch[1j]",
        "$mux.wiring.ro_ch['0']",
        "$mux.wiring.ro_ch[3]",
        "$mux.wiring.ro_ch[-4]",
        "$qubit.rate[0]",
        "$qubit.flux_dev[0]",
        "$mux.wiring.ro_ch",
        "min($mux.wiring.ro_ch)",
        "max($mux.wiring.ro_ch, 1)",
    ],
)
def test_list_subscripts_and_containers_reject_invalid_numeric_use(
    parameters: ParameterContext, expression: str
) -> None:
    with pytest.raises(ExpressionError):
        evaluate_scalar_expr(expression, parameters)


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("$qubit.flux.period['yoko2']", 8.0),
        ("$qubit.flux.period[$qubit.flux_dev]", 4.0),
        ("$mux.wiring['ro_ch'][max(0, $qubit.index)]", 5.0),
        (
            "max($mux.wiring.ro_ch[0], min(10, $qubit.flux.period[$qubit.flux_dev] * 2))",
            8.0,
        ),
    ],
)
def test_mapping_subscripts_read_literal_and_captured_string_keys(
    parameters: ParameterContext, expression: str, expected: float
) -> None:
    assert evaluate_scalar_expr(expression, parameters) == expected
    assert evaluate_numeric_expr(expression, parameters) == expected


@pytest.mark.parametrize(
    "expression",
    [
        "$qubit.flux.period['missing']",
        "$qubit.flux.period[0]",
        "$qubit.flux.period[True]",
        "$qubit.flux.period[1.0]",
        "$qubit.flux.period[1j]",
        "$qubit.flux.period[$qubit.index]",
        "$qubit.flux.period['yoko' + '1']",
        "$qubit.flux.period[$qubit.flux_dev + '']",
        "$qubit.flux.period[$qubit]",
        "$qubit.flux.period",
        "'yoko1'",
        "$qubit.flux_dev",
        "'1' * 2",
        "- '1'",
        "min('1', '2')",
        "max($qubit.flux_dev, 'text')",
        "abs('1')",
    ],
)
def test_mapping_keys_and_strings_do_not_expand_the_numeric_contract(
    parameters: ParameterContext, expression: str
) -> None:
    with pytest.raises(ExpressionError):
        evaluate_scalar_expr(expression, parameters)


@pytest.mark.parametrize(
    "expression",
    [
        "$mux.wiring.ro_ch[0:1]",
        "$mux.wiring.ro_ch[0, 1]",
        "$mux.wiring.ro_ch[0].real",
        "$mux.wiring.ro_ch.__class__",
        "$qubit.flux.period.get('yoko1')",
        "$qubit.flux.period['yoko1']()",
        "$qubit.flux.period[f'yoko{1}']",
        "__subscript($mux.wiring.ro_ch, 0)",
        "min(*$mux.wiring.ro_ch)",
        "max(1, value=2)",
    ],
)
def test_subscripts_do_not_admit_python_attributes_calls_or_slices(
    parameters: ParameterContext, expression: str
) -> None:
    with pytest.raises(ExpressionError):
        evaluate_scalar_expr(expression, parameters)


def test_selected_complex_value_retains_scalar_and_real_only_boundaries(
    parameters: ParameterContext,
) -> None:
    expression = "general.ext.samples[0]"
    assert evaluate_scalar_expr(expression, parameters) == 3 + 4j
    assert evaluate_scalar_expr("abs(general.ext.samples[0])", parameters) == 5.0
    with pytest.raises(ExpressionError, match="real number"):
        evaluate_numeric_expr(expression, parameters)
    assert parameters.value("general.ext.samples") == [{"__complex__": [3, 4]}]


@pytest.mark.parametrize(
    "expression",
    ["$mux.wiring.ro_ch[0]", "$qubit.flux.period[$qubit.flux_dev]"],
)
def test_subscript_source_faults_propagate_unchanged(
    parameters: ParameterContext, expression: str
) -> None:
    failure = OSError("capture source unavailable")
    capture = parameters.bind_aliases(
        (AliasRow("qubit", "Q1"), AliasRow("mux", "Q1")), {}, alias_error=failure
    )
    with pytest.raises(OSError, match="capture source unavailable") as raised:
        evaluate_scalar_expr(expression, capture)
    assert raised.value is failure
    with pytest.raises(OSError, match="capture source unavailable") as raised:
        evaluate_numeric_expr(expression, capture)
    assert raised.value is failure


def test_no_parameters_allows_math_but_not_parameter_paths() -> None:
    assert evaluate_scalar_expr("sqrt(9) + 2", None) == 5.0
    with pytest.raises(ExpressionError):
        evaluate_scalar_expr("$qubit.rate", None)
