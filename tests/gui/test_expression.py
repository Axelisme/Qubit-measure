from __future__ import annotations

import pytest
from zcu_tools.format_version import YamlMap
from zcu_tools.gui.session.context_types import EntryDraft
from zcu_tools.gui.session.expression import (
    coerce_eval_result,
    evaluate_numeric_expr,
    evaluate_scalar_expr,
)
from zcu_tools.gui.session.parameters import ParameterContext
from zcu_tools.resources.context.library import LibrarySnapshot, ModuleLibraryDocument
from zcu_tools.resources.entry.schema import PointDocument


def _parameters(values: YamlMap) -> ParameterContext:
    return ParameterContext(
        point=PointDocument.model_validate(
            {
                "format": "zcu.parameter-container",
                "format_version": "1.0",
                "general": {"created_at": "2026-10-08T00:00:00Z", "ext": values},
                "components": {},
            }
        ),
        library=LibrarySnapshot(
            ModuleLibraryDocument(format="zcu.module-library", format_version="1.0")
        ),
        target=EntryDraft(entry_id="entry"),
        roles=None,
    )


def test_complex_scalar_expression_preserves_real_only_numeric_boundary():
    parameters = _parameters({"g_center": -1})
    expression = "general.ext.g_center + 2.5j"
    assert evaluate_scalar_expr(expression, parameters) == -1 + 2.5j
    with pytest.raises(RuntimeError, match="real number"):
        evaluate_numeric_expr(expression, parameters)
    with pytest.raises(RuntimeError, match="Parameter paths must start with a name"):
        evaluate_scalar_expr("(1+2j).real", parameters)


def test_evaluate_numeric_expr_uses_captured_parameter_paths():
    parameters = _parameters({"r_f": 6000.0, "rf_w": 2.0})
    assert evaluate_numeric_expr(
        "general.ext.r_f - 1.5 * general.ext.rf_w", parameters
    ) == pytest.approx(5997.0)


@pytest.mark.parametrize(
    "expr",
    [
        "unknown_function(r_f)",
        "sin(x=r_f)",
        "1; 2",
        "x = 2",
        "1 if r_f else 2",
        "1 < r_f",
        "[1, 2]",
        "1 << 2",
        "r_f[0]",
        "'abc'",
        "True",
    ],
)
def test_evaluate_numeric_expr_rejects_unsupported_syntax(expr: str):
    with pytest.raises(RuntimeError, match="syntax|not numeric"):
        evaluate_numeric_expr(expr, None)


@pytest.mark.parametrize("path", ["md.r_f", "general.ext.r_f", "r_f"])
def test_evaluate_numeric_expr_rejects_unknown_path(path: str):
    with pytest.raises(RuntimeError, match="Invalid parameter path"):
        evaluate_numeric_expr(path, _parameters({}))


def test_evaluate_numeric_expr_rejects_non_numeric_path():
    parameters = _parameters({"label": "resonator"})
    with pytest.raises(RuntimeError, match="not numeric"):
        evaluate_numeric_expr("general.ext.label", parameters)


@pytest.mark.parametrize("path", ["general.ext.r_f", "r_f", "q.modules.drive.gain"])
def test_parameter_paths_require_a_capture(path: str):
    with pytest.raises(RuntimeError, match="requires a context"):
        evaluate_numeric_expr(path, None)


def test_coerce_eval_result_rejects_non_integral_int():
    with pytest.raises(RuntimeError, match="not an integer"):
        coerce_eval_result(1.5, int)
    assert coerce_eval_result(2.0, int) == 2


@pytest.mark.parametrize(
    ("expression", "expected"),
    [
        ("sin(pi / 2)", 1.0),
        ("cos(0)", 1.0),
        ("tan(pi / 4)", 1.0),
        ("sqrt(9)", 3.0),
        ("exp(0)", 1.0),
        ("log(e)", 1.0),
        ("log(8, 2)", 3.0),
        ("log10(100)", 2.0),
        ("abs(-3)", 3),
        ("abs(3+4j)", 5.0),
        ("1+1j", 1 + 1j),
        ("(1+1j) * (1-1j)", 2 + 0j),
        ("(1+1j)**2", 2j),
        ("(-2)**2", 4),
        ("-2**2", -4),
        ("2**3**2", 512),
        ("2**-2", 0.25),
        ("7 // 2", 3),
        ("7 % 2", 1),
        ("6 ^ 3", 5),
        ("2 ^ 3 ^ 2", 3),
        ("2 + 3 ^ 2", 7),
    ],
)
def test_registered_math_and_native_operators(
    expression: str, expected: object
) -> None:
    assert evaluate_scalar_expr(expression, None) == pytest.approx(expected)


def test_variable_types_are_preserved_for_xor_and_complex() -> None:
    parameters = _parameters({"flags": 6, "offset": 1})
    assert evaluate_scalar_expr("general.ext.flags ^ 3", parameters) == 5
    assert type(evaluate_scalar_expr("general.ext.flags", parameters)) is int
    assert (
        evaluate_scalar_expr("general.ext.offset + 1j + sin(pi / 2)", parameters)
        == 2 + 1j
    )
    assert type(evaluate_numeric_expr("general.ext.flags", parameters)) is float


@pytest.mark.parametrize("expression", ["1j", "(-1)**0.5", "(1+1j)*(1-1j)"])
def test_real_only_boundary_does_not_drop_imaginary_type(expression: str) -> None:
    with pytest.raises(RuntimeError, match="real number"):
        evaluate_numeric_expr(expression, None)


@pytest.mark.parametrize(
    "expression",
    [
        "1.0 ^ 2",
        "1j ^ 2",
        "sqrt(-1)",
        "log(0)",
        "sin(1j)",
        "exp(1000)",
        "1 / 0",
        "0**-1",
        "10.0**1000",
    ],
)
def test_arithmetic_failures_keep_a_cause(expression: str) -> None:
    with pytest.raises(RuntimeError, match="arithmetic") as caught:
        evaluate_scalar_expr(expression, None)
    assert caught.value.__cause__ is not None


@pytest.mark.parametrize("expression", ["1e999", "1e308 * 10", "1e999j"])
def test_nonfinite_inputs_and_results_are_rejected(expression: str) -> None:
    with pytest.raises(RuntimeError, match="finite"):
        evaluate_scalar_expr(expression, None)


def test_parameter_names_do_not_shadow_math_constants_or_functions() -> None:
    parameters = _parameters({"pi": 2.0, "e": 3.0, "sin": 4.0})
    assert evaluate_scalar_expr("general.ext.pi + general.ext.e", parameters) == 5
    assert evaluate_scalar_expr("general.ext.sin + sin(pi / 2)", parameters) == 5
    assert evaluate_scalar_expr("log(e)", parameters) == 1


@pytest.mark.parametrize(
    ("expression", "message"),
    [
        ("1" * 4097, "text length"),
        ("1+" * 70 + "1", "complexity"),
        ("2**1025", "exponent limit"),
        ("(2**1000)**5", "integer size"),
    ],
)
def test_expression_resource_limits(expression: str, message: str) -> None:
    with pytest.raises(RuntimeError, match=message):
        evaluate_scalar_expr(expression, None)
