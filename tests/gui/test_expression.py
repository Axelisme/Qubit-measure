from __future__ import annotations

import pytest
from zcu_tools.gui.session.expression import (
    coerce_eval_result,
    evaluate_numeric_expr,
    evaluate_scalar_expr,
)
from zcu_tools.resources.context import MetaDict


def test_complex_scalar_expression_preserves_real_only_numeric_boundary():
    md = MetaDict()
    md.g_center = -1 + 2j
    assert evaluate_scalar_expr("g_center + 0.5j", md) == -1 + 2.5j
    with pytest.raises(RuntimeError, match="real number"):
        evaluate_numeric_expr("g_center", md)
    with pytest.raises(RuntimeError, match="Unsupported expression syntax"):
        evaluate_scalar_expr("g_center.real", md)


def test_evaluate_numeric_expr_uses_metadict_variables():
    md = MetaDict()
    md.r_f = 6000.0
    md.rf_w = 2.0

    assert evaluate_numeric_expr("r_f - 1.5 * rf_w", md) == pytest.approx(5997.0)


@pytest.mark.parametrize(
    "expr",
    [
        "md.r_f",
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
    md = MetaDict()
    md.r_f = 6000.0

    with pytest.raises(RuntimeError, match="syntax|not numeric"):
        evaluate_numeric_expr(expr, md)


def test_evaluate_numeric_expr_rejects_unknown_or_non_numeric_name():
    md = MetaDict()
    md.label = "resonator"

    with pytest.raises(RuntimeError, match="not defined"):
        evaluate_numeric_expr("r_f", md)
    with pytest.raises(RuntimeError, match="not numeric"):
        evaluate_numeric_expr("label", md)


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
    assert evaluate_scalar_expr(expression, MetaDict()) == pytest.approx(expected)


def test_variable_types_are_preserved_for_xor_and_complex() -> None:
    md = MetaDict()
    md.flags = 6
    md.offset = 1 + 1j
    assert evaluate_scalar_expr("flags ^ 3", md) == 5
    assert type(evaluate_scalar_expr("flags", md)) is int
    assert evaluate_scalar_expr("offset + sin(pi / 2)", md) == 2 + 1j
    assert type(evaluate_numeric_expr("flags", md)) is float


@pytest.mark.parametrize("expression", ["1j", "(-1)**0.5", "(1+1j)*(1-1j)"])
def test_real_only_boundary_does_not_drop_imaginary_type(expression: str) -> None:
    with pytest.raises(RuntimeError, match="real number"):
        evaluate_numeric_expr(expression, MetaDict())


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
        evaluate_scalar_expr(expression, MetaDict())
    assert caught.value.__cause__ is not None


@pytest.mark.parametrize("expression", ["1e999", "1e308 * 10", "1e999j"])
def test_nonfinite_inputs_and_results_are_rejected(expression: str) -> None:
    with pytest.raises(RuntimeError, match="finite"):
        evaluate_scalar_expr(expression, MetaDict())


@pytest.mark.parametrize(
    ("name", "expression"), [("pi", "pi"), ("e", "log(e)"), ("sin", "sin(0)")]
)
def test_referenced_reserved_names_cannot_shadow_sources(
    name: str, expression: str
) -> None:
    md = MetaDict()
    setattr(md, name, 2.0)
    with pytest.raises(RuntimeError, match="conflicts"):
        evaluate_scalar_expr(expression, md)
    assert evaluate_scalar_expr("1+2", md) == 3


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
        evaluate_scalar_expr(expression, MetaDict())
