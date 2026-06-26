from __future__ import annotations

import pytest
from zcu_tools.gui.session.expression import coerce_eval_result, evaluate_numeric_expr
from zcu_tools.meta_tool import MetaDict


def test_evaluate_numeric_expr_uses_metadict_variables():
    md = MetaDict()
    md.r_f = 6000.0
    md.rf_w = 2.0

    assert evaluate_numeric_expr("r_f - 1.5 * rf_w", md) == pytest.approx(5997.0)


@pytest.mark.parametrize(
    "expr",
    [
        "md.r_f",
        "sin(r_f)",
        "r_f[0]",
        "'abc'",
        "True",
    ],
)
def test_evaluate_numeric_expr_rejects_unsupported_syntax(expr: str):
    md = MetaDict()
    md.r_f = 6000.0

    with pytest.raises(RuntimeError):
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
