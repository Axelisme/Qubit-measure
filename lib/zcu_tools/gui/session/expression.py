"""Safe numeric expression evaluation for GUI scalar eval fields.

Moved from gui.app.main.expression to the session layer so that both the
cfg-editor (app/main) and the device dialog (session/ui) can share the same
evaluator without creating a session→app upward dependency.
"""

from __future__ import annotations

import ast
import logging
import operator
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from zcu_tools.meta_tool import MetaDict

_BIN_OPS: dict[type[ast.operator], Callable[[float, float], float]] = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
}
_UNARY_OPS: dict[type[ast.unaryop], Callable[[float], float]] = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


@dataclass(frozen=True)
class EvalRef:
    """Marker returned by EvalNumericField.read_raw() when in eval mode.

    Carries the raw expression, the target scalar type, and the inclusive
    [minimum, maximum] bounds the resolved value must satisfy. The device dialog
    resolves it against the current MetaDict at apply time (Design 1: resolve
    once at apply, not per-keystroke). This type must NOT leak into or depend on
    app/main adapter machinery (EvalValue is a different, adapter-bound type).
    """

    expr: str
    type_: type
    minimum: float
    maximum: float


def evaluate_numeric_expr(expr: str, md: MetaDict) -> float:
    """Evaluate a restricted numeric expression against MetaDict attributes."""
    if not expr.strip():
        raise RuntimeError("Expression must not be empty")
    try:
        tree = ast.parse(expr, mode="eval")
    except SyntaxError as exc:
        raise RuntimeError(f"Invalid expression syntax: {expr!r}") from exc
    return _eval_node(tree, md)


def coerce_eval_result(value: float, type_: type) -> int | float:
    """Coerce an evaluated float to the target scalar type without surprises."""
    if type_ is float:
        return float(value)
    if type_ is int:
        if not float(value).is_integer():
            raise RuntimeError(f"Expression result {value!r} is not an integer")
        return int(value)
    raise RuntimeError(f"Eval mode only supports int or float, got {type_!r}")


def _eval_node(node: ast.AST, md: MetaDict) -> float:
    if isinstance(node, ast.Expression):
        return _eval_node(node.body, md)
    if isinstance(node, ast.Constant):
        if isinstance(node.value, bool) or not isinstance(node.value, (int, float)):
            raise RuntimeError("Only numeric constants are allowed")
        return float(node.value)
    if isinstance(node, ast.Name):
        try:
            value = getattr(md, node.id)
        except AttributeError as exc:
            raise RuntimeError(
                f"Variable {node.id!r} is not defined in MetaDict"
            ) from exc
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise RuntimeError(f"MetaDict variable {node.id!r} is not numeric")
        return float(value)
    if isinstance(node, ast.BinOp):
        op = _BIN_OPS.get(type(node.op))
        if op is None:
            raise RuntimeError(f"Unsupported binary operator: {type(node.op).__name__}")
        return op(_eval_node(node.left, md), _eval_node(node.right, md))
    if isinstance(node, ast.UnaryOp):
        op = _UNARY_OPS.get(type(node.op))
        if op is None:
            raise RuntimeError(f"Unsupported unary operator: {type(node.op).__name__}")
        return op(_eval_node(node.operand, md))
    raise RuntimeError(f"Unsupported expression syntax: {type(node).__name__}")
