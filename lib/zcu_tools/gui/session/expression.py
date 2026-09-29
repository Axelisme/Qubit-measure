"""Restricted shared scalar expressions, evaluated by simpleeval.

Inputs use registered math functions, numeric MetaDict names and constants.
Attribute access, indexing and arbitrary calls are not part of this language.
"""

from __future__ import annotations

import ast
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from simpleeval import DEFAULT_OPERATORS, InvalidExpression, SimpleEval, safe_power

if TYPE_CHECKING:
    from zcu_tools.resources.context import MetaDict

Scalar = int | float | complex
_MAX_TEXT = 4096
_MAX_NODES = 512
_MAX_DEPTH = 64
_MAX_INTEGER_BITS = 4096
_MAX_EXPONENT = 1024
_FUNCTIONS: dict[str, Callable[..., object]] = {
    "sin": math.sin,
    "cos": math.cos,
    "tan": math.tan,
    "sqrt": math.sqrt,
    "exp": math.exp,
    "log": math.log,
    "log10": math.log10,
    "abs": abs,
}
_CONSTANTS = {"pi": math.pi, "e": math.e}
_OPERATOR_TYPES = (
    ast.Add,
    ast.Sub,
    ast.Mult,
    ast.Div,
    ast.FloorDiv,
    ast.Mod,
    ast.Pow,
    ast.BitXor,
    ast.UAdd,
    ast.USub,
)
_ALLOWED_NODES = (
    ast.Expression,
    ast.Constant,
    ast.Name,
    ast.Load,
    ast.BinOp,
    ast.UnaryOp,
    ast.Call,
    *_OPERATOR_TYPES,
)


@dataclass(frozen=True)
class EvalRef:
    """Unresolved device-dialog expression, accepted at apply time.

    Bounds and target type belong to the field. This marker neither persists nor
    depends on cfg EvalValue or app-specific machinery.
    """

    expr: str
    type_: type
    minimum: float
    maximum: float


def evaluate_numeric_expr(expr: str, md: MetaDict) -> float:
    """Evaluate a real-valued expression for device and real-only inputs."""
    value = evaluate_scalar_expr(expr, md)
    if isinstance(value, complex):
        raise RuntimeError("Expression must resolve to a real number")
    try:
        return float(value)
    except OverflowError as exc:
        raise RuntimeError("Expression result exceeds the real number range") from exc


def evaluate_scalar_expr(expr: str, md: MetaDict) -> Scalar:
    """Evaluate finite arithmetic, preserving integers and complex values.

    ``**`` is power; ``^`` keeps simpleeval's native integer XOR semantics.
    Functions use math's real domains except abs, which also accepts complex.
    Only referenced reserved names are checked for MetaDict collisions.
    """
    tree = _parse_expression(expr)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            _require_unshadowed(node.func.id, md)

    def lookup(node: ast.Name) -> Scalar:
        if node.id in _FUNCTIONS:
            _require_unshadowed(node.id, md)
            raise RuntimeError(f"Function {node.id!r} must be called")
        if node.id in _CONSTANTS:
            _require_unshadowed(node.id, md)
            return _CONSTANTS[node.id]
        try:
            value = getattr(md, node.id)
        except AttributeError as exc:
            raise RuntimeError(
                f"Variable {node.id!r} is not defined in MetaDict"
            ) from exc
        return _number(value, f"MetaDict variable {node.id!r}")

    operators = {
        kind: _checked(_power if kind is ast.Pow else DEFAULT_OPERATORS[kind])
        for kind in _OPERATOR_TYPES
    }
    evaluator = SimpleEval(
        operators=operators,
        functions={name: _checked(function) for name, function in _FUNCTIONS.items()},
        names=lookup,
    )
    try:
        result = evaluator.eval(expr, previously_parsed=tree.body)
    except InvalidExpression as exc:
        raise RuntimeError(str(exc)) from exc
    return _number(result, "Expression result")


def coerce_eval_result(value: float, type_: type) -> int | float:
    """Coerce an evaluated float to the target scalar type without surprises."""
    _number(value, "Expression result")
    if type_ is float:
        return float(value)
    if type_ is int:
        if not float(value).is_integer():
            raise RuntimeError(f"Expression result {value!r} is not an integer")
        return int(value)
    raise RuntimeError(f"Eval mode only supports int or float, got {type_!r}")


def _parse_expression(expr: str) -> ast.Expression:
    if not expr.strip():
        raise RuntimeError("Expression must not be empty")
    if len(expr) > _MAX_TEXT:
        raise RuntimeError("Expression exceeds the text length limit")
    try:
        tree = ast.parse(expr, mode="eval")
    except (SyntaxError, RecursionError) as exc:
        raise RuntimeError(f"Invalid expression syntax: {expr!r}") from exc
    pending: list[tuple[ast.AST, int]] = [(tree, 0)]
    count = 0
    while pending:
        node, depth = pending.pop()
        count += 1
        if count > _MAX_NODES or depth > _MAX_DEPTH:
            raise RuntimeError("Expression exceeds the syntax complexity limit")
        if not isinstance(node, _ALLOWED_NODES):
            raise RuntimeError(f"Unsupported expression syntax: {type(node).__name__}")
        if isinstance(node, ast.Constant):
            _number(node.value, "Expression constant")
        if isinstance(node, ast.Call) and (
            not isinstance(node.func, ast.Name)
            or node.func.id not in _FUNCTIONS
            or node.keywords
        ):
            raise RuntimeError("Unsupported expression syntax: function call")
        pending.extend((child, depth + 1) for child in ast.iter_child_nodes(node))
    return tree


def _require_unshadowed(name: str, md: MetaDict) -> None:
    try:
        getattr(md, name)
    except AttributeError:
        return
    raise RuntimeError(
        f"MetaDict name {name!r} conflicts with a reserved expression name"
    )


def _number(value: object, label: str) -> Scalar:
    if isinstance(value, bool) or not isinstance(value, (int, float, complex)):
        raise RuntimeError(f"{label} is not numeric")
    if isinstance(value, int):
        if value.bit_length() > _MAX_INTEGER_BITS:
            raise RuntimeError(f"{label} exceeds the integer size limit")
    elif not (math.isfinite(value.real) and math.isfinite(value.imag)):
        raise RuntimeError(f"{label} must be finite")
    return value


def _checked(function: Callable[..., object]) -> Callable[..., Scalar]:
    def call(*args: Scalar) -> Scalar:
        try:
            result = function(*args)
        except (ArithmeticError, TypeError, ValueError) as exc:
            raise RuntimeError(f"Expression arithmetic failed: {exc}") from exc
        return _number(result, "Expression result")

    return call


def _power(base: Scalar, exponent: Scalar) -> Scalar:
    if abs(exponent) > _MAX_EXPONENT:
        raise RuntimeError("Expression exceeds the exponent limit")
    if (
        isinstance(base, int)
        and isinstance(exponent, int)
        and exponent > 0
        and max(0, abs(base).bit_length() - 1) * exponent > _MAX_INTEGER_BITS
    ):
        raise RuntimeError("Expression exceeds the integer size limit")
    return _number(safe_power(base, exponent), "Power result")
