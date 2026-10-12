"""Restricted shared scalar expressions, evaluated by simpleeval.

Inputs use registered math functions, captured parameter paths and constants.
Dotted paths are owner-resolved data references, not Python attribute access.
Indexing and arbitrary calls are not part of this language.
"""

from __future__ import annotations

import ast
import io
import math
import tokenize
from collections.abc import Callable
from dataclasses import dataclass
from typing import TYPE_CHECKING

from simpleeval import DEFAULT_OPERATORS, InvalidExpression, SimpleEval, safe_power

from zcu_tools.core.complex_codec import decode_complex
from zcu_tools.gui.expected_error import InvalidInputError
from zcu_tools.gui.session.parameters import ModuleSlotError

if TYPE_CHECKING:
    from zcu_tools.gui.session.types import ParameterContext


class ExpressionError(InvalidInputError):
    """An expression is invalid, unlike a failure of its source provider."""


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
    ast.Attribute,
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


def evaluate_numeric_expr(expr: str, parameters: ParameterContext | None) -> float:
    """Evaluate finite real arithmetic over captured parameter paths.

    None permits pure calculation only; parameter/library paths raise
    ExpressionError. Numeric source, grammar and real-domain failures raise.
    """
    value = evaluate_scalar_expr(expr, parameters)
    if isinstance(value, (complex, str)):
        raise ExpressionError("Expression must resolve to a real number")
    try:
        return float(value)
    except OverflowError as exc:
        raise ExpressionError(
            "Expression result exceeds the real number range"
        ) from exc


def evaluate_scalar_expr(
    expr: str, parameters: ParameterContext | None
) -> Scalar | str:
    """Evaluate arithmetic paths or a standalone $alias name string.

    A standalone $alias returns its fixed resolved component/device name.
    Names cannot enter arithmetic, functions or interpolation. Name reads need
    a context and reject missing/unbound aliases with an alias-naming error.

    None permits pure calculation only; parameter/library references raise
    ExpressionError. Resolve paths through ParameterContext, never the live
    session. Retain numeric limits, math functions, constants and XOR semantics.
    Parameter complex tags decode through the shared scalar codec; the captured
    context itself still returns raw YAML. Malformed tags raise ExpressionError
    naming the path and retain the codec error as cause. Strings are not decoded.
    Invalid grammar, missing/non-numeric paths, unset/dangling module slots and
    arithmetic failures raise ExpressionError. Slot errors retain their path and
    reason; unexpected source failures propagate unchanged.
    """
    tree = _parse_expression(expr)
    if isinstance(tree.body, ast.Name) and tree.body.id.startswith("$"):
        return _read_alias_name(tree.body.id, parameters)
    paths: dict[str, Scalar] = {}
    resolve_path = _parameter_reader(parameters)

    # Replace complete dotted paths before simpleeval can perform attribute access.
    class CapturePaths(ast.NodeTransformer):
        def visit_Attribute(self, node: ast.Attribute) -> ast.Name:
            parts: list[str] = []
            root: ast.expr = node
            while isinstance(root, ast.Attribute):
                parts.append(root.attr)
                root = root.value
            if not isinstance(root, ast.Name):
                raise ExpressionError("Parameter paths must start with a name")
            path = ".".join([root.id, *reversed(parts)])
            name = f"__parameter_{len(paths)}"
            paths[name] = resolve_path(path)
            return ast.copy_location(ast.Name(id=name, ctx=ast.Load()), node)

    CapturePaths().visit(tree)

    def lookup(node: ast.Name) -> Scalar:
        if node.id in paths:
            return paths[node.id]
        if node.id in _FUNCTIONS:
            raise ExpressionError(f"Function {node.id!r} must be called")
        if node.id in _CONSTANTS:
            return _CONSTANTS[node.id]
        return resolve_path(node.id)

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
        raise ExpressionError(str(exc)) from exc
    return _number(result, "Expression result")


def _read_alias_name(alias: str, parameters: ParameterContext | None) -> str:
    """Translate name-read errors at the expression boundary, not source faults."""
    if parameters is None:
        raise ExpressionError(f"Alias {alias!r} requires a context")
    try:
        name = parameters.value(alias)
    except (ValueError, InvalidInputError) as exc:
        raise ExpressionError(f"Invalid alias {alias!r}: {exc}") from exc
    if not isinstance(name, str):
        raise ExpressionError(f"Alias {alias!r} must resolve to a name")
    return name


def validate_scalar_expr(expr: str) -> None:
    """Check expression grammar and limits without reading variable sources.

    Capture preparation uses this before accessing any marked reference. Missing
    variables and arithmetic/domain errors belong to evaluation, not validation.
    """
    _parse_expression(expr)


def coerce_eval_result(value: float, type_: type) -> int | float:
    """Coerce an evaluated float to the target scalar type without surprises."""
    _number(value, "Expression result")
    if type_ is float:
        return float(value)
    if type_ is int:
        if not float(value).is_integer():
            raise ExpressionError(f"Expression result {value!r} is not an integer")
        return int(value)
    raise ExpressionError(f"Eval mode only supports int or float, got {type_!r}")


def _parse_expression(expr: str) -> ast.Expression:
    if not expr.strip():
        raise ExpressionError("Expression must not be empty")
    if len(expr) > _MAX_TEXT:
        raise ExpressionError("Expression exceeds the text length limit")
    try:
        python_expr, aliases = _alias_syntax(expr)
        tree = ast.parse(python_expr, mode="eval")
    except (SyntaxError, RecursionError) as exc:
        raise ExpressionError(f"Invalid expression syntax: {expr!r}") from exc
    for node in ast.walk(tree):
        if isinstance(node, ast.Name) and (node.lineno, node.col_offset) in aliases:
            node.id = aliases[node.lineno, node.col_offset]
    pending: list[tuple[ast.AST, int]] = [(tree, 0)]
    count = 0
    while pending:
        node, depth = pending.pop()
        count += 1
        if count > _MAX_NODES or depth > _MAX_DEPTH:
            raise ExpressionError("Expression exceeds the syntax complexity limit")
        if not isinstance(node, _ALLOWED_NODES):
            raise ExpressionError(
                f"Unsupported expression syntax: {type(node).__name__}"
            )
        if isinstance(node, ast.Constant):
            _number(node.value, "Expression constant")
        if isinstance(node, ast.Call) and (
            not isinstance(node.func, ast.Name)
            or node.func.id not in _FUNCTIONS
            or node.keywords
        ):
            raise ExpressionError("Unsupported expression syntax: function call")
        pending.extend((child, depth + 1) for child in ast.iter_child_nodes(node))
    return tree


def _alias_syntax(expr: str) -> tuple[str, dict[tuple[int, int], str]]:
    """Translate lexical $alias roots; AST handles paths and arithmetic."""
    lines = expr.splitlines(keepends=True)
    offsets = [0]
    for line in lines:
        offsets.append(offsets[-1] + len(line))
    try:
        tokens = list(tokenize.generate_tokens(io.StringIO(expr).readline))
    except (tokenize.TokenError, IndentationError) as exc:
        raise ExpressionError(f"Invalid expression syntax: {expr!r}") from exc
    aliases: dict[tuple[int, int], str] = {}
    characters = list(expr)
    for index, token in enumerate(tokens):
        if token.string != "$" or token.type not in (tokenize.OP, tokenize.ERRORTOKEN):
            continue
        following = tokens[index + 1 : index + 2]
        if (
            not following
            or following[0].type != tokenize.NAME
            or following[0].start != token.end
        ):
            raise ExpressionError("Alias references require adjacent $alias")
        dotted = tokens[index + 2 : index + 4]
        if (
            dotted
            and dotted[0].string == "."
            and (
                len(dotted) != 2
                or dotted[0].start != following[0].end
                or dotted[1].type != tokenize.NAME
                or dotted[1].start != dotted[0].end
            )
        ):
            raise ExpressionError("Alias paths require adjacent $alias.path")
        line, column = token.start
        byte_column = len(lines[line - 1][:column].encode("utf-8"))
        aliases[line, byte_column] = "$" + following[0].string
        characters[offsets[line - 1] + column] = "_"
    return "".join(characters), aliases


def _parameter_reader(
    parameters: ParameterContext | None,
) -> Callable[[str], Scalar]:
    """Read aliases from parameters and bare paths from one concrete-only view.

    Construct the bare view lazily once per evaluation. Missing/invalid paths
    become ExpressionError; source/provider faults retain their classification.
    """
    concrete_parameters: ParameterContext | None = None

    def read(path: str) -> Scalar:
        nonlocal concrete_parameters
        if parameters is None:
            raise ExpressionError(f"Parameter path {path!r} requires a context")
        try:
            # Bare namespaces must not inherit notebook's legacy role shorthand.
            if path.startswith("$"):
                result = parameters.lookup(path)
            else:
                if concrete_parameters is None:
                    concrete_parameters = parameters.bind_aliases((), {})
                result = concrete_parameters.lookup(path)
        except (ValueError, InvalidInputError, ModuleSlotError) as exc:
            raise ExpressionError(f"Invalid parameter path {path!r}: {exc}") from exc
        if not result.present:
            raise ExpressionError(f"Invalid parameter path {path!r}: missing parameter")
        return _parameter_number(result.value, path)

    return read


def _parameter_number(value: object, path: str) -> Scalar:
    if isinstance(value, dict) and "__complex__" in value:
        try:
            value = decode_complex(value)
        except (ValueError, OverflowError) as exc:
            raise ExpressionError(
                f"Invalid complex tag at parameter path {path!r}: {exc}"
            ) from exc
    return _number(value, f"Parameter path {path!r}")


def _number(value: object, label: str) -> Scalar:
    if isinstance(value, bool) or not isinstance(value, (int, float, complex)):
        raise ExpressionError(f"{label} is not numeric")
    if isinstance(value, int):
        if value.bit_length() > _MAX_INTEGER_BITS:
            raise ExpressionError(f"{label} exceeds the integer size limit")
    elif not (math.isfinite(value.real) and math.isfinite(value.imag)):
        raise ExpressionError(f"{label} must be finite")
    return value


def _checked(function: Callable[..., object]) -> Callable[..., Scalar]:
    def call(*args: Scalar) -> Scalar:
        try:
            result = function(*args)
        except (ArithmeticError, TypeError, ValueError) as exc:
            raise ExpressionError(f"Expression arithmetic failed: {exc}") from exc
        return _number(result, "Expression result")

    return call


def _power(base: Scalar, exponent: Scalar) -> Scalar:
    if abs(exponent) > _MAX_EXPONENT:
        raise ExpressionError("Expression exceeds the exponent limit")
    if (
        isinstance(base, int)
        and isinstance(exponent, int)
        and exponent > 0
        and max(0, abs(base).bit_length() - 1) * exponent > _MAX_INTEGER_BITS
    ):
        raise ExpressionError("Expression exceeds the integer size limit")
    return _number(safe_power(base, exponent), "Power result")
