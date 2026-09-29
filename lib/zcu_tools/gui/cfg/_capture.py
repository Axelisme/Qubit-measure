"""Capture token positions without defining a second arithmetic language."""

import io
import math
import tokenize
from collections.abc import Callable

from zcu_tools.gui.expected_error import InvalidInputError

from .resource import (
    CfgInputError,
    CfgInputReason,
    CfgPreconditionError,
    CfgPreconditionReason,
)


class ExpressionCapture:
    """Use one command's source view and memoize repeated capture references."""

    def __init__(
        self,
        read: Callable[[str], object],
        validate: Callable[[str], None],
    ) -> None:
        self._read = read
        self._validate = validate
        self._literals: dict[str, str] = {}

    def prepare(self, expression: str) -> str:
        spans = _capture_spans(expression)
        if not spans:
            return expression
        skeleton = _substitute(expression, spans, lambda name: "(0)")
        try:
            self._validate(skeleton)
        except InvalidInputError as exc:
            raise CfgInputError(CfgInputReason.CAPTURE_SYNTAX, str(exc)) from exc
        prepared = _substitute(expression, spans, self._literal)
        try:
            self._validate(prepared)
        except InvalidInputError as exc:
            raise CfgInputError(CfgInputReason.INVALID_VALUE, str(exc)) from exc
        return prepared

    def _literal(self, name: str) -> str:
        if name not in self._literals:
            try:
                value = self._read(name)
            except KeyError as exc:
                raise CfgPreconditionError(
                    CfgPreconditionReason.CAPTURE_UNAVAILABLE,
                    f"Capture source {name!r} is unavailable",
                ) from exc
            if isinstance(value, bool) or not isinstance(value, (int, float, complex)):
                raise CfgInputError(
                    CfgInputReason.INVALID_VALUE, "capture must be numeric"
                )
            if isinstance(value, (float, complex)) and not (
                math.isfinite(value.real) and math.isfinite(value.imag)
            ):
                raise CfgInputError(
                    CfgInputReason.INVALID_VALUE, "capture must be finite"
                )
            try:
                self._literals[name] = f"({value!r})"
            except ValueError as exc:
                raise CfgInputError(
                    CfgInputReason.INVALID_VALUE, "capture literal is too large"
                ) from exc
        return self._literals[name]


def _capture_spans(expression: str) -> list[tuple[int, int, str]]:
    # Incremental tokenization preserves the no-capture/incomplete-input path.
    tokens: list[tokenize.TokenInfo] = []
    try:
        tokens.extend(tokenize.generate_tokens(io.StringIO(expression).readline))
    except (tokenize.TokenError, IndentationError) as exc:
        if any(
            token.string == "$" and token.type in (tokenize.OP, tokenize.ERRORTOKEN)
            for token in tokens
        ):
            raise CfgInputError(CfgInputReason.CAPTURE_SYNTAX, str(exc)) from exc
        return []
    offsets = [0]
    for line in expression.splitlines(keepends=True):
        offsets.append(offsets[-1] + len(line))
    spans: list[tuple[int, int, str]] = []
    for index, token in enumerate(tokens):
        if token.string != "$" or token.type not in (tokenize.OP, tokenize.ERRORTOKEN):
            continue
        start = offsets[token.start[0] - 1] + token.start[1]
        end, name = _reference(tokens, index)
        spans.append((start, offsets[end[0] - 1] + end[1], name))
    return spans


def _reference(
    tokens: list[tokenize.TokenInfo], index: int
) -> tuple[tuple[int, int], str]:
    previous = tokens[index]
    parts: list[str] = []
    index += 1
    while index < len(tokens):
        name = tokens[index]
        if name.type != tokenize.NAME or name.start != previous.end:
            break
        parts.append(name.string)
        index += 1
        if index >= len(tokens) or tokens[index].string != ".":
            return name.end, ".".join(parts)
        previous = tokens[index]
        if previous.start != name.end:
            break
        index += 1
    raise CfgInputError(
        CfgInputReason.CAPTURE_SYNTAX, "capture requires an adjacent dotted identifier"
    )


def _substitute(
    expression: str,
    spans: list[tuple[int, int, str]],
    literal: Callable[[str], str],
) -> str:
    pieces: list[str] = []
    cursor = 0
    for start, end, name in spans:
        pieces.extend((expression[cursor:start], literal(name)))
        cursor = end
    pieces.append(expression[cursor:])
    return "".join(pieces)
