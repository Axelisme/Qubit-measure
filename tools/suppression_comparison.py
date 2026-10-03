"""Compare type-checker ignores by their source identity and diagnostic scope.

An unchanged AST statement in the same owner/branch keeps its identity through
comment deletion and formatting. Ambiguous statements receive no migration
credit. Raw usage counts remain the responsibility of check_suppressions.
"""

from __future__ import annotations

import ast
import io
import re
import tokenize
from collections import Counter
from dataclasses import dataclass
from typing import Literal

from check_suppressions import COMMENT_SUPPRESSIONS

IgnoreKind = Literal["type-ignore", "pyright-ignore"]
IgnoreReason = Literal[
    "new-position", "unproven-position", "scope-expanded", "not-narrow"
]
IGNORE_KINDS: tuple[IgnoreKind, ...] = ("type-ignore", "pyright-ignore")
_STATEMENT_FIELDS = frozenset({"body", "orelse", "finalbody", "handlers", "cases"})


@dataclass(frozen=True)
class IgnoreRegression:
    """One new or widened escape site, located in the candidate source."""

    line: int
    kind: IgnoreKind
    reason: IgnoreReason


@dataclass(frozen=True)
class _Comment:
    line: int
    column: int
    kind: IgnoreKind
    codes: frozenset[str] | None
    valid: bool


@dataclass(frozen=True)
class _Statement:
    node: ast.stmt
    key: tuple[str, ...]
    unique_path: bool


@dataclass(frozen=True)
class _SourceSites:
    comments: tuple[tuple[_Comment, tuple[str, ...] | None], ...]
    parsed: bool


def _scope(tail: str) -> tuple[frozenset[str] | None, bool]:
    tail = tail.lstrip()
    if not tail.startswith("["):
        return None, True
    match = re.match(r"\[([^\]]*)\]", tail)
    if match is None:
        return None, False
    codes = tuple(code.strip() for code in match[1].split(","))
    if not all(re.fullmatch(r"report[A-Z][A-Za-z0-9]*", code) for code in codes):
        return None, False
    return frozenset(codes), True


def _comments(source: str) -> tuple[_Comment, ...]:
    found: list[_Comment] = []
    try:
        for token in tokenize.generate_tokens(io.StringIO(source).readline):
            if token.type != tokenize.COMMENT:
                continue
            for kind in IGNORE_KINDS:
                match = COMMENT_SUPPRESSIONS[kind].match(token.string)
                if match is None:
                    continue
                # Pyright treats every type: ignore, even bracketed ones, as
                # blanket. Only its own ignore directive filters diagnostics.
                tail = token.string[match.end() :]
                if tail and not (tail[0].isspace() or tail[0] == "["):
                    codes, valid = None, False
                else:
                    codes, valid = (
                        _scope(tail) if kind == "pyright-ignore" else (None, True)
                    )
                found.append(_Comment(*token.start, kind, codes, valid))
    except (tokenize.TokenError, IndentationError, SyntaxError):
        # Preserve already observed comments, but AST provenance below must
        # still establish a position before any migration can be credited.
        pass
    return tuple(found)


def _value_key(value: object) -> str:
    if isinstance(value, ast.AST):
        return ast.dump(value, include_attributes=False)
    if isinstance(value, list):
        return repr(tuple(_value_key(item) for item in value))
    return repr(value)


def _header(node: ast.AST) -> str:
    fields = tuple(
        (name, _value_key(value))
        for name, value in ast.iter_fields(node)
        if name not in _STATEMENT_FIELDS
    )
    return repr((type(node).__name__, fields))


def _statements(source: str) -> tuple[_Statement, ...] | None:
    try:
        tree = ast.parse(source)
    except SyntaxError:
        return None
    found: list[tuple[ast.stmt, tuple[str, ...]]] = []
    occurrences: Counter[tuple[str, ...]] = Counter()

    def visit(node: ast.AST, owner: tuple[str, ...]) -> None:
        header = _header(node)
        key = (*owner, header)
        occurrences[key] += 1
        if isinstance(node, ast.stmt):
            found.append((node, key))
        for field, value in ast.iter_fields(node):
            children = value if isinstance(value, list) else [value]
            for child in children:
                if isinstance(child, (ast.stmt, ast.ExceptHandler, ast.match_case)):
                    visit(child, (*owner, header, field))

    visit(tree, ())
    # Keys alternate node headers and child fields. Every node-prefix must be
    # unique: a unique leaf can still belong to indistinguishable control owners.
    return tuple(
        _Statement(
            node,
            key,
            all(occurrences[key[:end]] == 1 for end in range(1, len(key) + 1, 2)),
        )
        for node, key in found
    )


def _line_role(node: ast.stmt, line: int) -> str:
    if line == node.lineno:
        return "start"
    if line == node.end_lineno:
        return "end"
    pieces = tuple(
        ast.dump(part, include_attributes=False)
        for part in ast.walk(node)
        if isinstance(part, (ast.expr, ast.arg, ast.alias))
        and part.lineno == line
        and part.end_lineno == line
    )
    return repr(pieces) if pieces else ""


def _unique_statement(
    line: int,
    statements: tuple[_Statement, ...],
) -> _Statement | None:
    covering = [
        statement
        for statement in statements
        if statement.node.lineno
        <= line
        <= (statement.node.end_lineno or statement.node.lineno)
    ]
    if not covering:
        return None
    smallest = min(
        (item.node.end_lineno or item.node.lineno) - item.node.lineno
        for item in covering
    )
    targets = [
        item
        for item in covering
        if (item.node.end_lineno or item.node.lineno) - item.node.lineno == smallest
    ]
    if len(targets) != 1:
        return None
    target = targets[0]
    return target if target.unique_path else None


def _anchor(
    comment: _Comment,
    lines: list[str],
    statements: tuple[_Statement, ...],
) -> tuple[str, ...] | None:
    if not lines[comment.line - 1][: comment.column].strip():
        return None
    target = _unique_statement(comment.line, statements)
    if target is None:
        return None
    role = _line_role(target.node, comment.line)
    if not role:
        return None
    if role not in {"start", "end"}:
        roles = Counter(
            _line_role(target.node, line)
            for line in range(
                target.node.lineno,
                (target.node.end_lineno or target.node.lineno) + 1,
            )
        )
        if roles[role] != 1:
            return None
    return (*target.key, role)


def _sites(source: str) -> _SourceSites:
    parsed = _statements(source)
    statements = () if parsed is None else parsed
    lines = source.split("\n")
    comments = tuple(
        (comment, _anchor(comment, lines, statements)) for comment in _comments(source)
    )
    return _SourceSites(comments, parsed is not None)


def _change_reason(before: _Comment, after: _Comment) -> IgnoreReason | None:
    if not before.valid:
        return "unproven-position"
    if not after.valid:
        return "not-narrow"
    if before.kind != after.kind:
        if (
            before.kind == "type-ignore"
            and after.kind == "pyright-ignore"
            and after.codes is not None
        ):
            return None
        return "not-narrow"
    if before.codes is None or (
        after.codes is not None and after.codes <= before.codes
    ):
        return None
    return "scope-expanded"


def compare_ignores(before: str, after: str) -> tuple[IgnoreRegression, ...]:
    """Return every new, broader or unproven type-checker escape site.

    Exact unchanged input preserves even ambiguous existing debt. Otherwise a
    unique AST statement, owner/branch and logical line role must prove the
    position. Type-ignore to pyright-ignore additionally requires a valid,
    nonempty diagnostic list. Deletion elsewhere never supplies credit.
    """
    if before == after:
        return ()
    old_sites = _sites(before)
    previous = {anchor: comment for comment, anchor in old_sites.comments if anchor}
    found: list[IgnoreRegression] = []
    for comment, anchor in _sites(after).comments:
        if anchor is None or not old_sites.parsed:
            reason: IgnoreReason | None = "unproven-position"
        elif anchor not in previous:
            reason = "new-position"
        else:
            reason = _change_reason(previous[anchor], comment)
        if reason is not None:
            found.append(IgnoreRegression(comment.line, comment.kind, reason))
    return tuple(found)
