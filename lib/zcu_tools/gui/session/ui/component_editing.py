"""Field text parsing and error presentation shared by component editors.

The Inspect component tree and the Add component dialog use the same rules:
one parser for typed field text and one translation from owner exceptions to
a short visible line. Owners keep raising their own exceptions; only the
display changes here.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import yaml
from pydantic import TypeAdapter, ValidationError
from qtpy.QtWidgets import QLabel

from zcu_tools.format_version import YamlValue
from zcu_tools.gui.widgets import error_colour, reset_foreground, set_foreground
from zcu_tools.resources.entry import FieldSpec, FieldValueKind, role_candidates

if TYPE_CHECKING:
    from zcu_tools.resources.entry.schema import PointDocument, SetupDocument


def parse_field_input(value_kind: FieldValueKind, raw: str) -> YamlValue:
    """Convert editor text for a field of value_kind into the value to submit.

    "text" and "choice" fields return raw unchanged. Every other kind parses
    raw as one YAML value ("2" -> 2, "true" -> True, "{ch: 2}" -> {"ch": 2});
    the model owner validates the result later. Text that is not valid YAML,
    or YAML that is not a plain scalar, list or string-keyed mapping, raises
    ValueError with a one-line message and the parser error as its cause.
    """
    if value_kind in ("text", "choice"):
        return raw
    try:
        loaded = yaml.safe_load(raw)
    except yaml.YAMLError as exc:
        # Marked errors carry the actionable part ("expected ',' or ']'") apart
        # from their position context; other parser errors keep their text.
        problem = exc.problem if isinstance(exc, yaml.MarkedYAMLError) else None
        raise ValueError(f"Invalid YAML: {problem or exc}") from exc
    try:
        return TypeAdapter(YamlValue).validate_python(loaded, strict=True)
    except ValidationError as exc:
        raise ValueError(
            "Invalid YAML: expected a scalar, list or string-keyed mapping"
        ) from exc


def field_choices(
    value_kind: FieldValueKind, choices: tuple[str, ...]
) -> tuple[str, ...]:
    """Return the drop-down options for a field, or () for a free-text editor.

    "boolean" fields choose between "true" and "false"; "choice" fields use
    their declared Literal choices. Options never include a blank entry;
    editors add one themselves when unset must stay selectable.
    """
    if value_kind == "boolean":
        return ("true", "false")
    return choices


def edit_choices(
    document: SetupDocument | PointDocument, spec: FieldSpec
) -> tuple[str, ...]:
    """Return a field's drop-down options, or () for a free-text editor.

    A role reference (spec.reference_role) offers document's components whose
    kind matches the role, in document order; other fields use field_choices.
    """
    if spec.reference_role is not None:
        return role_candidates(document, spec.reference_role)
    return field_choices(spec.value_kind, spec.choices)


def edit_completions(
    document: SetupDocument | PointDocument, spec: FieldSpec
) -> tuple[str, ...]:
    """Return component names a free-text "text" field completes, else ().

    Role references use a drop-down instead (see edit_choices), so they get
    no completions; the editor still accepts any text.
    """
    if spec.value_kind != "text" or spec.reference_role is not None:
        return ()
    return tuple(document.components)


def field_tooltip(spec: FieldSpec, path: str) -> str:
    """Return the description (when declared) above the field's path."""
    return path if spec.description is None else f"{spec.description}\n{path}"


def error_summary(exc: BaseException) -> str:
    """Return the short visible text for a failed component write.

    When exc, or an exception it was raised from (__cause__), is a pydantic
    ValidationError, every entry of its errors() becomes one line
    "<field path>: <message>", for example "t1: Input should be a valid
    number"; an entry without a location shows only its message. Other
    exceptions show the first nonblank line of str(exc), or the exception
    type name when that is empty. The full text belongs in a tooltip.
    """
    cause: BaseException | None = exc
    while cause is not None:
        if isinstance(cause, ValidationError):
            return "\n".join(
                _located(".".join(str(part) for part in error["loc"]), error["msg"])
                for error in cause.errors()
            )
        cause = cause.__cause__
    lines = [line.strip() for line in str(exc).splitlines() if line.strip()]
    return lines[0] if lines else type(exc).__name__


def show_error(label: QLabel, exc: BaseException, *, field: str | None = None) -> None:
    """Show exc on label as error_summary text with str(exc) as the tooltip.

    field, when given, prefixes the summary as "<field>: <summary>"; use it
    for local failures, such as unparsable text, that name no field
    themselves. The label becomes visible with the current theme error foreground.
    """
    summary = error_summary(exc)
    label.setText(summary if field is None else f"{field}: {summary}")
    label.setToolTip(str(exc))
    set_error_foreground(label, error=True)
    label.setVisible(True)


def set_error_foreground(label: QLabel, *, error: bool) -> None:
    """Apply the label's theme error colour, or restore inheritance when error is False."""
    if error:
        set_foreground(label, error_colour(label))
    else:
        reset_foreground(label)


def _located(location: str, message: str) -> str:
    return f"{location}: {message}" if location else message
