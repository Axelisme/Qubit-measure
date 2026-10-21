"""Measure Form wire projection, using each Form owner's fixed capture."""

from __future__ import annotations

from collections.abc import Mapping
from typing import NotRequired, TypedDict

from pydantic import JsonValue, TypeAdapter

from zcu_tools.cfg_model import ConfigBase
from zcu_tools.gui.app.measure.services.tab_form import TabFormResource
from zcu_tools.gui.form import FormErrors, FormPath, FormPreview, FormWritePath
from zcu_tools.gui.remote.errors import ErrorCode, RemoteError
from zcu_tools.gui.session.library_entry import (
    LibraryEntryInvalid,
    ModuleEntryForm,
    WaveformEntryForm,
)

from .form_description import RecipeFieldObservation, recipe_field_wire
from .form_errors import FormError
from .handlers._wire_values import context_wire_value


class FormWrite(TypedDict):
    """One ordinary leaf: path has field names/FixedList indexes; value is JSON."""

    path: list[str | int]
    value: JsonValue


class FormObservation(TypedDict):
    """Detached Form view, not an acceptance or a source refresh.

    input retains missing
    fields and expression tags. values contains evaluated ordinary leaves, not
    proof of validity. errors includes input and lowered cfg validation failures.
    """

    input: dict[str, JsonValue]
    values: list[FormWrite]
    errors: list[FormError]


class EditPathHints(TypedDict):
    """Active canonical leaves and selectors for correcting a rejected cfg edit.

    editable_paths excludes hidden, fixed and readonly subtrees. selection_paths
    identifies active option fields; module choices use select_recipe_module
    with logical paths rather than replacing them with a key string.
    """

    editable_paths: list[list[str | int]]
    selection_paths: list[list[str | int]]


def edit_path_hints(resource: TabFormResource) -> EditPathHints:
    """List active editable paths from owner descriptions without evaluating input.

    resource is the current fixed tab Form. Paths remain canonical, including
    generated wrappers. Ref/disabled descendants are not editable; their selector
    path is listed instead. This never changes selections or captures sources.
    """
    hints: EditPathHints = {"editable_paths": [], "selection_paths": []}
    pending: list[FormWritePath] = [()]
    while pending:
        description = resource.describe(pending.pop())
        for field in description.children:
            if field.hidden or field.readonly or field.fixed:
                continue
            if field.options is not None:
                hints["selection_paths"].append(list(field.path))
            if field.can_expand:
                pending.append(field.path)
            elif field.options is None:
                hints["editable_paths"].append(list(field.path))
    return hints


def observe_form(resource: TabFormResource) -> FormObservation:
    """Project current input/preview/errors without capture or mutation."""
    return _observation(
        resource.input.data,
        resource.preview(),
        resource.result(),
    )


class ElementObservationWire(TypedDict):
    """One original element with canonical input, preview and directed errors.

    fields, when requested, contains same-turn logical observations in relative
    path request order. Each observation carries the owner-captured identity.
    """

    index: int
    name: str
    path: list[str | int]
    input: dict[str, JsonValue]
    selections: list[FormWrite]
    values: list[FormWrite]
    errors: list[FormError]
    fields: NotRequired[list[RecipeFieldObservation]]


class ElementsObservationWire(TypedDict):
    """The owner's model-list aggregate and complete Form errors, in one turn."""

    path: list[str | int]
    elements: list[ElementObservationWire]
    errors: list[FormError]


def observe_elements(
    resource: TabFormResource,
    path: FormWritePath,
    *,
    fields: tuple[FormWritePath, ...] = (),
) -> ElementsObservationWire:
    """Serialize the owner's single aggregate; never assemble per-element reads.

    path is the canonical model-list path. All rows retain original indexes,
    authored input, available preview leaves and directed errors. Reads do not
    lower cfg, capture sources, mutate state or focus a widget. fields optionally
    requests nonempty relative logical paths in each element, in request order.
    Invalid paths propagate owner errors; refs use only fixed captured sources.
    """
    observed = resource.read_elements(path, fields=fields)
    rows: list[ElementObservationWire] = [
        {
            "index": item.index,
            "name": item.name,
            "path": list(item.path),
            "input": item.input,
            "selections": [
                {"path": list(key), "value": value}
                for key, value in item.selections.items()
            ],
            "values": [
                {"path": list(key), "value": context_wire_value(value)}
                for key, value in item.values.items()
            ],
            "errors": _error_rows(item.errors.messages) if item.errors else [],
        }
        for item in observed.elements
    ]
    if fields:
        for row, item in zip(rows, observed.elements, strict=True):
            row["fields"] = [recipe_field_wire(field) for field in item.fields]
    return {
        "path": list(observed.path),
        "elements": rows,
        "errors": _error_rows(observed.errors.messages) if observed.errors else [],
    }


def observe_entry(entry: ModuleEntryForm | WaveformEntryForm) -> FormObservation:
    """Project a library entry Form like observe_form, at the entry's fixed env.

    errors include input errors and Spec validation errors. Nothing is
    evaluated beyond preview/result and the entry is not changed.
    """
    return _observation(
        entry.state.input.data,
        entry.state.preview(entry.env),
        entry.result(),
    )


def entry_invalid_to_remote(exc: LibraryEntryInvalid) -> RemoteError:
    """Report a rejected entry Form as invalid_params with its field errors.

    data.errors lists every {path, message} with the same rows as an
    observation's errors; reason is form_errors.
    """
    return RemoteError(
        ErrorCode.INVALID_PARAMS,
        str(exc),
        reason="form_errors",
        data={"errors": _error_rows(exc.errors.messages)},
    )


def _observation(
    data: dict[str, JsonValue],
    preview: FormPreview,
    result: ConfigBase | FormErrors,
) -> FormObservation:
    errors = dict(preview.errors.messages) if preview.errors is not None else {}
    if isinstance(result, FormErrors):
        errors.update(result.messages)
    return {
        "input": data,
        "values": [
            {"path": list(path), "value": context_wire_value(value)}
            for path, value in preview.values.items()
        ],
        "errors": _error_rows(errors),
    }


def _error_rows(messages: Mapping[FormPath, str]) -> list[FormError]:
    return [
        {"path": list(path), "message": message} for path, message in messages.items()
    ]


def form_writes(raw: object) -> Mapping[FormWritePath, JsonValue]:
    """Decode unique path/value edits; Form owns leaf and value validation.

    Reject malformed edit objects, paths, duplicate paths and non-JSON input
    with invalid_params before invoking the resource. Values retain Form tags.
    """
    if not isinstance(raw, list):
        raise RemoteError(ErrorCode.INVALID_PARAMS, "edits must be an array")
    batch: dict[FormWritePath, JsonValue] = {}
    for edit in raw:
        if not isinstance(edit, dict) or set(edit) != {"path", "value"}:
            raise RemoteError(ErrorCode.INVALID_PARAMS, "edits require path and value")
        raw_path = edit["path"]
        if not isinstance(raw_path, list) or not raw_path:
            raise RemoteError(ErrorCode.INVALID_PARAMS, "path must be a nonempty array")
        segments: list[str | int] = []
        for segment in raw_path:
            if not isinstance(segment, (str, int)) or (
                isinstance(segment, str) and not segment
            ):
                raise RemoteError(
                    ErrorCode.INVALID_PARAMS,
                    "path segments must be field names or integer indexes",
                )
            segments.append(segment)
        path = tuple(segments)
        if path in batch:
            raise RemoteError(ErrorCode.INVALID_PARAMS, "duplicate edit path")
        try:
            value = TypeAdapter(JsonValue).validate_python(edit["value"])
        except ValueError as error:
            raise RemoteError(ErrorCode.INVALID_PARAMS, str(error)) from error
        batch[path] = value
    return batch
