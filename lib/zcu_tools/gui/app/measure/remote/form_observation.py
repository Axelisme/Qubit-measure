"""Measure Form wire projection, using the tab's fixed owner capture."""

from collections.abc import Mapping
from typing import TypedDict

from pydantic import JsonValue, TypeAdapter

from zcu_tools.gui.app.measure.services.tab_form import TabFormResource
from zcu_tools.gui.form import FormErrors, FormWritePath
from zcu_tools.gui.remote.errors import ErrorCode, RemoteError

from .handlers._wire_values import context_wire_value


class FormWrite(TypedDict):
    """One ordinary leaf: path contains model field names; value is encoded JSON."""

    path: list[str]
    value: object


class FormError(TypedDict):
    """One validation error; path may contain model fields and sequence indexes."""

    path: list[str | int]
    message: str


class TabFormObservation(TypedDict):
    """Detached Form view, not an acceptance or a source refresh.

    schema describes editable input in the fixed capture. input retains missing
    fields and expression tags. values contains evaluated ordinary leaves, not
    proof of validity. errors includes input and lowered cfg validation failures.
    """

    schema: dict[str, JsonValue]
    input: dict[str, JsonValue]
    values: list[FormWrite]
    errors: list[FormError]


def observe_form(resource: TabFormResource) -> TabFormObservation:
    """Project current input/schema/preview/errors without capture or mutation."""
    preview = resource.preview()
    errors = dict(preview.errors.messages) if preview.errors is not None else {}
    result = resource.result()
    if isinstance(result, FormErrors):
        errors.update(result.messages)
    return {
        "schema": resource.input_schema(),
        "input": resource.input.data,
        "values": [
            {"path": list(path), "value": context_wire_value(value)}
            for path, value in preview.values.items()
        ],
        "errors": [
            {"path": list(path), "message": message} for path, message in errors.items()
        ],
    }


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
        segments: list[str] = []
        for segment in raw_path:
            if not isinstance(segment, str) or not segment:
                raise RemoteError(
                    ErrorCode.INVALID_PARAMS, "path segments must be nonempty strings"
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
