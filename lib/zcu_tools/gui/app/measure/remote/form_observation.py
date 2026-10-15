"""Measure Form wire projection, using each Form owner's fixed capture."""

from collections.abc import Mapping
from typing import TypedDict

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

from .handlers._wire_values import context_wire_value


class FormWrite(TypedDict):
    """One ordinary leaf: path contains model field names; value is encoded JSON."""

    path: list[str]
    value: JsonValue


class FormError(TypedDict):
    """One validation error; path may contain model fields and sequence indexes."""

    path: list[str | int]
    message: str


class FormObservation(TypedDict):
    """Detached Form view, not an acceptance or a source refresh.

    input retains missing
    fields and expression tags. values contains evaluated ordinary leaves, not
    proof of validity. errors includes input and lowered cfg validation failures.
    """

    input: dict[str, JsonValue]
    values: list[FormWrite]
    errors: list[FormError]


def observe_form(resource: TabFormResource) -> FormObservation:
    """Project current input/preview/errors without capture or mutation."""
    return _observation(
        resource.input.data,
        resource.preview(),
        resource.result(),
    )


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
