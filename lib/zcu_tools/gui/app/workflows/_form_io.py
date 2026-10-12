"""Call-local Start/form inspection mechanics, without lifecycle or source capture.

Inputs use the caller's fixed FormEnv; errors remain directed FormErrors. Start
prepares defaults or parsed panel input without publishing, while inspection
projects ordinary leaves without committing to draft/base or Engine.
"""

from __future__ import annotations

from collections.abc import Mapping
from math import isfinite

from pydantic import JsonValue
from pydantic_core import PydanticCustomError

from zcu_tools.gui.cfg.codec import encode_complex
from zcu_tools.gui.form import (
    FormEnv,
    FormErrors,
    FormInput,
    FormModel,
    FormState,
    FormValue,
    FormWritePath,
)
from zcu_tools.gui.widgets.form import FormEditor

from .remote import FormInspectionReply


def prepare_start_form[F: FormModel](
    model: type[F],
    state: FormState[F],
    editor: FormEditor[F],
    batch: Mapping[FormWritePath, JsonValue] | None,
    env: FormEnv,
) -> tuple[F, FormInput[F]] | FormErrors:
    """Select draft or captured defaults, returning validated values and input.

    model is the catalog's current form type; state/editor are its shared parsed
    GUI draft and local text view. batch=None reads them. An explicit batch uses
    fresh model defaults in env and never reads the editor/draft. env is the
    caller's fixed evaluation capture. Return ordinary model and original input,
    or directed FormErrors without editing the supplied state/editor. Native
    directed default/write errors retain their field path; undirected structural
    errors use (). Unexpected/default source failures propagate; no I/O occurs.
    """
    if batch is None:
        errors = editor.local_parse_errors()
        if errors is not None:
            return errors
    else:
        try:
            state = FormState(model.default_input(env))
            state.write(batch, env)
        except PydanticCustomError as exc:
            # field_error owns the directed path; other native errors propagate.
            if exc.type != "form_field" or exc.context is None:
                raise
            return FormErrors({exc.context["path"]: exc.context["message"]})
        except ValueError as exc:
            # Batch structure errors have no directed path in the core contract.
            return FormErrors({(): str(exc)})
    form = state.result(env)
    if isinstance(form, FormErrors):
        return form
    return form, state.input


def _encode_form_value(value: FormValue) -> JsonValue:
    """Project ordinary display leaves; scalar complex format belongs to codec."""
    if isinstance(value, complex):
        encoded = encode_complex(value)
        return {"__complex__": list[JsonValue](encoded["__complex__"])}
    if isinstance(value, (list, tuple)):
        return [_encode_form_value(item) for item in value]
    if isinstance(value, dict):
        return {key: _encode_form_value(item) for key, item in value.items()}
    if isinstance(value, float) and not isfinite(value):
        raise ValueError("Nonfinite form values cannot be encoded as JSON")
    return value


def inspect_form_input(
    input: FormInput[FormModel],
    env: FormEnv,
    batch: Mapping[FormWritePath, JsonValue] | None,
    local_errors: FormErrors | None,
    revision: int | None,
) -> FormInspectionReply | FormErrors:
    """Project input/candidate and partial ordinary values without publication.

    input is the parsed current FormInput, including expression original text.
    env/revision are fixed owner observations, not sources to refresh; revision
    may be None before Start. batch=None reads input; an explicit batch previews
    edits via the core's write contract in call-local state. local_errors is the
    editor's parse report for reads, or None for batch preview. Return the typed
    schema/input/ordinary leaves/errors projection or directed FormErrors for
    rejected writes, without changing input/base/editor/Engine. Declaration and
    unexpected failures propagate; nonfinite ordinary values raise ValueError.
    No source capture, I/O or hardware access occurs.
    """
    state = FormState(input)
    if batch is not None:
        try:
            state.write(batch, env)
        except PydanticCustomError as exc:
            if exc.type != "form_field" or exc.context is None:
                raise
            return FormErrors({exc.context["path"]: exc.context["message"]})
        except ValueError as exc:
            return FormErrors({(): str(exc)})
    preview = state.preview(env)
    errors = dict(preview.errors.messages) if preview.errors is not None else {}
    if local_errors is not None:
        errors.update(local_errors.messages)
    return {
        "schema": state.input_schema(env),
        "input": state.input.data,
        "values": [
            {"path": list(path), "value": _encode_form_value(value)}
            for path, value in preview.values.items()
        ],
        "errors": [
            {"path": list(path), "message": message} for path, message in errors.items()
        ],
        "revision": revision,
    }
