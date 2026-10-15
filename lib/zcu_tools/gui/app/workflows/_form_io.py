"""Call-local Start/form inspection mechanics, without lifecycle or source capture.

Inputs use the caller's fixed FormEnv; errors remain directed FormErrors. Start
prepares defaults or parsed panel input without publishing, while inspection
projects ordinary leaves without committing to draft/base or Engine.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from math import isfinite

from pydantic import JsonValue
from pydantic_core import PydanticCustomError

from zcu_tools.core.complex_codec import encode_complex
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

from .remote import FormDescriptionInspectionReply, FormInspectionReply
from .remote.form_description import project_form_description


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


@dataclass(frozen=True)
class FormInspection:
    """Call-local observation source selected by the Window on its owner thread.

    input is the selected parsed draft, env its fixed evaluation capture.
    local_errors is the selected editor's unparsed-text report, used only for
    reads. revision is the attached Engine's native revision or None before
    Start; it is not a guard. This source never captures or publishes state.
    read/describe construct one private candidate per call from this input.
    """

    input: FormInput[FormModel]
    env: FormEnv
    local_errors: FormErrors | None
    revision: int | None

    def _candidate(
        self,
        batch: Mapping[FormWritePath, JsonValue] | None,
    ) -> FormState[FormModel] | FormErrors:
        state = FormState(self.input)
        if batch is not None:
            try:
                state.write(batch, self.env)
            except PydanticCustomError as exc:
                if exc.type != "form_field" or exc.context is None:
                    raise
                return FormErrors({exc.context["path"]: exc.context["message"]})
            except ValueError as exc:
                return FormErrors({(): str(exc)})
        return state

    def _observe(
        self, state: FormState[FormModel], *, include_local: bool
    ) -> FormInspectionReply:
        preview = state.preview(self.env)
        errors = dict(preview.errors.messages) if preview.errors is not None else {}
        if include_local and self.local_errors is not None:
            errors.update(self.local_errors.messages)
        return {
            "input": state.input.data,
            "values": [
                {"path": list(path), "value": _encode_form_value(value)}
                for path, value in preview.values.items()
            ],
            "errors": [
                {"path": list(path), "message": message}
                for path, message in errors.items()
            ],
            "revision": self.revision,
        }

    def read(
        self,
        batch: Mapping[FormWritePath, JsonValue] | None,
    ) -> FormInspectionReply | FormErrors:
        """Read draft or preview parsed edits without publication or I/O.

        batch=None reads this source with local parse errors. An explicit batch
        (including {}) previews edits to current parsed input, ignoring local
        text. Return lean input/ordinary values/errors/revision or directed
        FormErrors for rejected writes. Model-invalid input stays readable.
        Source/declaration failures propagate; nonfinite values raise ValueError.
        """
        state = self._candidate(batch)
        if isinstance(state, FormErrors):
            return state
        return self._observe(state, include_local=batch is None)

    def describe(
        self,
        path: FormWritePath,
        batch: Mapping[FormWritePath, JsonValue] | None,
    ) -> FormDescriptionInspectionReply | FormErrors:
        """Describe path and observe the exact same call-local candidate.

        path is a tuple of model-field names; () is root. batch follows read's
        contract, including local errors only when None. Return lean observation
        plus single-layer core metadata, never another draft or live-branch
        lookup. Rejected writes return FormErrors. Unknown/inactive/whole-leaf
        descendant paths raise FormPathError from core; declaration/source and
        unexpected failures propagate. No defaults, captures or I/O are added.
        """
        state = self._candidate(batch)
        if isinstance(state, FormErrors):
            return state
        description = state.describe(path, self.env)
        return {
            **self._observe(state, include_local=batch is None),
            "description": project_form_description(description),
        }
