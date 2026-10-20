"""Form refusal projection keeps the existing taxonomy and readonly details."""

from typing import Annotated

import pytest
from pydantic import JsonValue
from zcu_tools.gui.app.measure.remote.form_errors import input_rejected_to_remote
from zcu_tools.gui.form import (
    Display,
    FormEnv,
    FormInput,
    FormInputRejected,
    FormModel,
    FormState,
)
from zcu_tools.gui.remote.errors import ErrorCode, ErrorEnvelope


class ReadonlyForm(FormModel):
    value: Annotated[list[int] | None, Display(readonly=True)]


@pytest.mark.parametrize("current", [[1, 2], None])
def test_readonly_refusal_wire_preserves_path_reason_and_json_current(
    current: JsonValue,
) -> None:
    state = FormState(FormInput(ReadonlyForm, {"value": current}))
    with pytest.raises(FormInputRejected) as caught:
        state.write({("value",): [3]}, FormEnv())

    wire = ErrorEnvelope.from_remote_error(
        input_rejected_to_remote(caught.value)
    ).to_wire()

    assert wire["code"] == ErrorCode.INVALID_PARAMS.value
    assert wire["reason"] == "form_input_rejected"
    assert wire["data"] == {
        "form_path": ["value"],
        "reason": "readonly",
        "current": current,
    }


def test_other_refusals_keep_the_original_wire_data_without_readonly_details() -> None:
    wire = ErrorEnvelope.from_remote_error(
        input_rejected_to_remote(FormInputRejected(("items", 1, "name"), "identity"))
    ).to_wire()

    assert wire["reason"] == "form_input_rejected"
    assert wire["data"] == {"form_path": ["items", 1, "name"]}
