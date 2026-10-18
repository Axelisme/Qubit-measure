"""Declaration admission observes copies and cannot rewrite the installed input."""

from typing import ClassVar

from pydantic import JsonValue
from zcu_tools.gui import form as api


class AdmissionForm(api.FormModel):
    entries: list[int]
    calls: ClassVar[list[tuple[dict[str, JsonValue], dict[str, JsonValue]]]] = []

    @classmethod
    def check_input_change(
        cls, before: dict[str, JsonValue], after: dict[str, JsonValue]
    ) -> None:
        cls.calls.append((before, after))
        old = before["entries"]
        new = after["entries"]
        assert isinstance(old, list) and isinstance(new, list)
        old.append(90)
        new.append(99)


def test_admission_receives_detached_copies_without_mutating_draft_or_base():
    AdmissionForm.calls.clear()
    try:
        state = api.FormState(api.FormInput(AdmissionForm, {"entries": [1]}))
        state.write({("entries",): [2]}, api.FormEnv())
        assert len(AdmissionForm.calls) == 1
        assert state.input.data == {"entries": [2]}
        assert state.base.data == {"entries": [1]}
    finally:
        AdmissionForm.calls.clear()


def test_restore_and_reset_do_not_run_input_change_admission():
    AdmissionForm.calls.clear()
    try:
        assert hasattr(api.FormModel, "check_input_change"), (
            "Missing neutral admission seam"
        )
        state = api.FormState(api.FormInput(AdmissionForm, {"entries": [1]}))
        state.write_compatible({"entries": [2, 3]}, api.FormEnv())
        state.reset(api.FormInput(AdmissionForm, {"entries": [4]}))
        assert AdmissionForm.calls == []
        assert state.input.data == {"entries": [4]}
    finally:
        AdmissionForm.calls.clear()
