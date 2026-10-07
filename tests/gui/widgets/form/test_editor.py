"""Public Qt controls edit drafts while preserving local parse failures."""

from collections.abc import Iterator, Mapping
from typing import Annotated, Literal, Self

import pytest
from pydantic import Field, JsonValue, model_validator
from qtpy.QtCore import QEvent
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QLabel,
    QLineEdit,
    QWidget,
)
from zcu_tools.gui.form import (
    FormEnv,
    FormErrors,
    FormInput,
    FormModel,
    FormPath,
    FormState,
    field_error,
)
from zcu_tools.gui.widgets.form import FormEditor
from zcu_tools.resources.context import MetaDict


class StaticBranch(FormModel):
    """Initial branch with one shared numeric field."""

    kind: Literal["static"] = "static"
    shared: float = 2.0
    old_only: float = 1.0


class AlternateBranch(FormModel):
    """New branch with a required field and a context-provided default."""

    kind: Literal["alternate"] = "alternate"
    shared: float = 9.0
    required: float

    @classmethod
    def defaults(
        cls, env: FormEnv, previous: Mapping[str, JsonValue]
    ) -> dict[str, JsonValue]:
        """Supply the required field from this host-captured context."""
        return {
            "kind": "alternate",
            "shared": 9.0,
            "required": previous.get("required", env.md.get("seed", 4.0)),
        }


class Section(FormModel):
    """Nullable nested section with a directed validation error."""

    start: float = 1.0
    stop: float = 2.0

    @model_validator(mode="after")
    def validate_order(self) -> Self:
        """Direct invalid ordering to this section's stop field."""
        if self.stop <= self.start:
            raise field_error(("stop",), "Stop must exceed start")
        return self


class SampleForm(FormModel):
    """Scalar, whole-container, branch and nullable input scenarios."""

    amount: float = Field(default=2.0, gt=0)
    count: int = 2
    enabled: bool = False
    amplitude: complex = 1 + 0j
    samples: tuple[float, ...] = (1.0,)
    offsets: dict[str, float] = Field(default_factory=dict)
    branch: Annotated[StaticBranch | AlternateBranch, Field(discriminator="kind")] = (
        Field(default_factory=StaticBranch)
    )
    section: Section | None = None
    mode: Literal["alpha", "beta"] = "alpha"

    @classmethod
    def choices(cls, env: FormEnv) -> Mapping[FormPath, tuple[JsonValue, ...]]:
        """Restrict mode according to the context provided by the host."""
        return {
            ("mode",): ("alpha", "beta") if env.md.get("allow_beta") else ("alpha",)
        }

    @model_validator(mode="after")
    def validate_total(self) -> Self:
        """Leave a model-wide constraint attributed to the root."""
        if self.amount + self.count > 100:
            raise ValueError("Total exceeds the limit")
        return self


def environment(seed: float = 4.0, *, allow_beta: bool = False) -> FormEnv:
    """Build an isolated memory-only host context with numeric seed and choices."""
    md = MetaDict()
    md.update({"seed": seed, "allow_beta": allow_beta})
    return FormEnv(md)


@pytest.fixture
def views(qapp: QApplication) -> Iterator[list[QWidget]]:
    """Delete this test's views on the existing QApplication after each test."""
    owned: list[QWidget] = []
    yield owned
    for widget in owned:
        widget.deleteLater()
    qapp.processEvents()


def line(editor: FormEditor[SampleForm], path: str) -> QLineEdit:
    """Find the documented text input control for a dotted model-field path."""
    widget = editor.findChild(QLineEdit, f"form.input.{path}")
    assert isinstance(widget, QLineEdit)
    return widget


def combo(editor: FormEditor[SampleForm], name: str) -> QComboBox:
    """Find a documented combo control by its complete object name."""
    widget = editor.findChild(QComboBox, name)
    assert isinstance(widget, QComboBox)
    return widget


def label(editor: FormEditor[SampleForm], name: str) -> QLabel:
    """Find a documented result/error label by its complete object name."""
    widget = editor.findChild(QLabel, name)
    assert isinstance(widget, QLabel)
    return widget


def test_direct_numeric_and_expression_controls_write_only_parsed_input(
    views: list[QWidget],
) -> None:
    env = environment()
    state = FormState(SampleForm.default_input(env))
    editor = FormEditor(state, env)
    views.append(editor)
    line(editor, "amount").setText("3.5")
    result = state.result(env)
    assert isinstance(result, SampleForm)
    assert result.amount == 3.5

    mode = combo(editor, "form.mode.amount")
    mode.setCurrentIndex(mode.findData("expr"))
    line(editor, "amount").setText("seed / 2")
    assert state.input.data["amount"] == {"expr": "seed / 2"}
    result = state.result(env)
    assert isinstance(result, SampleForm)
    assert result.amount == 2.0
    assert float(label(editor, "form.value.amount").text()) == 2.0
    assert editor.local_parse_errors() is None
    assert editor.errors() is None


def test_boolean_control_writes_an_ordinary_bool(views: list[QWidget]) -> None:
    env = environment()
    state = FormState(SampleForm.default_input(env))
    editor = FormEditor(state, env)
    views.append(editor)
    control = editor.findChild(QCheckBox, "form.input.enabled")
    assert isinstance(control, QCheckBox)
    control.setChecked(True)
    result = state.result(env)
    assert isinstance(result, SampleForm)
    assert result.enabled is True
    assert state.input.data["enabled"] is True
    assert editor.errors() is None


def test_partial_text_stays_local_and_validity_recovers_after_correction(
    views: list[QWidget],
) -> None:
    env = environment()
    state = FormState(SampleForm.default_input(env))
    editor = FormEditor(state, env)
    views.append(editor)
    observed: list[bool] = []
    editor.validity_changed.connect(observed.append)
    before = state.input.data
    line(editor, "amount").setText("-")
    assert state.input.data == before
    local = editor.local_parse_errors()
    assert isinstance(local, FormErrors)
    assert ("amount",) in local.messages
    assert isinstance(editor.errors(), FormErrors)
    assert label(editor, "form.error.amount").text()
    assert observed == [False]

    editor.refresh(environment(8.0))
    assert line(editor, "amount").text() == "-"
    assert state.input.data == before
    line(editor, "amount").setText("5")
    assert state.input.data["amount"] == 5.0
    assert editor.local_parse_errors() is None
    assert editor.errors() is None
    assert observed == [False, True]


def test_model_invalid_value_and_expression_remain_in_state_with_errors(
    views: list[QWidget],
) -> None:
    env = environment()
    state = FormState(SampleForm.default_input(env))
    editor = FormEditor(state, env)
    views.append(editor)
    line(editor, "amount").setText("-2")
    assert state.input.data["amount"] == -2.0
    assert editor.local_parse_errors() is None
    errors = editor.errors()
    assert isinstance(errors, FormErrors)
    assert ("amount",) in errors.messages
    assert label(editor, "form.error.amount").text()

    mode = combo(editor, "form.mode.amount")
    mode.setCurrentIndex(mode.findData("expr"))
    line(editor, "amount").setText("absent_name")
    assert state.input.data["amount"] == {"expr": "absent_name"}
    assert editor.local_parse_errors() is None
    errors = editor.errors()
    assert isinstance(errors, FormErrors)
    assert ("amount",) in errors.messages


def test_integer_expression_and_direct_complex_keep_their_input_forms(
    views: list[QWidget],
) -> None:
    env = environment()
    state = FormState(SampleForm.default_input(env))
    editor = FormEditor(state, env)
    views.append(editor)
    mode = combo(editor, "form.mode.count")
    mode.setCurrentIndex(mode.findData("expr"))
    line(editor, "count").setText("seed + 1")
    line(editor, "amplitude").setText("1+2j")
    assert state.input.data["count"] == {"expr": "seed + 1"}
    assert state.input.data["amplitude"] == {"__complex__": [1.0, 2.0]}
    result = state.result(env)
    assert isinstance(result, SampleForm)
    assert result.count == 5
    assert result.amplitude == 1 + 2j
    assert editor.errors() is None


def test_branch_selector_preserves_shared_value_and_uses_core_defaults(
    views: list[QWidget], qapp: QApplication
) -> None:
    env = environment()
    state = FormState(SampleForm.default_input(env))
    editor = FormEditor(state, env)
    views.append(editor)
    line(editor, "branch.shared").setText("6")
    selector = combo(editor, "form.input.branch.kind")
    selector.setCurrentIndex(selector.findData("alternate"))
    qapp.processEvents()
    result = state.result(env)
    assert isinstance(result, SampleForm)
    assert isinstance(result.branch, AlternateBranch)
    assert result.branch.shared == 6.0
    assert result.branch.required == 4.0
    assert editor.findChild(QLineEdit, "form.input.branch.old_only") is None
    assert float(line(editor, "branch.required").text()) == 4.0


def test_nullable_section_activation_child_edit_and_deactivation(
    views: list[QWidget], qapp: QApplication
) -> None:
    env = environment()
    state = FormState(SampleForm.default_input(env))
    editor = FormEditor(state, env)
    views.append(editor)
    enabled = editor.findChild(QCheckBox, "form.enabled.section")
    assert isinstance(enabled, QCheckBox)
    enabled.setChecked(True)
    qapp.processEvents()
    line(editor, "section.stop").setText("5")
    result = state.result(env)
    assert isinstance(result, SampleForm)
    assert result.section == Section(start=1.0, stop=5.0)

    enabled = editor.findChild(QCheckBox, "form.enabled.section")
    assert isinstance(enabled, QCheckBox)
    enabled.setChecked(False)
    qapp.processEvents()
    assert state.input.data["section"] is None
    assert editor.findChild(QLineEdit, "form.input.section.stop") is None
    assert editor.errors() is None


def test_nested_and_root_model_errors_are_visible_in_public_labels(
    views: list[QWidget],
) -> None:
    env = environment()
    state = FormState(SampleForm.default_input(env))
    state.write({("section",): {"start": 1.0, "stop": 0.5}}, env)
    editor = FormEditor(state, env)
    views.append(editor)
    errors = editor.errors()
    assert isinstance(errors, FormErrors)
    assert ("section", "stop") in errors.messages
    assert label(editor, "form.error.section.stop").text()
    line(editor, "section.stop").setText("3")
    state.write({("amount",): 90.0, ("count",): 20}, env)
    errors = editor.errors()
    assert isinstance(errors, FormErrors)
    assert () in errors.messages
    assert label(editor, "form.error").text()


def test_whole_sequence_and_mapping_text_support_replacement_and_clear(
    views: list[QWidget],
) -> None:
    env = environment()
    state = FormState(SampleForm.default_input(env))
    editor = FormEditor(state, env)
    views.append(editor)
    line(editor, "samples").setText("[2, 3]")
    line(editor, "offsets").setText('{"rack.left": {"expr": "seed"}, "remove": 1}')
    result = state.result(env)
    assert isinstance(result, SampleForm)
    assert result.samples == (2.0, 3.0)
    assert result.offsets == {"rack.left": 4.0, "remove": 1.0}
    line(editor, "offsets").setText('{"renamed": 4}')
    result = state.result(env)
    assert isinstance(result, SampleForm)
    assert result.offsets == {"renamed": 4.0}
    line(editor, "offsets").setText("{}")
    result = state.result(env)
    assert isinstance(result, SampleForm)
    assert result.offsets == {}
    changes = state.changes(env)
    assert not isinstance(changes, FormErrors)
    assert {change.path for change in changes} == {("samples",)}
    assert editor.errors() is None


def test_incomplete_container_json_stays_local_until_it_parses(
    views: list[QWidget],
) -> None:
    env = environment()
    state = FormState(SampleForm.default_input(env))
    editor = FormEditor(state, env)
    views.append(editor)
    line(editor, "offsets").setText('{"unfinished":')
    assert state.input.data["offsets"] == {}
    errors = editor.local_parse_errors()
    assert isinstance(errors, FormErrors)
    assert ("offsets",) in errors.messages
    line(editor, "offsets").setText('{"finished": 2}')
    assert state.input.data["offsets"] == {"finished": 2}
    assert editor.local_parse_errors() is None


def test_refresh_changes_choices_and_evaluation_without_writing_state(
    views: list[QWidget],
) -> None:
    initial = environment(4.0, allow_beta=True)
    state = FormState(
        FormInput(
            SampleForm,
            {**SampleForm.default_input(initial).data, "amount": {"expr": "seed"}},
        )
    )
    editor = FormEditor(state, initial)
    views.append(editor)
    notices: list[int] = []
    state.subscribe(lambda: notices.append(1))
    selector = combo(editor, "form.input.mode")
    selector.setCurrentIndex(selector.findData("beta"))
    assert state.input.data["mode"] == "beta"
    before = state.input.data
    notices.clear()
    editor.refresh(environment(8.0))
    assert state.input.data == before
    assert notices == []
    assert float(label(editor, "form.value.amount").text()) == 8.0
    errors = editor.errors()
    assert isinstance(errors, FormErrors)
    assert ("mode",) in errors.messages
    selector = combo(editor, "form.input.mode")
    assert selector.findData("beta") == -1


def test_external_publication_reset_and_readonly_do_not_echo_writes(
    views: list[QWidget],
) -> None:
    env = environment()
    base = SampleForm.default_input(env)
    state = FormState(base)
    editor = FormEditor(state, env)
    views.append(editor)
    notices: list[int] = []
    state.subscribe(lambda: notices.append(1))
    editor.set_editing_enabled(enabled=False)
    assert not line(editor, "amount").isEnabled()
    state.write({("amount",): 6.0}, env)
    assert float(line(editor, "amount").text()) == 6.0
    assert notices == [1]
    assert not line(editor, "amount").isEnabled()
    state.reset(base)
    assert float(line(editor, "amount").text()) == 2.0
    assert notices == [1, 1]
    editor.set_editing_enabled(enabled=True)
    assert line(editor, "amount").isEnabled()


def test_destroying_the_view_leaves_the_caller_state_usable(qapp: QApplication) -> None:
    env = environment()
    state = FormState(SampleForm.default_input(env))
    editor = FormEditor(state, env)
    editor.deleteLater()
    qapp.sendPostedEvents(None, QEvent.Type.DeferredDelete.value)
    state.write({("amount",): 7.0}, env)
    result = state.result(env)
    assert isinstance(result, SampleForm)
    assert result.amount == 7.0


@pytest.mark.parametrize("action", ["detach", "close"])
def test_detach_and_close_stop_publications_without_resetting_the_state(
    views: list[QWidget], action: str
) -> None:
    env = environment()
    state = FormState(SampleForm.default_input(env))
    editor = FormEditor(state, env)
    views.append(editor)
    control = line(editor, "amount")
    before = control.text()
    if action == "close":
        editor.close()
    else:
        editor.detach()
        editor.detach()
    state.write({("amount",): 7.0}, env)
    assert state.input.data["amount"] == 7.0
    assert control.text() == before
    with pytest.raises(RuntimeError, match="detach|closed|inactive"):
        editor.refresh(env)
    with pytest.raises(RuntimeError, match="detach|closed|inactive"):
        editor.local_parse_errors()
