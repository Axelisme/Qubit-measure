"""Public Qt controls edit drafts while preserving local parse failures."""

import gc
from collections.abc import Iterator, Mapping
from typing import Annotated, Literal, Self

import pytest
from pydantic import Field, JsonValue, model_validator
from PyQt6.QtGui import QAction
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

from tests.gui._form_context import captured_ext_env


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
            "required": previous.get(
                "required",
                (
                    env.parameters.value("general.ext.seed")
                    if env.parameters is not None
                    else 4.0
                ),
            ),
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
            ("mode",): ("alpha", "beta")
            if (
                env.parameters is not None
                and env.parameters.lookup("general.ext.allow_beta").value
            )
            else ("alpha",)
        }

    @model_validator(mode="after")
    def validate_total(self) -> Self:
        """Leave a model-wide constraint attributed to the root."""
        if self.amount + self.count > 100:
            raise ValueError("Total exceeds the limit")
        return self


def environment(
    seed: float = 4.0,
    *,
    allow_beta: bool = False,
    empty_choices: bool = False,
    empty_tags: bool = False,
) -> FormEnv:
    """Build an isolated memory-only host context with numeric seed and choices."""
    return captured_ext_env(
        seed=seed,
        allow_beta=allow_beta,
        empty_choices=empty_choices,
        empty_tags=empty_tags,
    )


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


def combo[F: FormModel](editor: FormEditor[F], name: str) -> QComboBox:
    """Find a documented combo control by its complete object name."""
    widget = editor.findChild(QComboBox, name)
    assert isinstance(widget, QComboBox)
    return widget


def mode[F: FormModel](editor: FormEditor[F], path: str) -> QAction:
    """Find the documented Expression action of one row's input."""
    action = editor.findChild(QAction, f"form.mode.{path}")
    assert isinstance(action, QAction)
    return action


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

    mode(editor, "amount").trigger()
    line(editor, "amount").setText("general.ext.seed / 2")
    assert state.input.data["amount"] == {"expr": "general.ext.seed / 2"}
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

    mode(editor, "amount").trigger()
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
    mode(editor, "count").trigger()
    line(editor, "count").setText("general.ext.seed + 1")
    line(editor, "amplitude").setText("1+2j")
    assert state.input.data["count"] == {"expr": "general.ext.seed + 1"}
    assert state.input.data["amplitude"] == {"__complex__": [1.0, 2.0]}
    result = state.result(env)
    assert isinstance(result, SampleForm)
    assert result.count == 5
    assert result.amplitude == 1 + 2j
    assert editor.errors() is None


def test_branch_dropdown_switches_to_fresh_core_defaults(
    views: list[QWidget], qapp: QApplication
) -> None:
    env = environment()
    state = FormState(SampleForm.default_input(env))
    editor = FormEditor(state, env)
    views.append(editor)
    line(editor, "branch.shared").setText("6")
    selector = combo(editor, "form.input.branch")
    selector.setCurrentIndex(selector.findText("alternate"))
    qapp.processEvents()
    result = state.result(env)
    assert isinstance(result, SampleForm)
    assert isinstance(result.branch, AlternateBranch)
    # A new branch starts from its own defaults, not from the old branch.
    assert result.branch.shared == 9.0
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
    line(editor, "offsets").setText(
        '{"rack.left": {"expr": "general.ext.seed"}, "remove": 1}'
    )
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
            {
                **SampleForm.default_input(initial).data,
                "amount": {"expr": "general.ext.seed"},
            },
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
    assert [selector.itemText(i) for i in range(selector.count())] == [
        "alpha",
        "beta (missing)",
    ]
    assert selector.currentText() == "beta (missing)"


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
    state.write({("amount",): 6.0}, env)
    assert float(line(editor, "amount").text()) == 6.0
    assert notices == [1]
    state.reset(base)
    assert float(line(editor, "amount").text()) == 2.0
    assert notices == [1, 1]
    editor.set_editing_enabled(enabled=True)
    assert notices == [1, 1]


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


class CycleHost(QWidget):
    """A parentless host that refers to itself, so only cyclic GC frees it."""

    def __init__(self) -> None:
        """Create the host and its self-reference."""
        super().__init__()
        self.cycle: CycleHost | None = self


@pytest.mark.parametrize("detach", [True, False], ids=["detached", "attached"])
def test_cyclic_gc_frees_an_editor_host_and_leaves_its_state_writable(
    qapp: QApplication, detach: bool
) -> None:
    env = environment()
    states: list[FormState[SampleForm]] = []
    for _ in range(20):
        state = FormState(SampleForm.default_input(env))
        states.append(state)
        editor = FormEditor(state, env, CycleHost())
        if detach:
            editor.detach()
        del editor
        gc.collect()
    qapp.processEvents()
    for state in states:
        state.write({("amount",): 7.0}, env)
        assert state.input.data["amount"] == 7.0


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


class NumericChoiceForm(FormModel):
    """A numeric choice remains editable through an Expr alternative."""

    amount: float = 2.0

    @classmethod
    def choices(cls, _env: FormEnv, /) -> Mapping[FormPath, tuple[JsonValue, ...]]:
        """Offer direct values; a choice leaf takes no expression."""
        return {("amount",): (2.0, 4.0)}


def test_numeric_choices_show_a_dropdown_without_expression_mode(
    views: list[QWidget],
) -> None:
    env = environment()
    state = FormState(NumericChoiceForm.default_input(env))
    editor = FormEditor(state, env)
    views.append(editor)
    direct = editor.findChild(QComboBox, "form.input.amount")
    assert isinstance(direct, QComboBox)
    direct.setCurrentIndex(direct.findData(4.0))
    assert state.input.data["amount"] == 4.0
    assert editor.findChild(QAction, "form.mode.amount") is None
    assert editor.errors() is None


def test_invalid_complex_envelope_is_displayed_without_breaking_construction(
    views: list[QWidget],
) -> None:
    env = environment()
    state = FormState(SampleForm.default_input(env))
    state.write({("amplitude",): {"__complex__": [1.0]}}, env)
    editor = FormEditor(state, env)
    views.append(editor)
    assert isinstance(editor.errors(), FormErrors)
    assert label(editor, "form.error.amplitude").text()
    assert label(editor, "form.value.amplitude").text() == ""
    line(editor, "amplitude").setText("2+3j")
    assert state.input.data["amplitude"] == {"__complex__": [2.0, 3.0]}
    assert editor.errors() is None


def test_inactive_branch_discards_only_its_local_unparsed_text(
    views: list[QWidget],
) -> None:
    env = environment()
    state = FormState(SampleForm.default_input(env))
    editor = FormEditor(state, env)
    views.append(editor)
    line(editor, "branch.old_only").setText("-")
    line(editor, "amount").setText("-")
    selector = combo(editor, "form.input.branch")
    selector.setCurrentIndex(selector.findText("alternate"))
    local = editor.local_parse_errors()
    assert isinstance(local, FormErrors)
    assert set(local.messages) == {("amount",)}
    assert line(editor, "amount").text() == "-"


class NullableNumericForm(FormModel):
    """Nullable scalar inputs retain numeric syntax and Expr alternatives."""

    optional_float: float | None = None
    optional_int: int | None = None


@pytest.mark.parametrize(
    ("path", "direct"), [("optional_float", 3.5), ("optional_int", 3)]
)
def test_nullable_numeric_direct_partial_null_and_expression(
    views: list[QWidget], path: str, direct: float | int
) -> None:
    env = environment()
    state = FormState(
        FormInput(NullableNumericForm, {path: {"expr": "general.ext.seed"}})
    )
    editor = FormEditor(state, env)
    views.append(editor)
    text = editor.findChild(QLineEdit, f"form.input.{path}")
    assert isinstance(text, QLineEdit)
    action = mode(editor, path)
    assert action.isChecked()
    assert text.text() == "general.ext.seed"
    action.trigger()
    text.setText(str(direct))
    assert state.input.data[path] == direct
    assert type(state.input.data[path]) is type(direct)
    before = state.input.data
    text.setText("-")
    assert state.input.data == before
    assert isinstance(editor.local_parse_errors(), FormErrors)
    text.setText("")
    assert state.input.data[path] is None
    assert editor.local_parse_errors() is None
    action.trigger()
    text.setText("general.ext.seed + 1")
    assert state.input.data[path] == {"expr": "general.ext.seed + 1"}
    notices: list[int] = []
    state.subscribe(lambda: notices.append(1))
    before = state.input.data
    refreshed = environment(8.0)
    editor.refresh(refreshed)
    assert state.input.data == before
    assert notices == []
    assert text.text() == "general.ext.seed + 1"
    result = state.result(refreshed)
    assert isinstance(result, NullableNumericForm)
    assert getattr(result, path) == 9
    value = editor.findChild(QLabel, f"form.value.{path}")
    assert isinstance(value, QLabel)
    assert float(value.text()) == 9


class EmptyChoiceForm(FormModel):
    """Static Literal, free text and numeric choice sets may all become empty."""

    literal: Literal["alpha", "beta"] = "alpha"
    text: str = "alpha"
    number: float = 2.0

    @classmethod
    def choices(cls, env: FormEnv) -> Mapping[FormPath, tuple[JsonValue, ...]]:
        """Use only host-captured data to publish legal option sets."""
        empty = (
            env.parameters is not None
            and env.parameters.lookup("general.ext.empty_choices").value
        )
        return {
            ("literal",): () if empty else ("alpha", "beta"),
            ("text",): () if empty else ("alpha", "beta"),
            ("number",): () if empty else (2.0, 4.0),
        }


def test_empty_choices_refresh_preserves_input(
    views: list[QWidget],
) -> None:
    env = environment()
    state = FormState(EmptyChoiceForm.default_input(env))
    editor = FormEditor(state, env)
    views.append(editor)
    notices: list[int] = []
    state.subscribe(lambda: notices.append(1))
    before = state.input.data
    empty = environment(empty_choices=True)
    for captured, listed in ((env, True), (empty, False), (env, True), (empty, False)):
        editor.refresh(captured)
        for path in ("literal", "text", "number"):
            control = editor.findChild(QComboBox, f"form.input.{path}")
            assert isinstance(control, QComboBox)
            current = before[path]
            shown = [control.itemData(i) for i in range(control.count())]
            if listed:
                # Only the current choices, with the input's value selected.
                assert len(shown) == 2 and current in shown
                assert control.currentData() == current
            else:
                # No choice matches: the input shows as missing, unchanged.
                assert shown == [current]
                assert control.currentText() == f"{current} (missing)"
        assert state.input.data == before
        assert notices == []
        errors = editor.errors()
        if listed:
            assert errors is None
        else:
            assert isinstance(errors, FormErrors)
