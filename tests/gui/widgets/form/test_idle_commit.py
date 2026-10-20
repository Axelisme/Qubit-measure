"""Idle text commits share one timer policy across forms and scalar helpers.

UI evidence uses ComponentDriver snapshots. Other tests observe the public host
write boundary and FormState. Only the external Qt timer is replaced.
"""

import os
import sys
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Annotated, Literal

import pytest
from pydantic import Field, JsonValue
from qtpy.QtCore import Qt
from qtpy.QtWidgets import QApplication, QLineEdit, QSpinBox, QVBoxLayout, QWidget
from zcu_tools.gui.form import Display, FormEnv, FormModel, FormState, FormWritePath
from zcu_tools.gui.widgets.form import FormEditor, value_widgets
from zcu_tools.gui.widgets.spinbox import TrimDoubleSpinBox

from tests.gui._user_input import ComponentDriver, key_click, type_text
from tests.gui.widgets.form._clock import Clock


class Child(FormModel):
    value: int = 2


class IdleForm(FormModel):
    count: int = 2
    amount: float = Field(default=2.0, gt=0)
    name: str = Field(default="aa", min_length=2)
    expression: float = 2.0
    amplitude: complex = 1 + 0j
    child: Annotated[Child, Display(collapsed=False)] = Field(default_factory=Child)
    optional: Child | None = Field(default_factory=Child)
    choice: Literal["first", "second"] = "first"


type Batch = dict[FormWritePath, JsonValue]


@pytest.fixture
def form(
    qapp: QApplication,
    clock: Clock,
) -> Iterator[tuple[FormEditor[IdleForm], FormState[IdleForm], list[Batch]]]:
    """Show a form with a recording public host write callback."""
    env = FormEnv()
    state = FormState(IdleForm.default_input(env))
    batches: list[Batch] = []

    def write(batch: Mapping[FormWritePath, JsonValue]) -> None:
        batches.append(dict(batch))
        state.write(batch, env)

    editor = FormEditor(
        state,
        env,
        write=write,
        select=lambda path, value: state.select(path, value, env),
        run_command=lambda path, name: state.run_command(path, name, env),
    )
    editor.resize(700, 500)
    editor.show()
    editor.activateWindow()
    qapp.processEvents()
    yield editor, state, batches
    editor.close()
    editor.deleteLater()
    qapp.processEvents()


@pytest.fixture
def evidence(tmp_path: Path, request: pytest.FixtureRequest):
    """Save each UI step to task scratch when requested, otherwise tmp_path."""
    directory = (
        Path(os.environ.get("FORM_IDLE_SCREENSHOTS", str(tmp_path))) / request.node.name
    )
    directory.mkdir(parents=True, exist_ok=True)

    def capture(driver: ComponentDriver, step: str) -> None:
        driver.screenshot(directory / f"{step}.png")

    return capture


@pytest.mark.parametrize(
    "path,value",
    [
        ("count", "12"),
        ("amount", "3.5"),
        ("name", "hello"),
        ("expression", "1 / 4"),
        ("amplitude", "2+3j"),
    ],
)
def test_idle_validation_and_repair_are_visible(
    form, clock: Clock, evidence, path: str, value: str
) -> None:
    driver = ComponentDriver(form[0])
    name = f"form.input.{path}"
    baseline_help = driver.observe(object_name=f"form.row.{path}").tooltip
    # An incomplete integer, malformed scalar or invalid float is retained.
    invalid = "-1.0" if path == "amount" else "1." if path == "count" else "x"
    if path == "expression":
        driver.invoke_context_action(action="Expression", object_name=name)
        invalid = "1 /"
    elif path == "amplitude":
        invalid = "2+"
    driver.enter_text(value=invalid, object_name=name, commit=False)
    evidence(driver, "01-typing")
    clock.advance(500)
    failed = driver.observe(object_name=name)
    evidence(driver, "02-idle-error")
    assert failed.text == invalid
    assert driver.observe(object_name=f"form.row.{path}").tooltip != baseline_help
    driver.enter_text(value=value, object_name=name, commit=False)
    evidence(driver, "03-repair-typing")
    clock.advance(500)
    repaired = driver.observe(object_name=name)
    evidence(driver, "04-idle-valid")
    assert driver.observe(object_name=f"form.row.{path}").tooltip == baseline_help
    if path == "expression":
        assert driver.observe(object_name="form.value.expression").text == "= 0.25"
    else:
        assert repaired.text == value or (
            path == "amplitude" and repaired.text == "(2+3j)"
        )


@pytest.mark.parametrize("finish", ["idle", "enter", "focus"])
def test_latest_text_commits_once_and_new_keys_restart_deadline(
    form, clock: Clock, finish: str
) -> None:
    editor, state, batches = form
    driver = ComponentDriver(editor)
    # This is a host-write seam contract, not the UI evidence scenario.
    count = editor.findChild(QLineEdit, "form.input.count")
    amount = editor.findChild(QLineEdit, "form.input.amount")
    assert count is not None and amount is not None
    count.setFocus()
    QApplication.processEvents()
    driver.enter_text(value="7", object_name="form.input.count", commit=False)
    clock.advance(499)
    assert batches == []
    driver.enter_text(value="73", object_name="form.input.count", commit=False)
    clock.advance(1)
    assert batches == []
    if finish == "enter":
        driver.enter_text(value="73", object_name="form.input.count")
    elif finish == "focus":
        amount.setFocus()
        QApplication.processEvents()
    else:
        clock.advance(499)
    assert batches == [{("count",): 73}]
    assert state.input.data["count"] == 73
    clock.advance(1000)
    driver.click(object_name="form.input.amount")
    assert batches == [{("count",): 73}]


@pytest.mark.parametrize("lifetime", ["fold", "readonly", "detach", "remove"])
def test_pending_text_never_writes_after_lifetime_change(
    form, clock: Clock, lifetime: str
) -> None:
    editor, state, batches = form
    driver = ComponentDriver(editor)
    driver.enter_text(value="3", object_name="form.input.child.value", commit=False)
    clock.advance(500)
    assert batches == [{("child", "value"): 3}]
    batches.clear()
    path = "optional.value" if lifetime == "remove" else "child.value"
    driver.enter_text(value="9", object_name=f"form.input.{path}", commit=False)
    if lifetime == "fold":
        driver.click(object_name="form.fold.child")
        driver.click(object_name="form.fold.child")
        assert driver.observe(object_name="form.input.child.value").text == "9"
    elif lifetime == "readonly":
        editor.set_editing_enabled(enabled=False)
        editor.set_editing_enabled(enabled=True)
    elif lifetime == "detach":
        editor.detach()
    else:
        # Replacing a branch is a public FormState publication, not a focus edit.
        state.write({("optional",): None}, FormEnv())
    clock.advance(1000)
    assert batches == []
    assert state.input.data["child"] == {"value": 3}
    if lifetime == "fold":
        child = editor.findChild(QLineEdit, "form.input.child.value")
        count = editor.findChild(QLineEdit, "form.input.count")
        assert child is not None and count is not None
        child.setFocus()
        QApplication.processEvents()
        count.setFocus()
        QApplication.processEvents()
        assert batches == [{("child", "value"): 9}]


@pytest.mark.parametrize(
    "type_,default,typed", [(str, "a", "hello"), (int, 2, "7"), (float, 2.0, "3.5")]
)
def test_shared_scalar_helpers_use_the_same_idle_policy(
    qapp: QApplication, clock: Clock, type_: type, default: object, typed: str
) -> None:
    window = QWidget()
    layout = QVBoxLayout(window)
    widget = value_widgets.make_value_widget(type_, default, None)
    widget.setObjectName("scalar")
    layout.addWidget(widget)
    commits: list[object] = []
    value_widgets.connect_committed_value_widget(
        widget, lambda: commits.append(value_widgets.read_value_widget(widget, type_))
    )
    window.show()
    window.activateWindow()
    qapp.processEvents()
    try:
        # Shared scalar callback contract, separate from driver UI acceptance.
        text = widget if isinstance(widget, QLineEdit) else widget.findChild(QLineEdit)
        assert text is not None
        text.setFocus()
        key_click(text, Qt.Key.Key_A, Qt.KeyboardModifier.ControlModifier)
        type_text(text, typed)
        assert commits == []
        clock.advance(500)
        assert commits == [type_(typed)]
        clock.advance(500)
        assert commits == [type_(typed)]
    finally:
        window.close()
        window.deleteLater()
        qapp.processEvents()


@pytest.mark.parametrize("factory", [QSpinBox, TrimDoubleSpinBox])
def test_scalar_steps_preserve_qt_callback_argument_adaptation(
    qapp: QApplication,
    clock: Clock,
    monkeypatch: pytest.MonkeyPatch,
    factory,
) -> None:
    """Public scalar steps support both no-argument and value-taking callbacks."""
    exceptions: list[BaseException] = []
    # Qt reports slot exceptions through this external boundary. Capture them
    # so a missing callback remains an assertion failure rather than aborting.
    monkeypatch.setattr(
        sys, "excepthook", lambda kind, error, trace: exceptions.append(error)
    )
    no_argument = factory()
    with_value = factory()
    notifications: list[str] = []
    values: list[int | float] = []
    value_widgets.connect_committed_value_widget(
        no_argument, lambda: notifications.append("committed")
    )
    value_widgets.connect_committed_value_widget(with_value, values.append)
    try:
        no_argument.stepUp()
        with_value.stepUp()
        assert notifications == ["committed"]
        assert values == [1]
        assert exceptions == []
    finally:
        no_argument.deleteLater()
        with_value.deleteLater()
        qapp.processEvents()
