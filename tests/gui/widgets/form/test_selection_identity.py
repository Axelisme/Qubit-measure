"""A selector without a converter keeps the visible authored child draft."""

import json
import sys
from collections.abc import Mapping
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Literal

import pytest
from pydantic import Field, JsonValue, create_model
from qtpy.QtWidgets import QApplication, QLabel, QPushButton, QVBoxLayout, QWidget
from zcu_tools.gui import form as api
from zcu_tools.gui.form import FormEnv, FormInput, FormModel, FormState
from zcu_tools.gui.widgets.form import FormEditor

from tests.gui._user_input import ComponentDriver


class First(FormModel):
    kind: Literal["first"] = "first"
    amount: float = 2


class Second(FormModel):
    kind: Literal["second"] = "second"
    required: float


class SelectionForm(FormModel):
    branch: First | Second = Field(discriminator="kind")


@pytest.mark.parametrize("with_converter", [False, True], ids=["plain", "converter"])
def test_restored_unlisted_selection_is_shown_without_rewriting_raw_input(
    qapp: QApplication, tmp_path: Path, with_converter: bool
) -> None:
    conversions: list[JsonValue] = []

    def convert(
        _old: api.FormCommandSnapshot,
        selection: JsonValue,
        _context: api.FormCommandContext,
    ) -> api.FormConversionResult:
        conversions.append(selection)
        return api.FormConversionResult(None, selection)

    class ChoiceForm(FormModel):
        key: str

        @classmethod
        def options(cls, _env: FormEnv) -> Mapping[api.FormPath, tuple[api.Option, ...]]:
            return {("key",): (api.Option("A", "A"), api.Option("B", "B"))}

        @classmethod
        def converters(cls, _env: FormEnv) -> Mapping[api.FormPath, api.InputConverter]:
            return {("key",): api.InputConverter(convert)} if with_converter else {}

    authored = FormInput(ChoiceForm, {"key": "A"}, selections={("key",): "C"})
    state = FormState(FormInput.from_json(ChoiceForm, authored.to_json()))
    publications: list[bool] = []
    state.subscribe(lambda: publications.append(True))
    env = FormEnv()
    editor = FormEditor(state, env)
    editor.resize(640, 180)
    editor.show()
    qapp.processEvents()
    driver = ComponentDriver(editor)

    def snapshot(name: str) -> None:
        driver.screenshot(tmp_path / f"{name}.png")
        (tmp_path / f"{name}.json").write_text(
            json.dumps(asdict(driver.visible()), indent=2), encoding="utf-8"
        )

    try:
        snapshot("01-restored-selection")
        assert driver.observe(object_name="form.input.key").text == "C (missing)"
        assert "current options" in driver.observe(object_name="form.row.key").tooltip
        editor.refresh(env)
        snapshot("02-refreshed-selection")
        assert driver.observe(object_name="form.input.key").text == "C (missing)"
        assert state.input.same_as(authored)
        assert publications == []
        assert conversions == []

        driver.choose_option(object_name="form.input.key", value="B")
        snapshot("03-valid-selection")
        assert driver.observe(object_name="form.input.key").text == "B"
        assert "current options" not in driver.observe(object_name="form.row.key").tooltip
        assert state.input.data == {"key": "B"}
        # Converted raw B needs no override; a plain switch keeps its explicit choice.
        assert state.input.selections == ({} if with_converter else {("key",): "B"})
        assert publications == [True]
        assert conversions == (["B"] if with_converter else [])
    finally:
        editor.close()


def test_selection_only_keeps_visible_child_and_local_parse_error(
    qapp: QApplication, tmp_path: Path
):
    editor = FormEditor(
        FormState(FormInput(SelectionForm, {"branch": {"kind": "first", "amount": 2}})),
        FormEnv(),
    )
    editor.resize(620, 300)
    editor.show()
    qapp.processEvents()
    driver = ComponentDriver(editor)

    def snapshot(name: str) -> None:
        driver.screenshot(tmp_path / f"{name}.png")
        (tmp_path / f"{name}.json").write_text(
            json.dumps(asdict(driver.visible()), indent=2), encoding="utf-8"
        )

    try:
        snapshot("01-first")
        driver.enter_text(object_name="form.input.branch.amount", value="broken")
        snapshot("02-local-error")
        driver.choose_option(object_name="form.input.branch", value="second")
        snapshot("03-new-selection-old-branch")
        assert driver.observe(object_name="form.input.branch").text == "second"
        assert driver.observe(object_name="form.input.branch.amount").text == "broken"
    finally:
        editor.close()


def test_selection_only_synthetic_host_accepts_and_discards_authored_identity(
    qapp: QApplication, tmp_path: Path
):
    state = FormState(
        FormInput(SelectionForm, {"branch": {"kind": "first", "amount": 2}})
    )
    assert hasattr(state, "is_dirty"), "Hosts need selection-aware dirty observation"
    host = QWidget()
    host.resize(620, 320)
    layout = QVBoxLayout(host)
    status = QLabel("Saved")
    status.setObjectName("synthetic.status")
    layout.addWidget(status)
    layout.addWidget(FormEditor(state, FormEnv(), host))
    accept = QPushButton("Accept base")
    discard = QPushButton("Discard")
    layout.addWidget(accept)
    layout.addWidget(discard)
    accept.clicked.connect(lambda: state.reset(state.input))
    discard.clicked.connect(lambda: state.reset(state.base))
    state.subscribe(lambda: status.setText("Unsaved" if state.is_dirty else "Saved"))
    host.show()
    qapp.processEvents()
    driver = ComponentDriver(host)

    def snapshot(name: str) -> None:
        driver.screenshot(tmp_path / f"{name}.png")
        (tmp_path / f"{name}.json").write_text(
            json.dumps(asdict(driver.visible()), indent=2), encoding="utf-8"
        )

    try:
        snapshot("01-saved")
        driver.choose_option(object_name="form.input.branch", value="second")
        snapshot("02-selection-only-dirty")
        assert driver.observe(object_name="synthetic.status").text == "Unsaved"
        assert driver.observe(object_name="form.input.branch.amount").text == "2"
        driver.click("Accept base")
        snapshot("03-accepted-base")
        assert driver.observe(object_name="synthetic.status").text == "Saved"
        assert driver.observe(object_name="form.input.branch").text == "second"
        driver.choose_option(object_name="form.input.branch", value="first")
        snapshot("04-dirty-again")
        assert driver.observe(object_name="synthetic.status").text == "Unsaved"
        driver.click("Discard")
        snapshot("05-discarded")
        assert driver.observe(object_name="synthetic.status").text == "Saved"
        assert driver.observe(object_name="form.input.branch").text == "second"
        assert driver.observe(object_name="form.input.branch.amount").text == "2"
        driver.choose_option(object_name="form.input.branch", value="second")
        assert driver.observe(object_name="synthetic.status").text == "Saved"
    finally:
        host.close()


@pytest.mark.parametrize("guarded", [False, True], ids=["direct", "guarded"])
def test_converter_rejection_restores_selector_and_preserves_visible_drafts(
    qapp: QApplication,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    guarded: bool,
):
    assert hasattr(api, "InputConverter"), (
        "Selection writes need the converter contract"
    )

    def reject(old, selection, context):
        raise api.FormCommandError("Cannot convert missing participating value")

    model = create_model(
        "RejectingSelection",
        __base__=SelectionForm,
        branch=(
            Annotated[First | Second, api.InputConverter(reject)],
            Field(discriminator="kind"),
        ),
    )
    env = FormEnv()
    state = FormState(FormInput(model, {"branch": {"kind": "first", "amount": 2}}))
    events = []
    calls = []
    unhandled = []
    state.subscribe(lambda: events.append(state.input.data))
    monkeypatch.setattr(sys, "excepthook", lambda *args: unhandled.append(args))

    def write(batch: Mapping[api.FormWritePath, JsonValue]):
        calls.append(batch)
        state.write(batch, env)

    def command(path, name):
        state.run_command(path, name, env)

    editor = FormEditor(
        state,
        env,
        write=write if guarded else None,
        run_command=command if guarded else None,
    )
    editor.resize(620, 300)
    editor.show()
    qapp.processEvents()
    driver = ComponentDriver(editor)
    try:
        driver.enter_text(object_name="form.input.branch.amount", value="broken")
        driver.screenshot(tmp_path / "01-child-draft.png")
        driver.choose_option(object_name="form.input.branch", value="second")
        driver.screenshot(tmp_path / "02-rejected-selection.png")
        (tmp_path / "02-rejected-selection.json").write_text(
            json.dumps(asdict(driver.visible()), indent=2), encoding="utf-8"
        )
        assert unhandled == [], "Expected converter refusal must not escape the Qt slot"
        assert driver.observe(object_name="form.input.branch").text == "first"
        assert (
            "Cannot convert missing participating value"
            in driver.observe(object_name="form.row.branch").tooltip
        )
        assert driver.observe(object_name="form.input.branch.amount").text == "broken"
        assert state.input.data == {"branch": {"kind": "first", "amount": 2}}
        assert events == []
        assert len(calls) == (1 if guarded else 0)
    finally:
        editor.close()
