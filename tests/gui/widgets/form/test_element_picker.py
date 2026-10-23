"""Named picker navigation keeps drafts and errors independent of participants."""

import json
from dataclasses import asdict
from pathlib import Path
from typing import Annotated

from qtpy.QtWidgets import QApplication
from zcu_tools.gui.form import Display, FormEnv, FormInput, FormModel, FormState
from zcu_tools.gui.form_adapter import FixedElements, FixedElementsForm
from zcu_tools.gui.widgets.form import FormEditor

from tests.gui._user_input import ComponentDriver
from tests.gui.form_adapter._fixed_models import named_form, named_input


def test_picker_shows_four_editable_fields_when_the_list_expands(
    qapp: QApplication, tmp_path: Path
):
    class Tone(FormModel):
        name: Annotated[str, Display(readonly=True)]
        freq: float
        gain: float
        phase: float

    class Tones(FixedElementsForm):
        tones: Annotated[list[Tone], Display(element_label="name")]

    Tones.fixed_elements = (FixedElements(("tones",)),)
    state = FormState(
        FormInput(
            Tones, {"tones": [{"name": "Q1", "freq": 100, "gain": 0.2, "phase": 0}]}
        )
    )
    editor = FormEditor(state, FormEnv())
    editor.resize(640, 340)
    editor.show()
    qapp.processEvents()
    driver = ComponentDriver(editor)
    try:
        driver.click(object_name="form.fold.tones")
        driver.screenshot(tmp_path / "01-four-fields.png")
        visible = {item.object_name: item for item in driver.visible().widgets}
        for field in ("name", "freq", "gain", "phase"):
            assert f"form.input.tones.0.{field}" in visible
        driver.enter_text(object_name="form.input.tones.0.freq", value="125")
        driver.screenshot(tmp_path / "02-edited-frequency.png")
        assert driver.observe(object_name="form.input.tones.0.freq").text == "125"
        driver.click(object_name="form.fold.tones")
        driver.screenshot(tmp_path / "03-list-collapsed.png")
        assert not driver.observe(object_name="form.input.tones.0.freq").visible
        driver.click(object_name="form.fold.tones")
        driver.screenshot(tmp_path / "04-list-reopened.png")
        assert driver.observe(object_name="form.input.tones.0.freq").text == "125"
    finally:
        editor.close()


def test_native_indexed_module_and_sweep_navigation_keeps_visible_drafts(
    qapp: QApplication,
    tmp_path: Path,
):
    from tests.gui.form_adapter._indexed_models import displayed_form, indexed_env

    env = indexed_env()
    definition = displayed_form()
    editor = FormEditor(FormState(definition.default_input(env)), env)
    editor.resize(850, 1100)
    editor.show()
    qapp.processEvents()
    driver = ComponentDriver(editor)
    try:
        driver.click(object_name="form.fold.tones")
        assert any(
            item.object_name == "form.input.tones.0.drive"
            for item in driver.visible().widgets
        )
        driver.click(object_name="form.fold.tones.0.sweep")
        driver.screenshot(tmp_path / "01-native-open.png")
        assert driver.observe(object_name="form.picker.tones").text == "0: A"
        driver.choose_option(object_name="form.picker.tones", value="1: B")
        assert driver.observe(object_name="form.input.tones.1.drive").visible
        driver.click(object_name="form.fold.tones.1.sweep")
        driver.invoke_context_action(
            object_name="form.input.tones.1.drive", action="Convert to inline"
        )
        driver.click(object_name="form.fold.tones.1.drive")
        driver.enter_text(
            object_name="form.input.tones.1.drive.module.gain", value="broken"
        )
        driver.screenshot(tmp_path / "02-inline-invalid.png")
        assert (
            "could not convert"
            in driver.observe(object_name="form.row.tones.1.drive.module.gain").tooltip
        )
        driver.choose_option(object_name="form.picker.tones", value="0: A")
        driver.screenshot(tmp_path / "03-sibling-reference.png")
        assert (
            driver.observe(object_name="form.input.tones.0.drive").text
            == "Q.pulses.drive"
        )
        assert driver.observe(object_name="form.input.participants").text == "[1, 0]"
        driver.choose_option(object_name="form.picker.tones", value="1: B")
        assert (
            driver.observe(object_name="form.input.tones.1.drive.module.gain").text
            == "broken"
        )
        driver.enter_text(
            object_name="form.input.tones.1.drive.module.gain", value="0.4"
        )
        driver.invoke_context_action(
            object_name="form.fold.tones.1.sweep", action="Switch to Center / Span"
        )
        driver.screenshot(tmp_path / "04-corrected-center.png")
        assert (
            driver.observe(object_name="form.input.tones.1.sweep.center").text == "12.0"
        )
        assert (
            driver.observe(object_name="form.input.tones.1.drive.module.gain").text
            == "0.4"
        )
        driver.choose_option(object_name="form.picker.tones", value="0: A")
        driver.screenshot(tmp_path / "05-sibling-range-preserved.png")
        assert driver.observe(object_name="form.input.tones.0.sweep.start").text == "2"
    finally:
        editor.close()


def test_picker_keeps_uncommitted_text_errors_and_participants(
    qapp: QApplication, tmp_path: Path
):
    state = FormState(named_input(named_form()))
    editor = FormEditor(state, FormEnv())
    editor.resize(640, 340)
    editor.show()
    qapp.processEvents()
    driver = ComponentDriver(editor)

    def snapshot(name: str) -> None:
        driver.screenshot(tmp_path / f"{name}.png")
        (tmp_path / f"{name}.json").write_text(
            json.dumps(asdict(driver.visible()), indent=2), encoding="utf-8"
        )

    try:
        driver.click(object_name="form.fold.tones")
        snapshot("01-open")
        assert any(
            item.object_name == "form.input.tones.0.gain"
            for item in driver.visible().widgets
        )
        assert driver.observe(object_name="form.picker.tones").text == "0: A"
        driver.enter_text(
            object_name="form.input.tones.0.gain", value="uncommitted", commit=False
        )
        snapshot("02-uncommitted")
        driver.choose_option(object_name="form.picker.tones", value="1: B")
        snapshot("03-second-element")
        assert driver.observe(object_name="form.input.tones.1.gain").visible
        assert driver.observe(object_name="form.input.tones.1.gain").text == "2"
        assert driver.observe(object_name="form.input.participants").text == "[1]"
        driver.choose_option(object_name="form.picker.tones", value="0: A")
        snapshot("04-returned-draft")
        assert (
            driver.observe(object_name="form.input.tones.0.gain").text == "uncommitted"
        )
        driver.enter_text(object_name="form.input.tones.0.gain", value="broken")
        snapshot("05-parse-error")
        assert (
            "could not convert"
            in driver.observe(object_name="form.row.tones.0.gain").tooltip
        )
        driver.choose_option(object_name="form.picker.tones", value="1: B")
        snapshot("06-hidden-error")
        assert "A" in driver.observe(object_name="form.row.tones").tooltip
        driver.choose_option(object_name="form.picker.tones", value="0: A")
        snapshot("07-error-restored")
        assert driver.observe(object_name="form.input.tones.0.gain").text == "broken"
        assert (
            "could not convert"
            in driver.observe(object_name="form.row.tones.0.gain").tooltip
        )
        driver.enter_text(object_name="form.input.tones.0.gain", value="3")
        driver.enter_text(object_name="form.input.participants", value="[1, 0]")
        snapshot("08-corrected-and-ordered")
        assert driver.observe(object_name="form.input.participants").text == "[1, 0]"
        assert (
            "could not convert"
            not in driver.observe(object_name="form.row.tones.0.gain").tooltip
        )
        editor.set_editing_enabled(enabled=False)
        driver.enter_text(object_name="form.input.tones.0.gain", value="8")
        driver.choose_option(object_name="form.picker.tones", value="1: B")
        snapshot("09-readonly-navigation")
        assert driver.observe(object_name="form.input.tones.1.gain").text == "2"
        driver.choose_option(object_name="form.picker.tones", value="0: A")
        snapshot("10-readonly-unchanged")
        assert driver.observe(object_name="form.input.tones.0.gain").text == "3"
    finally:
        editor.close()


def test_readonly_refusal_is_visible_and_navigation_still_works(
    qapp: QApplication, tmp_path: Path
):
    from pydantic import create_model
    from zcu_tools.gui import form as core
    from zcu_tools.gui import form_adapter as owner

    assert hasattr(owner, "FixedElementsForm"), "Missing declaring owner"

    class Element(core.FormModel):
        label: Annotated[str, core.Display(readonly=True)]
        gain: float

    model = create_model(
        "Owner",
        __base__=owner.FixedElementsForm,
        tones=(
            Annotated[
                list[Element],
                core.Display(element_label="label"),
                core.InputCommand(
                    "clear",
                    "Clear elements",
                    (),
                    lambda _snapshot, _context: core.FormCommandResult(None, None, []),
                ),
            ],
            [{"label": "A", "gain": 1}, {"label": "B", "gain": 2}],
        ),
        participants=(list[int], [1]),
    )
    model.fixed_elements = (
        owner.FixedElements(("tones",), "label", ("participants",)),
    )
    editor = FormEditor(core.FormState(model.default_input(FormEnv())), FormEnv())
    editor.resize(640, 340)
    editor.show()
    qapp.processEvents()
    driver = ComponentDriver(editor)
    try:
        driver.click(object_name="form.fold.tones")
        driver.screenshot(tmp_path / "01-owner-open.png")
        assert any(
            item.object_name == "form.input.tones.0.gain"
            for item in driver.visible().widgets
        )
        driver.invoke_context_action(
            object_name="form.fold.tones",
            action="Clear elements",
            evidence=tmp_path / "02-owner-menu.png",
        )
        driver.screenshot(tmp_path / "03-owner-refused.png")
        assert "Readonly" in driver.observe(object_name="form.row.tones").tooltip
        assert driver.observe(object_name="form.picker.tones").text == "0: A"
        driver.choose_option(object_name="form.picker.tones", value="1: B")
        assert driver.observe(object_name="form.input.tones.1.gain").visible
        driver.enter_text(object_name="form.input.tones.1.gain", value="4")
        driver.enter_text(object_name="form.input.participants", value="[4]")
        driver.screenshot(tmp_path / "04-owner-invalid-participant.png")
        assert driver.observe(object_name="form.input.tones.1.gain").text == "4"
        assert "4" in driver.observe(object_name="form.row.participants").tooltip
        driver.enter_text(object_name="form.input.participants", value="[1, 0]")
        driver.screenshot(tmp_path / "05-owner-recovered.png")
        assert driver.observe(object_name="form.input.participants").text == "[1, 0]"
        (tmp_path / "owner-visible.json").write_text(
            json.dumps(asdict(driver.visible()), indent=2), encoding="utf-8"
        )
    finally:
        editor.close()
