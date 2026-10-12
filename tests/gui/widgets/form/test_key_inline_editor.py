"""A str-or-model field shows a Key/Inline switch instead of one repr text box."""

from collections.abc import Callable, Iterator
from typing import Annotated, Literal

import pytest
from pydantic import BaseModel, ConfigDict, Field, JsonValue
from qtpy.QtWidgets import QApplication, QComboBox, QLineEdit
from zcu_tools.cfg_model import ConstWaveformCfg, PulseSpec
from zcu_tools.gui.form import FormEnv, FormInput, FormModel, FormState
from zcu_tools.gui.widgets.form import FormEditor
from zcu_tools.resources.context.library import (
    ComponentLibrary,
    LibrarySnapshot,
    ModuleLibraryDocument,
)


class Tone(BaseModel):
    """The single inline model of a key-or-inline field."""

    model_config = ConfigDict(extra="forbid")

    ch: int
    gain: float = 0.5


class FlatShape(BaseModel):
    """First tagged branch: the default inline shape."""

    model_config = ConfigDict(extra="forbid")

    style: Literal["flat"] = "flat"
    length: float = 1.0


class RampShape(BaseModel):
    """Second tagged branch with its own field."""

    model_config = ConfigDict(extra="forbid")

    style: Literal["ramp"] = "ramp"
    length: float = 1.0
    sigma: float = 0.1


type Shape = Annotated[FlatShape | RampShape, Field(discriminator="style")]


class KeyInlineForm(FormModel):
    """One key-or-single-model field and one key-or-tagged-union field."""

    tone: str | Tone
    shape: str | Shape


type Editor = FormEditor[KeyInlineForm]
type Build = Callable[[dict[str, JsonValue]], tuple[Editor, FormState[KeyInlineForm]]]


@pytest.fixture
def make_editor(qapp: QApplication) -> Iterator[Build]:
    """Build editors over given input on the existing app; delete them afterwards."""
    editors: list[Editor] = []

    def build(data: dict[str, JsonValue]) -> tuple[Editor, FormState[KeyInlineForm]]:
        state = FormState(FormInput(KeyInlineForm, data))
        editor = FormEditor(state, FormEnv())
        editors.append(editor)
        return editor, state

    yield build
    for editor in editors:
        editor.deleteLater()
    qapp.processEvents()


def combo(editor: Editor, name: str) -> QComboBox:
    """Find a combo box by its public object name."""
    widget = editor.findChild(QComboBox, name)
    assert isinstance(widget, QComboBox), name
    return widget


def line(editor: Editor, name: str) -> QLineEdit:
    """Find a line edit by its public object name."""
    widget = editor.findChild(QLineEdit, name)
    assert isinstance(widget, QLineEdit), name
    return widget


def choose(widget: QComboBox, data: str) -> None:
    """Select the item carrying data, as a user would."""
    index = widget.findData(data)
    assert index >= 0, data
    widget.setCurrentIndex(index)


def test_editing_an_inline_sub_field_keeps_the_model(make_editor: Build) -> None:
    editor, state = make_editor(
        {"tone": {"ch": 1, "gain": 0.25}, "shape": {"style": "ramp", "sigma": 0.2}}
    )

    line(editor, "form.input.tone.gain").setText("0.3")
    line(editor, "form.input.shape.sigma").setText("0.4")

    assert state.input.data == {
        "tone": {"ch": 1, "gain": 0.3},
        "shape": {"style": "ramp", "sigma": 0.4},
    }
    result = state.result(FormEnv())
    assert isinstance(result, KeyInlineForm)
    assert result.tone == Tone(ch=1, gain=0.3)
    assert result.shape == RampShape(sigma=0.4)


def test_string_shows_key_mode_and_object_shows_inline_mode(
    make_editor: Build,
) -> None:
    editor, state = make_editor(
        {"tone": "Q1.pulses.pi", "shape": {"style": "ramp", "sigma": 0.2}}
    )

    mode = combo(editor, "form.mode.tone")
    assert [mode.itemData(index) for index in range(mode.count())] == [
        "key",
        "inline",
    ]
    assert mode.currentData() == "key"
    key = line(editor, "form.input.tone")
    assert key.text() == "Q1.pulses.pi"
    assert editor.findChild(QLineEdit, "form.input.tone.gain") is None
    assert combo(editor, "form.mode.shape").currentData() == "inline"
    style = combo(editor, "form.input.shape.style")
    assert {style.itemData(index) for index in range(style.count())} == {
        "flat",
        "ramp",
    }
    assert style.currentData() == "ramp"
    assert line(editor, "form.input.shape.sigma").text() == "0.2"
    assert editor.findChild(QLineEdit, "form.input.shape") is None

    key.setText("Q1.pulses.pi2")
    assert state.input.data["tone"] == "Q1.pulses.pi2"


def test_switching_mode_writes_that_mode_default_not_a_repr(
    make_editor: Build,
) -> None:
    editor, state = make_editor(
        {"tone": {"ch": 1, "gain": 0.25}, "shape": "Q1.waveforms.flat"}
    )

    choose(combo(editor, "form.mode.tone"), "key")
    choose(combo(editor, "form.mode.shape"), "inline")

    assert state.input.data == {
        "tone": "",
        "shape": {"style": "flat", "length": 1.0},
    }
    assert line(editor, "form.input.tone").text() == ""
    assert combo(editor, "form.input.shape.style").currentData() == "flat"
    assert line(editor, "form.input.shape.length").text() == "1.0"

    choose(combo(editor, "form.mode.tone"), "inline")
    choose(combo(editor, "form.input.shape.style"), "ramp")

    assert state.input.data == {
        "tone": {"gain": 0.5},
        "shape": {"style": "ramp", "length": 1.0, "sigma": 0.1},
    }
    assert line(editor, "form.input.tone.gain").text() == "0.5"
    assert line(editor, "form.input.shape.sigma").text() == "0.1"


class PulseKeyForm(FormModel):
    """A library-backed key-or-inline pulse field."""

    drive: str | PulseSpec


def test_captured_library_keys_are_choices_of_an_editable_key(
    qapp: QApplication,
) -> None:
    pulse = PulseSpec(ch=0, nqz=1, freq=100.0, gain=0.2, waveform="Q1.waveforms.flat")
    env = FormEnv(
        library=LibrarySnapshot(
            ModuleLibraryDocument(
                format="zcu.module-library",
                format_version="1.0",
                components={
                    "Q1": ComponentLibrary(
                        pulses={"pi": pulse, "pi2": pulse},
                        waveforms={"flat": ConstWaveformCfg(length=1.0)},
                    )
                },
            )
        )
    )
    state = FormState(FormInput(PulseKeyForm, {"drive": "Q1.pulses.pi"}))
    editor = FormEditor(state, env)
    try:
        key = editor.findChild(QComboBox, "form.input.drive")
        assert isinstance(key, QComboBox)
        assert key.isEditable()
        assert [key.itemText(index) for index in range(key.count())] == [
            "Q1.pulses.pi",
            "Q1.pulses.pi2",
        ]
        assert key.currentText() == "Q1.pulses.pi"

        key.setCurrentIndex(key.findText("Q1.pulses.pi2"))
        assert state.input.data == {"drive": "Q1.pulses.pi2"}
        line_edit = key.lineEdit()
        assert line_edit is not None
        line_edit.setText("Q1.pulses.custom")
        assert state.input.data == {"drive": "Q1.pulses.custom"}
    finally:
        editor.deleteLater()
        qapp.processEvents()
