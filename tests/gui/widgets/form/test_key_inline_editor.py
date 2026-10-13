"""A key-or-inline field shows one dropdown: Inline first, then library keys."""

from collections.abc import Callable, Iterator, Mapping
from typing import Annotated, Literal

import pytest
from pydantic import BaseModel, ConfigDict, Field, JsonValue
from PyQt6.QtGui import QAction
from qtpy.QtWidgets import QApplication, QComboBox, QLabel, QLineEdit, QWidget
from zcu_tools.cfg_model import ConstWaveformCfg, PulseSpec
from zcu_tools.gui.form import (
    FormEnv,
    FormErrors,
    FormInput,
    FormModel,
    FormState,
    FormWritePath,
    ModuleInput,
    WriteRejected,
)
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


class PulseKeyForm(FormModel):
    """A library-backed str-or-model pulse field."""

    drive: str | PulseSpec


class ModuleForm(FormModel):
    """A ModuleInput, the ref-or-inline module selection of workflow forms."""

    readout: ModuleInput


type Editor = FormEditor[FormModel]
type Write = Callable[[Mapping[FormWritePath, JsonValue]], None]
type Build = Callable[
    [type[FormModel], dict[str, JsonValue], FormEnv, Write | None], Editor
]

INLINE = "Inline"
KEYS = ["Q1.pulses.pi", "Q1.pulses.pi2"]


def library_env() -> FormEnv:
    """Two pulse modules and one waveform under component Q1."""
    pulse = PulseSpec(ch=0, nqz=1, freq=100.0, gain=0.2, waveform="Q1.waveforms.flat")
    other = PulseSpec(ch=1, nqz=2, freq=200.0, gain=0.4, waveform="Q1.waveforms.flat")
    return FormEnv(
        library=LibrarySnapshot(
            ModuleLibraryDocument(
                format="zcu.module-library",
                format_version="1.0",
                components={
                    "Q1": ComponentLibrary(
                        pulses={"pi": pulse, "pi2": other},
                        waveforms={"flat": ConstWaveformCfg(length=1.0)},
                    )
                },
            )
        )
    )


@pytest.fixture
def states() -> list[FormState[FormModel]]:
    """States built by the build fixture, in build order."""
    return []


@pytest.fixture
def build(qapp: QApplication, states: list[FormState[FormModel]]) -> Iterator[Build]:
    """Build editors over input, env and an optional host write; delete them later."""
    editors: list[Editor] = []

    def make(
        form: type[FormModel],
        data: dict[str, JsonValue],
        env: FormEnv,
        write: Write | None,
    ) -> Editor:
        state = FormState(FormInput(form, data))
        states.append(state)
        editor = (
            FormEditor(state, env)
            if write is None
            else FormEditor(state, env, write=write)
        )
        editors.append(editor)
        return editor

    yield make
    for editor in editors:
        editor.close()
        editor.deleteLater()
    qapp.processEvents()


def child[W: QWidget](view: QWidget, kind: type[W], name: str) -> W:
    """Find a documented public child by its complete object name."""
    widget = view.findChild(kind, name)
    assert isinstance(widget, kind), name
    return widget


def items(widget: QComboBox) -> list[tuple[str, object]]:
    """Return the dropdown's (text, data) items in order."""
    return [(widget.itemText(i), widget.itemData(i)) for i in range(widget.count())]


def choose(widget: QComboBox, text: str) -> None:
    """Select the item showing text, as a user would."""
    index = widget.findText(text)
    assert index >= 0, text
    widget.setCurrentIndex(index)


def test_str_or_model_dropdown_lists_inline_then_matching_keys(
    build: Build, states: list[FormState[FormModel]]
) -> None:
    editor = build(PulseKeyForm, {"drive": "Q1.pulses.pi"}, library_env(), None)
    state = states[-1]

    dropdown = child(editor, QComboBox, "form.input.drive")
    assert not dropdown.isEditable()
    assert items(dropdown) == [(INLINE, None), *((key, key) for key in KEYS)]
    assert dropdown.currentText() == "Q1.pulses.pi"
    assert editor.findChild(QComboBox, "form.mode.drive") is None
    assert editor.findChild(QAction, "form.mode.drive") is None
    assert editor.findChild(QLineEdit, "form.input.drive.gain") is None

    choose(dropdown, "Q1.pulses.pi2")
    assert state.input.data == {"drive": "Q1.pulses.pi2"}

    choose(child(editor, QComboBox, "form.input.drive"), INLINE)
    drive = state.input.data["drive"]
    assert isinstance(drive, dict)
    dropdown = child(editor, QComboBox, "form.input.drive")
    assert dropdown.currentIndex() == 0
    assert isinstance(editor.findChild(QLineEdit, "form.input.drive.gain"), QLineEdit)

    choose(dropdown, "Q1.pulses.pi")
    assert state.input.data == {"drive": "Q1.pulses.pi"}
    assert editor.findChild(QLineEdit, "form.input.drive.gain") is None


def test_object_selects_inline_and_shows_the_inline_rows(
    build: Build, states: list[FormState[FormModel]]
) -> None:
    editor = build(
        KeyInlineForm,
        {"tone": {"ch": 1, "gain": 0.25}, "shape": {"style": "ramp", "sigma": 0.2}},
        FormEnv(),
        None,
    )
    state = states[-1]

    for name in ("tone", "shape"):
        dropdown = child(editor, QComboBox, f"form.input.{name}")
        assert dropdown.currentIndex() == 0
        assert items(dropdown) == [(INLINE, None)]
    style = child(editor, QComboBox, "form.input.shape.style")
    assert style.currentData() == "ramp"
    assert child(editor, QLineEdit, "form.input.shape.sigma").text() == "0.2"

    child(editor, QLineEdit, "form.input.tone.gain").setText("0.3")
    child(editor, QLineEdit, "form.input.shape.sigma").setText("0.4")
    assert state.input.data == {
        "tone": {"ch": 1, "gain": 0.3},
        "shape": {"style": "ramp", "sigma": 0.4},
    }
    result = state.result(FormEnv())
    assert isinstance(result, KeyInlineForm)
    assert result.tone == Tone(ch=1, gain=0.3)
    assert result.shape == RampShape(sigma=0.4)


def test_choosing_inline_writes_the_inline_defaults_not_a_converted_key(
    build: Build, states: list[FormState[FormModel]]
) -> None:
    editor = build(
        KeyInlineForm,
        {"tone": "Q1.pulses.pi", "shape": "Q1.waveforms.flat"},
        FormEnv(),
        None,
    )
    state = states[-1]

    choose(child(editor, QComboBox, "form.input.tone"), INLINE)
    choose(child(editor, QComboBox, "form.input.shape"), INLINE)

    assert state.input.data == {
        "tone": {"gain": 0.5},
        "shape": {"style": "flat", "length": 1.0},
    }
    assert child(editor, QLineEdit, "form.input.tone.gain").text() == "0.5"
    assert child(editor, QComboBox, "form.input.shape.style").currentData() == "flat"


def test_a_key_outside_the_library_shows_as_missing_until_another_choice(
    build: Build, states: list[FormState[FormModel]]
) -> None:
    editor = build(PulseKeyForm, {"drive": "Q1.pulses.gone"}, library_env(), None)
    state = states[-1]

    dropdown = child(editor, QComboBox, "form.input.drive")
    assert items(dropdown) == [
        (INLINE, None),
        *((key, key) for key in KEYS),
        ("Q1.pulses.gone (missing)", "Q1.pulses.gone"),
    ]
    assert dropdown.currentText() == "Q1.pulses.gone (missing)"
    assert state.input.data == {"drive": "Q1.pulses.gone"}

    choose(dropdown, "Q1.pulses.pi2")
    assert state.input.data == {"drive": "Q1.pulses.pi2"}
    assert items(child(editor, QComboBox, "form.input.drive")) == [
        (INLINE, None),
        *((key, key) for key in KEYS),
    ]


def test_without_library_keys_the_dropdown_has_only_inline_and_a_missing_key(
    build: Build,
) -> None:
    editor = build(KeyInlineForm, {"tone": "Q1.pulses.pi"}, FormEnv(), None)

    tone = child(editor, QComboBox, "form.input.tone")
    assert items(tone) == [(INLINE, None), ("Q1.pulses.pi (missing)", "Q1.pulses.pi")]
    assert tone.currentIndex() == 1
    shape = child(editor, QComboBox, "form.input.shape")
    assert items(shape) == [(INLINE, None)]
    # Unfilled: nothing is selected and the field shows its required error.
    assert shape.currentIndex() == -1
    assert child(editor, QLabel, "form.error.shape").text() != ""


def ref(name: JsonValue) -> dict[str, JsonValue]:
    """A ModuleInput reference input naming name."""
    return {"readout": {"kind": "ref", "name": name}}


def test_module_input_dropdown_selects_a_ref_or_copies_it_inline(
    build: Build, states: list[FormState[FormModel]]
) -> None:
    env = library_env()
    editor = build(ModuleForm, ref("Q1.pulses.pi2"), env, None)
    state = states[-1]

    dropdown = child(editor, QComboBox, "form.input.readout")
    assert not dropdown.isEditable()
    assert items(dropdown) == [(INLINE, None), *((key, key) for key in KEYS)]
    assert dropdown.currentText() == "Q1.pulses.pi2"
    for path in ("readout.kind", "readout.name"):
        assert editor.findChild(QWidget, f"form.input.{path}") is None, path
    assert editor.findChild(QAction, "form.mode.readout") is None

    choose(dropdown, INLINE)
    readout = state.input.data["readout"]
    assert isinstance(readout, dict)
    assert readout["kind"] == "inline"
    library = env.library
    assert library is not None
    assert readout["module"] == library.module_spec("Q1.pulses.pi2").model_dump(
        mode="json"
    )
    assert child(editor, QComboBox, "form.input.readout").currentIndex() == 0
    assert child(editor, QLineEdit, "form.input.readout.module.gain").text() == "0.4"
    # Inline shows the module rows only; its kind tag has no row.
    assert editor.findChild(QWidget, "form.input.readout.kind") is None

    choose(child(editor, QComboBox, "form.input.readout"), "Q1.pulses.pi")
    assert state.input.data == ref("Q1.pulses.pi")
    assert editor.findChild(QWidget, "form.input.readout.module.gain") is None


def test_module_input_missing_key_shows_its_error_until_another_choice(
    build: Build, states: list[FormState[FormModel]]
) -> None:
    editor = build(ModuleForm, ref("Q1.pulses.gone"), library_env(), None)
    state = states[-1]

    dropdown = child(editor, QComboBox, "form.input.readout")
    assert dropdown.currentText() == "Q1.pulses.gone (missing)"
    assert dropdown.currentData() == "Q1.pulses.gone"
    error = child(editor, QLabel, "form.error.readout")
    assert "gone" in error.text()
    assert state.input.data == ref("Q1.pulses.gone")

    choose(dropdown, "Q1.pulses.pi")
    assert state.input.data == ref("Q1.pulses.pi")
    assert "Q1.pulses.gone (missing)" not in [
        text for text, _ in items(child(editor, QComboBox, "form.input.readout"))
    ]
    assert error.text() == ""


def test_module_input_without_library_offers_only_inline(
    build: Build,
) -> None:
    keyed = build(ModuleForm, ref("Q1.pulses.pi"), FormEnv(), None)
    assert items(child(keyed, QComboBox, "form.input.readout")) == [
        (INLINE, None),
        ("Q1.pulses.pi (missing)", "Q1.pulses.pi"),
    ]

    unfilled = build(ModuleForm, {}, FormEnv(), None)
    dropdown = child(unfilled, QComboBox, "form.input.readout")
    assert items(dropdown) == [(INLINE, None)]
    assert dropdown.currentIndex() == -1
    assert child(unfilled, QLabel, "form.error.readout").text() != ""


@pytest.mark.parametrize(
    ("form", "data", "shown"),
    [
        (PulseKeyForm, {"drive": "Q1.pulses.pi"}, "Q1.pulses.pi"),
        (ModuleForm, ref("Q1.pulses.pi"), "Q1.pulses.pi"),
    ],
)
def test_rejected_choice_shows_the_state_value_and_the_host_message(
    build: Build,
    states: list[FormState[FormModel]],
    form: type[FormModel],
    data: dict[str, JsonValue],
    shown: str,
) -> None:
    def write(_batch: Mapping[FormWritePath, JsonValue]) -> None:
        raise WriteRejected("Locked by the host")

    editor = build(form, data, library_env(), write)
    state = states[-1]
    (name,) = data

    for text in ("Q1.pulses.pi2", INLINE):
        choose(child(editor, QComboBox, f"form.input.{name}"), text)
        assert state.input.data == data
        assert child(editor, QComboBox, f"form.input.{name}").currentText() == shown
        assert child(editor, QLabel, f"form.error.{name}").text() == (
            "Locked by the host"
        )
        local = editor.local_parse_errors()
        assert isinstance(local, FormErrors)
        assert local.messages == {(name,): "Locked by the host"}

    editor.discard_local_text()
    assert editor.errors() is None
