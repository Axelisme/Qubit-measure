"""A module union with a named nested member union edits its active branch."""

from collections.abc import Iterator
from typing import Annotated, Literal

import pytest
from pydantic import BaseModel, ConfigDict, Field
from qtpy.QtWidgets import QApplication, QComboBox, QLineEdit
from zcu_tools.gui.form import FormEnv, FormInput, FormModel, FormState
from zcu_tools.gui.widgets.form import FormEditor

from tests.gui._form_converters import RebuildingForm
from tests.gui.widgets.form._sections import expand_sections
from tests.gui.widgets.form._text_input import commit_text


class DriveModule(BaseModel):
    """Inline drive Spec: a single-model member of the module union."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["pulse"] = "pulse"
    gain: float = 0.5


class ShortReadoutModule(BaseModel):
    """Nested readout member without the pulse readout's gain."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["sample/short"] = "sample/short"
    ro_length: float = 1.0


class PulseReadoutModule(BaseModel):
    """Nested readout member with its own branch-only gain."""

    model_config = ConfigDict(extra="forbid")

    type: Literal["readout/pulse"] = "readout/pulse"
    ro_length: float = 2.0
    ro_gain: float = 0.2


type ReadoutModule = Annotated[
    ShortReadoutModule | PulseReadoutModule, Field(discriminator="type")
]
"""Named readout union nested in the module union on the same tag key."""

type AnyModule = Annotated[DriveModule | ReadoutModule, Field(discriminator="type")]


class ModuleRefChoice(FormModel):
    """Library reference branch of a module selection."""

    kind: Literal["ref"] = "ref"
    name: str


class ModuleInlineChoice(RebuildingForm):
    """Inline branch whose module is the nested tagged module union."""

    branch_models = {("module",): (DriveModule, ShortReadoutModule, PulseReadoutModule)}

    kind: Literal["inline"] = "inline"
    module: AnyModule


type ModuleChoice = Annotated[
    ModuleRefChoice | ModuleInlineChoice, Field(discriminator="kind")
]


class InlineModuleForm(RebuildingForm):
    """Drive and readout selections shaped like workflow module inputs."""

    branch_models = {
        ("qub_pulse",): (ModuleRefChoice, ModuleInlineChoice),
        ("readout",): (ModuleRefChoice, ModuleInlineChoice),
    }

    qub_pulse: ModuleChoice
    readout: ModuleChoice


@pytest.fixture
def editor_state(
    qapp: QApplication,
) -> Iterator[tuple[FormEditor[InlineModuleForm], FormState[InlineModuleForm]]]:
    """Own a view over inline drive and pulse-readout input on the existing app."""
    state = FormState(
        FormInput(
            InlineModuleForm,
            {
                "qub_pulse": {
                    "kind": "inline",
                    "module": {"type": "pulse", "gain": 0.4},
                },
                "readout": {
                    "kind": "inline",
                    "module": {
                        "type": "readout/pulse",
                        "ro_length": 3.0,
                        "ro_gain": 0.25,
                    },
                },
            },
        )
    )
    editor = FormEditor(state, FormEnv())
    expand_sections(
        editor, "qub_pulse", "qub_pulse.module", "readout", "readout.module"
    )
    yield editor, state
    editor.deleteLater()
    qapp.processEvents()


def selector(editor: FormEditor[InlineModuleForm], path: str) -> QComboBox:
    """Find the option dropdown of a dotted union field path."""
    widget = editor.findChild(QComboBox, f"form.input.{path}")
    assert isinstance(widget, QComboBox), path
    return widget


def text(editor: FormEditor[InlineModuleForm], path: str) -> QLineEdit:
    """Find the text input of an active dotted model-field path."""
    widget = editor.findChild(QLineEdit, f"form.input.{path}")
    assert isinstance(widget, QLineEdit), path
    return widget


def labels(combo: QComboBox) -> list[str]:
    """The dropdown's item texts in order."""
    return [combo.itemText(index) for index in range(combo.count())]


def test_nested_module_union_shows_and_edits_the_selected_inline_branch(
    editor_state: tuple[FormEditor[InlineModuleForm], FormState[InlineModuleForm]],
) -> None:
    editor, state = editor_state
    env = FormEnv()
    for name in ("readout", "qub_pulse"):
        assert selector(editor, name).currentText() == "inline", name
        combo = selector(editor, f"{name}.module")
        # The named nested union expands into its members, one option each.
        assert labels(combo) == ["pulse", "sample/short", "readout/pulse"], name
        for tag in ("kind", "module.type"):
            assert editor.findChild(QComboBox, f"form.input.{name}.{tag}") is None
    assert selector(editor, "readout.module").currentText() == "readout/pulse"
    assert text(editor, "readout.module.ro_length").text() == "3.0"
    gain = text(editor, "readout.module.ro_gain")
    assert gain.text() == "0.25"
    assert selector(editor, "qub_pulse.module").currentText() == "pulse"
    assert text(editor, "qub_pulse.module.gain").text() == "0.4"

    commit_text(gain, "0.3")
    assert state.input.data["readout"] == {
        "kind": "inline",
        "module": {"type": "readout/pulse", "ro_length": 3.0, "ro_gain": 0.3},
    }
    result = state.result(env)
    assert isinstance(result, InlineModuleForm)
    assert isinstance(result.readout, ModuleInlineChoice)
    assert result.readout.module == PulseReadoutModule(ro_length=3.0, ro_gain=0.3)

    combo = selector(editor, "readout.module")
    combo.setCurrentIndex(combo.findText("sample/short"))
    result = state.result(env)
    assert isinstance(result, InlineModuleForm)
    assert isinstance(result.readout, ModuleInlineChoice)
    # The new member starts from its own defaults.
    assert result.readout.module == ShortReadoutModule()
    assert selector(editor, "readout.module").currentText() == "sample/short"
    assert text(editor, "readout.module.ro_length").text() == "1.0"
    assert editor.findChild(QLineEdit, "form.input.readout.module.ro_gain") is None
