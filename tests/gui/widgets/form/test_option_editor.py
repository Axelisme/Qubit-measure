"""Every selection field shows one dropdown drawn only from x-form-options."""

from collections.abc import Callable, Iterator, Mapping
from enum import StrEnum
from typing import Annotated, Literal

import pytest
from pydantic import BaseModel, ConfigDict, Field, JsonValue
from PyQt6.QtGui import QAction
from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QLineEdit,
    QWidget,
)
from zcu_tools.cfg_model import (
    ConfigBase,
    ConstWaveformCfg,
    PulseCfg,
    PulseSpec,
    ResetCfg,
)
from zcu_tools.gui import form_adapter
from zcu_tools.gui.form import (
    FormCommandResult,
    FormEnv,
    FormErrors,
    FormInput,
    FormModel,
    FormPath,
    FormState,
    FormWritePath,
    Option,
    OptionCommand,
    WriteRejected,
)
from zcu_tools.gui.widgets.form import FormEditor
from zcu_tools.resources.context.library import (
    ComponentLibrary,
    LibrarySnapshot,
    ModuleLibraryDocument,
)

from tests.gui._form_converters import RebuildingForm
from tests.gui.widgets.form._field_errors import field_error

type Editor = FormEditor[FormModel]
type Batch = Mapping[FormWritePath, JsonValue]
type Write = Callable[[Batch], None]
type Command = Callable[[FormWritePath, str], None]
type Build = Callable[
    [FormState[FormModel], FormEnv, Write | None, Command | None], Editor
]

KEY = "Q.waveforms.flat"


def waveform_env() -> FormEnv:
    """One captured const waveform key, detached from its library document."""
    return FormEnv(
        library=LibrarySnapshot(
            ModuleLibraryDocument(
                format="zcu.module-library",
                format_version="2.0",
                components={
                    "Q": ComponentLibrary(
                        waveforms={"flat": ConstWaveformCfg(length=0.1)}
                    )
                },
            )
        )
    )


def module_env() -> FormEnv:
    """One captured pulse module key, Q1.pulses.pi."""
    pulse = PulseSpec(
        ch=0, nqz=1, freq=100.0, gain=0.2, waveform=ConstWaveformCfg(length=0.1)
    )
    return FormEnv(
        library=LibrarySnapshot(
            ModuleLibraryDocument(
                format="zcu.module-library",
                format_version="2.0",
                components={"Q1": ComponentLibrary(pulses={"pi": pulse})},
            )
        )
    )


def pulse_state(waveform: str, env: FormEnv) -> FormState[FormModel]:
    """A generated persistent pulse form whose waveform holds the key waveform."""
    form = form_adapter.module_form(PulseSpec)
    spec = PulseSpec(ch=0, nqz=1, freq=100.0, gain=0.25, waveform=waveform)
    return FormState(form.input_from_spec(spec, env))


def new_state(
    model: type[FormModel], data: dict[str, JsonValue]
) -> FormState[FormModel]:
    """A state over model's input data, typed for the shared editor builder."""
    return FormState(FormInput(model, data))


@pytest.fixture
def build(qapp: QApplication) -> Iterator[Build]:
    """Build editors over a state with optional host routes; delete them later."""
    editors: list[Editor] = []

    def make(
        state: FormState[FormModel],
        env: FormEnv,
        write: Write | None,
        command: Command | None,
    ) -> Editor:
        if write is None and command is None:
            editor = FormEditor(state, env)
        else:
            editor = FormEditor(state, env, write=write, run_command=command)
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


def schema_items(
    state: FormState[FormModel], env: FormEnv, name: str
) -> list[tuple[str, JsonValue]]:
    """(label, value) of the top-level field name's core-described options."""
    target = state.describe((name,), env).target
    assert target is not None and target.options is not None
    return [(option.label, option.value) for option in target.options]


def commands(widget: QComboBox) -> list[tuple[str, str, bool]]:
    """(object name, text, enabled) of the right-click actions of a dropdown."""
    return [
        (action.objectName(), action.text(), action.isEnabled())
        for action in widget.actions()
    ]


def command(view: QWidget, path: str, name: str) -> QAction:
    """The documented right-click action of one option command."""
    action = view.findChild(QAction, f"form.command.{path}.{name}")
    assert isinstance(action, QAction), name
    return action


def recording_write(
    state: FormState[FormModel],
    env: FormEnv,
    batches: list[dict[FormWritePath, JsonValue]],
) -> Write:
    """A host write that records each batch and installs it into state."""

    def write(batch: Batch) -> None:
        batches.append(dict(batch))
        state.write(batch, env)

    return write


def test_a_valid_waveform_key_selects_its_own_option_label_and_value(
    build: Build,
) -> None:
    env = waveform_env()
    state = pulse_state(KEY, env)
    editor = build(state, env, None, None)
    current = state.input.data["waveform"]
    declared = form_adapter.module_form(PulseSpec).model.options(env)[("waveform",)]
    first = next(option for option in declared if option.matches(current))

    dropdown = child(editor, QComboBox, "form.input.waveform")
    assert dropdown.currentText() == first.label == KEY
    assert dropdown.currentData() == first.value == current
    assert Option(dropdown.currentText(), dropdown.currentData()).matches(current)
    assert items(dropdown) == schema_items(state, env, "waveform")
    assert not any(text.endswith("(missing)") for text, _ in items(dropdown))
    assert state.input.data["waveform"] == {"kind": "ref", "name": KEY}


def test_choosing_an_option_writes_its_value_once_and_draws_its_fields(
    build: Build,
) -> None:
    env = waveform_env()
    state = pulse_state(KEY, env)
    batches: list[dict[FormWritePath, JsonValue]] = []
    editor = build(
        state, env, recording_write(state, env, batches), lambda _p, _n: None
    )

    choose(child(editor, QComboBox, "form.input.waveform"), "Inline: gauss")

    gauss: JsonValue = {"kind": "inline", "module": {"style": "gauss"}}
    assert batches == [{("waveform",): gauss}]
    assert Option("Inline: gauss", gauss).matches(state.input.data["waveform"])
    dropdown = child(editor, QComboBox, "form.input.waveform")
    assert dropdown.currentText() == "Inline: gauss"
    assert dropdown.currentData() == gauss
    for path in ("waveform.module.length", "waveform.module.sigma"):
        assert editor.findChild(QWidget, f"form.input.{path}") is not None, path
    # The option decides the tags: no kind or style row is drawn.
    for path in ("waveform.kind", "waveform.module", "waveform.module.style"):
        assert editor.findChild(QWidget, f"form.input.{path}") is None, path

    choose(dropdown, KEY)

    assert batches[-1] == {("waveform",): {"kind": "ref", "name": KEY}}
    assert state.input.data["waveform"] == {"kind": "ref", "name": KEY}
    assert editor.findChild(QWidget, "form.input.waveform.module.length") is None


def test_unmatched_input_shows_the_current_value_as_missing_and_keeps_state(
    build: Build,
) -> None:
    env = waveform_env()
    state = pulse_state("Q.waveforms.gone", env)
    editor = build(state, env, None, None)
    current = state.input.data["waveform"]

    dropdown = child(editor, QComboBox, "form.input.waveform")
    assert items(dropdown)[:-1] == schema_items(state, env, "waveform")
    assert items(dropdown)[-1] == (
        '{"kind": "ref", "name": "Q.waveforms.gone"} (missing)',
        current,
    )
    assert dropdown.currentIndex() == dropdown.count() - 1
    assert state.input.data["waveform"] == current

    choose(dropdown, KEY)

    assert state.input.data["waveform"] == {"kind": "ref", "name": KEY}
    shown = items(child(editor, QComboBox, "form.input.waveform"))
    assert shown == schema_items(state, env, "waveform")


class ModuleCore(ConfigBase):
    """A nullable single-kind module and a nullable multi-kind module."""

    pulse: PulseCfg | None = None
    reset: ResetCfg | None = None


def test_nullable_modules_offer_none_keys_and_inline_kinds_without_enabled(
    build: Build,
) -> None:
    env = module_env()
    form = form_adapter.Form(ModuleCore, {})
    state = new_state(form.model, {"pulse": None, "reset": None})
    editor = build(state, env, None, None)

    pulse = child(editor, QComboBox, "form.input.pulse")
    assert [text for text, _ in items(pulse)] == ["None", "Q1.pulses.pi", "Inline"]
    assert pulse.currentText() == "None"
    reset = child(editor, QComboBox, "form.input.reset")
    assert items(reset) == schema_items(state, env, "reset")
    assert [text for text, _ in items(reset)] == [
        "None",
        "Inline: reset/pulse",
        "Inline: reset/two_pulse",
        "Inline: reset/bath",
    ]
    for name in ("pulse", "reset"):
        assert editor.findChild(QCheckBox, f"form.enabled.{name}") is None, name

    choose(reset, "Inline: reset/two_pulse")

    selected = state.input.data["reset"]
    assert isinstance(selected, dict)
    two_pulse: JsonValue = {"kind": "inline", "module": {"type": "reset/two_pulse"}}
    assert Option("Inline: reset/two_pulse", two_pulse).matches(selected)
    assert field_error(editor, "reset.module.pulse1_cfg")
    assert editor.findChild(QWidget, "form.input.reset.module.type") is None
    assert editor.findChild(QCheckBox, "form.enabled.reset") is None

    choose(child(editor, QComboBox, "form.input.reset"), "None")

    assert state.input.data["reset"] is None
    assert editor.findChild(QWidget, "form.input.reset.module.pulse1_cfg") is None


def test_default_none_module_selection_via_component_events(
    build: Build, qapp: QApplication, tmp_path
) -> None:
    from tests.gui._user_input import ComponentDriver

    class EventCore(ConfigBase):
        pulse: PulseCfg | None = None

    env = module_env()
    form = form_adapter.Form(EventCore, {})
    state = FormState(form.default_input(env))
    editor = build(state, env, None, None)
    editor.resize(760, 360)
    editor.show()
    qapp.processEvents()
    driver = ComponentDriver(editor)

    driver.screenshot(tmp_path / "01-default-none.png")
    assert driver.observe(object_name="form.input.pulse").text == "None"

    driver.choose_option(object_name="form.input.pulse", value="Q1.pulses.pi")
    driver.screenshot(tmp_path / "02-selected-module.png")
    assert driver.observe(object_name="form.input.pulse").text == "Q1.pulses.pi"

    driver.choose_option(object_name="form.input.pulse", value="None")
    driver.screenshot(tmp_path / "03-disabled-again.png")
    assert driver.observe(object_name="form.input.pulse").text == "None"


class Color(StrEnum):
    """An Enum leaf; its options come from the core."""

    RED = "red"
    BLUE = "blue"


class PickForm(FormModel):
    """Literal, nullable Literal, Enum and choices leaves."""

    nqz: Literal[1, 2] = 1
    maybe: Literal["a", "b"] | None = None
    color: Color
    device: str = "yoko1"

    @classmethod
    def choices(cls, _env: FormEnv, /) -> Mapping[FormPath, tuple[JsonValue, ...]]:
        """Offer two captured device names."""
        return {("device",): ("yoko1", "yoko2")}


def test_literal_enum_and_choice_leaves_use_the_same_option_dropdown(
    build: Build,
) -> None:
    env = FormEnv()
    data: dict[str, JsonValue] = {
        "nqz": 1,
        "maybe": None,
        "color": "red",
        "device": "yoko1",
    }
    state = new_state(PickForm, data)
    editor = build(state, env, None, None)

    for name in ("nqz", "maybe", "color", "device"):
        dropdown = child(editor, QComboBox, f"form.input.{name}")
        assert items(dropdown) == schema_items(state, env, name), name
        assert editor.findChild(QAction, f"form.mode.{name}") is None, name
    assert child(editor, QComboBox, "form.input.maybe").currentText() == "None"

    choose(child(editor, QComboBox, "form.input.nqz"), "2")
    assert state.input.data["nqz"] == 2
    assert type(state.input.data["nqz"]) is int
    choose(child(editor, QComboBox, "form.input.maybe"), "b")
    assert state.input.data["maybe"] == "b"

    state.write({("device",): "yoko9"}, env)

    device = child(editor, QComboBox, "form.input.device")
    assert items(device)[-1] == ("yoko9 (missing)", "yoko9")
    assert device.currentText() == "yoko9 (missing)"
    assert state.input.data["device"] == "yoko9"
    assert field_error(editor, "device") != ""

    choose(device, "yoko2")
    assert state.input.data["device"] == "yoko2"
    assert field_error(editor, "device") == ""


class Flat(BaseModel):
    """First branch of a tagged union."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["flat"] = "flat"
    length: float = 1.0


class Ramp(BaseModel):
    """Second branch with its own defaults."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["ramp"] = "ramp"
    length: float = 2.0
    sigma: float = 0.1


class ShapeForm(RebuildingForm):
    """One tagged union field; the core lists one option per branch."""

    branch_models = {("shape",): (Flat, Ramp)}

    shape: Annotated[Flat | Ramp, Field(discriminator="kind")]


def test_tagged_union_dropdown_draws_the_selected_branch_and_starts_new_ones_fresh(
    build: Build,
) -> None:
    env = FormEnv()
    state = new_state(ShapeForm, {"shape": {"kind": "flat", "length": 5.0}})
    editor = build(state, env, None, None)

    dropdown = child(editor, QComboBox, "form.input.shape")
    assert items(dropdown) == [("flat", {"kind": "flat"}), ("ramp", {"kind": "ramp"})]
    assert dropdown.currentText() == "flat"
    assert editor.findChild(QWidget, "form.input.shape.kind") is None
    assert child(editor, QWidget, "form.input.shape.length") is not None

    choose(dropdown, "ramp")

    assert state.input.data["shape"] == {"kind": "ramp", "length": 2.0, "sigma": 0.1}
    assert child(editor, QComboBox, "form.input.shape").currentText() == "ramp"
    assert child(editor, QWidget, "form.input.shape.sigma") is not None
    assert editor.findChild(QWidget, "form.input.shape.kind") is None


class One(BaseModel):
    """A numeric-tag branch."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal[1] = 1
    a_only: float = 3.0


class Three(BaseModel):
    """Another numeric-tag branch."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal[3] = 3
    b_only: float = 4.0


class NumericForm(RebuildingForm):
    """Numeric tags are written as typed JSON numbers."""

    branch_models = {("branch",): (One, Three)}

    branch: Annotated[One | Three, Field(discriminator="kind")]


def test_numeric_tags_are_written_as_typed_option_values(build: Build) -> None:
    env = FormEnv()
    state = new_state(NumericForm, {"branch": {"kind": 1}})
    editor = build(state, env, None, None)

    dropdown = child(editor, QComboBox, "form.input.branch")
    assert items(dropdown) == [("1", {"kind": 1}), ("3", {"kind": 3})]
    choose(dropdown, "3")

    branch = state.input.data["branch"]
    assert isinstance(branch, dict)
    assert type(branch["kind"]) is int and branch["kind"] == 3
    assert child(editor, QWidget, "form.input.branch.b_only") is not None
    assert editor.findChild(QWidget, "form.input.branch.a_only") is None


class AliasFlat(BaseModel):
    """A branch with two intrinsic tags; plateau is an alias of flat."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["flat", "plateau"] = "flat"
    length: float = 1.0


class AliasShapeForm(RebuildingForm):
    """A tagged union whose first branch accepts an alias tag."""

    branch_models = {("shape",): (AliasFlat, Ramp)}

    shape: Annotated[AliasFlat | Ramp, Field(discriminator="kind")]


def test_an_alias_tag_is_its_own_option_with_its_branch_fields(build: Build) -> None:
    env = FormEnv()
    state = new_state(AliasShapeForm, {"shape": {"kind": "plateau", "length": 5.0}})
    editor = build(state, env, None, None)

    dropdown = child(editor, QComboBox, "form.input.shape")
    assert items(dropdown) == schema_items(state, env, "shape")
    assert items(dropdown) == [
        ("flat", {"kind": "flat"}),
        ("plateau", {"kind": "plateau"}),
        ("ramp", {"kind": "ramp"}),
    ]
    assert dropdown.currentText() == "plateau"
    assert dropdown.currentData() == {"kind": "plateau"}
    assert Option("plateau", {"kind": "plateau"}).matches(state.input.data["shape"])
    assert child(editor, QWidget, "form.input.shape.length") is not None
    for path in ("shape.kind", "shape.sigma"):
        assert editor.findChild(QWidget, f"form.input.{path}") is None, path

    choose(dropdown, "ramp")
    assert state.input.data["shape"] == {"kind": "ramp", "length": 2.0, "sigma": 0.1}

    choose(child(editor, QComboBox, "form.input.shape"), "plateau")

    # The alias starts its branch from fresh defaults, like any other option.
    assert state.input.data["shape"] == {"kind": "plateau", "length": 1.0}
    assert child(editor, QComboBox, "form.input.shape").currentText() == "plateau"
    assert editor.findChild(QWidget, "form.input.shape.sigma") is None


class AliasOne(BaseModel):
    """A numeric-tag branch with the alias tag 2."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal[1, 2] = 1
    a_only: float = 3.0


class NumericAliasForm(RebuildingForm):
    """A numeric tagged union whose first branch accepts an alias tag."""

    branch_models = {("branch",): (AliasOne, Three)}

    branch: Annotated[AliasOne | Three, Field(discriminator="kind")]


def test_a_numeric_alias_tag_is_a_typed_option_with_its_branch_fields(
    build: Build,
) -> None:
    env = FormEnv()
    state = new_state(NumericAliasForm, {"branch": {"kind": 2, "a_only": 7.0}})
    editor = build(state, env, None, None)

    dropdown = child(editor, QComboBox, "form.input.branch")
    assert items(dropdown) == [
        ("1", {"kind": 1}),
        ("2", {"kind": 2}),
        ("3", {"kind": 3}),
    ]
    assert dropdown.currentText() == "2"
    data = dropdown.currentData()
    assert isinstance(data, dict) and type(data["kind"]) is int
    assert Option("2", {"kind": 2}).matches(state.input.data["branch"])
    assert child(editor, QWidget, "form.input.branch.a_only") is not None
    assert editor.findChild(QWidget, "form.input.branch.b_only") is None

    choose(dropdown, "3")
    assert state.input.data["branch"] == {"kind": 3, "b_only": 4.0}

    choose(child(editor, QComboBox, "form.input.branch"), "1")

    branch = state.input.data["branch"]
    assert branch == {"kind": 1, "a_only": 3.0}
    assert isinstance(branch, dict) and type(branch["kind"]) is int
    assert child(editor, QWidget, "form.input.branch.a_only") is not None


def test_a_key_option_lists_its_commands_and_runs_them_on_the_state(
    build: Build,
) -> None:
    env = waveform_env()
    state = pulse_state(KEY, env)
    editor = build(state, env, None, None)
    dropdown = child(editor, QComboBox, "form.input.waveform")

    assert dropdown.contextMenuPolicy() == Qt.ContextMenuPolicy.ActionsContextMenu
    assert commands(dropdown) == [
        ("form.command.waveform.to_inline", "Convert to inline", True)
    ]

    command(editor, "waveform", "to_inline").trigger()

    assert state.input.data["waveform"] == {
        "kind": "inline",
        "module": {"style": "const", "length": 0.1},
    }
    dropdown = child(editor, QComboBox, "form.input.waveform")
    assert dropdown.currentText() == "Inline: const"
    # The inline option declares no command.
    assert commands(dropdown) == []
    assert child(editor, QWidget, "form.input.waveform.module.length") is not None


def test_host_command_route_receives_the_command_and_its_rejection_shows(
    build: Build,
) -> None:
    env = waveform_env()
    state = pulse_state(KEY, env)
    calls: list[tuple[FormWritePath, str]] = []
    rejecting = False

    def run(path: FormWritePath, name: str) -> None:
        calls.append((path, name))
        if rejecting:
            raise WriteRejected("Busy host")

    batches: list[dict[FormWritePath, JsonValue]] = []
    editor = build(state, env, recording_write(state, env, batches), run)
    before = state.input.data

    command(editor, "waveform", "to_inline").trigger()

    assert calls == [(("waveform",), "to_inline")]
    # The host owns the command: the editor does not run it on the state.
    assert state.input.data == before
    assert batches == []

    rejecting = True
    command(editor, "waveform", "to_inline").trigger()

    assert calls[-1] == (("waveform",), "to_inline")
    assert state.input.data == before
    assert field_error(editor, "waveform") == "Busy host"
    local = editor.local_parse_errors()
    assert isinstance(local, FormErrors)
    assert local.messages == {("waveform",): "Busy host"}
    editor.discard_local_text()
    assert editor.errors() is None


class FixedShapeForm(FormModel):
    """One fixed option value, with editable children and an explicit command."""

    shape: Flat

    @classmethod
    def options(cls, _env: FormEnv) -> Mapping[FormPath, tuple[Option, ...]]:
        return {
            ("shape",): (
                Option(
                    "Flat",
                    {"kind": "flat"},
                    fields=Flat,
                    commands=(
                        OptionCommand(
                            "lengthen",
                            "Lengthen",
                            lambda _old, _context: FormCommandResult(
                                {"kind": "flat"}, Flat, {"length": 8.0}
                            ),
                        ),
                    ),
                ),
            )
        }


def test_fixed_option_keeps_editable_children_and_guarded_commands(
    build: Build,
) -> None:
    env = FormEnv()
    state = new_state(FixedShapeForm, {"shape": {"kind": "flat", "length": 5.0}})
    calls: list[tuple[FormWritePath, str]] = []

    def run(path: FormWritePath, name: str) -> None:
        calls.append((path, name))
        # The host owns execution; this plain model accepts only leaf writes.
        state.write({("shape", "length"): 8.0}, env)

    editor = build(state, env, lambda batch: state.write(batch, env), run)
    length = child(editor, QLineEdit, "form.input.shape.length")
    assert not length.isReadOnly()
    action = command(editor, "shape", "lengthen")
    assert action.isEnabled()
    action.trigger()
    assert calls == [(("shape",), "lengthen")]
    assert state.input.data["shape"] == {"kind": "flat", "length": 8.0}
    assert child(editor, QLineEdit, "form.input.shape.length") is length
    editor.set_editing_enabled(enabled=False)
    assert not command(editor, "shape", "lengthen").isEnabled()
    assert length.isReadOnly()
    command(editor, "shape", "lengthen").trigger()
    assert calls == [(("shape",), "lengthen")]
    assert state.input.data["shape"] == {"kind": "flat", "length": 8.0}


def test_read_only_editor_keeps_commands_listed_but_disabled(build: Build) -> None:
    env = waveform_env()
    state = pulse_state(KEY, env)
    editor = build(state, env, None, None)
    before = state.input.data

    editor.set_editing_enabled(enabled=False)
    action = command(editor, "waveform", "to_inline")
    assert not action.isEnabled()
    action.trigger()
    assert state.input.data == before

    editor.set_editing_enabled(enabled=True)
    assert command(editor, "waveform", "to_inline").isEnabled()


def test_a_host_write_without_a_host_command_route_is_refused(
    qapp: QApplication,
) -> None:
    del qapp
    env = waveform_env()
    state = pulse_state(KEY, env)
    before = state.input.data
    notices: list[int] = []
    state.subscribe(lambda: notices.append(1))

    with pytest.raises(ValueError, match="run_command"):
        FormEditor(state, env, write=lambda batch: state.write(batch, env))

    state.write({("gain",): 0.5}, env)
    assert notices == [1]
    assert state.input.data == {**before, "gain": 0.5}


def test_generated_nested_copy_and_fresh_selection_via_component_events(
    build: Build, qapp: QApplication, tmp_path
) -> None:
    from tests.gui._user_input import ComponentDriver

    class EventCore(ConfigBase):
        pulse: PulseCfg | None = None
        note: str = "$literal-name"

    env = FormEnv(
        library=LibrarySnapshot(
            ModuleLibraryDocument(
                format="zcu.module-library",
                format_version="2.0",
                components={
                    "Q1": ComponentLibrary(
                        pulses={
                            "pi": PulseSpec(
                                ch=0,
                                nqz=1,
                                freq=100,
                                gain=0.2,
                                waveform="Q1.waveforms.wave",
                            )
                        },
                        waveforms={"wave": ConstWaveformCfg(length=0.1)},
                    )
                },
            )
        )
    )
    form = form_adapter.Form(EventCore, {})
    state = FormState(
        FormInput(
            form.model,
            {
                "pulse": {"kind": "ref", "name": "Q1.pulses.pi"},
                "note": "$literal-name",
            },
        )
    )
    editor = build(state, env, None, None)
    editor.resize(760, 1000)
    editor.show()
    qapp.processEvents()
    driver = ComponentDriver(editor)
    driver.screenshot(tmp_path / "01-ref.png")
    assert driver.observe(object_name="form.input.pulse").text == "Q1.pulses.pi"
    assert driver.observe(object_name="form.input.note").text == "$literal-name"
    driver.invoke_context_action(
        object_name="form.input.pulse", action="Convert to inline"
    )
    driver.screenshot(tmp_path / "02-outer-inline.png")
    assert driver.observe(object_name="form.input.pulse").text == "Inline"
    assert driver.observe(object_name="form.input.pulse.module.freq").text == "100.0"
    assert (
        driver.observe(object_name="form.input.pulse.module.waveform").text
        == "Q1.waveforms.wave"
    )
    driver.invoke_context_action(
        object_name="form.input.pulse.module.waveform", action="Convert to inline"
    )
    driver.screenshot(tmp_path / "03-child-inline.png")
    assert (
        driver.observe(object_name="form.input.pulse.module.waveform").text
        == "Inline: const"
    )
    assert (
        driver.observe(
            object_name="form.input.pulse.module.waveform.module.length"
        ).text
        == "0.1"
    )
    driver.enter_text(
        object_name="form.input.pulse.module.waveform.module.length", value="0.25"
    )
    driver.screenshot(tmp_path / "04-edited-length.png")
    assert (
        driver.observe(
            object_name="form.input.pulse.module.waveform.module.length"
        ).text
        == "0.25"
    )
    driver.choose_option(
        object_name="form.input.pulse.module.waveform", value="Inline: cosine"
    )
    driver.screenshot(tmp_path / "05-inherited-kind.png")
    assert (
        driver.observe(object_name="form.input.pulse.module.waveform").text
        == "Inline: cosine"
    )
    assert (
        driver.observe(
            object_name="form.input.pulse.module.waveform.module.length"
        ).text
        == "0.25"
    )
    driver.enter_text(object_name="form.input.note", value="$still-literal")
    driver.screenshot(tmp_path / "06-literal-text.png")
    assert driver.observe(object_name="form.input.note").text == "$still-literal"
