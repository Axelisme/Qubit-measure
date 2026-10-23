"""Core module fields accept family-limited references and persistent inline Specs."""

from typing import Annotated

import pytest
from pydantic import JsonValue, PydanticSchemaGenerationError
from zcu_tools.cfg_model import (
    ConfigBase,
    FieldMeta,
    MuxReadoutCfg,
    MuxReadoutSpec,
    PulseSpec,
    ResetCfg,
    Slot,
)
from zcu_tools.gui.form import (
    FormEnv,
    FormErrors,
    FormInput,
    FormInputRejected,
    FormModel,
    FormState,
)
from zcu_tools.gui.form_adapter import Form
from zcu_tools.program.v2.modules import PulseCfg
from zcu_tools.resources.context.library import (
    ComponentLibrary,
    LibrarySnapshot,
    ModuleLibraryDocument,
)


class ModuleSection(ConfigBase):
    """The core declares a specific pulse family, not GUI reference modes."""

    probe: Annotated[PulseCfg, FieldMeta(label="Probe", description="Drive the qubit.")]


class NullableModuleCore(ConfigBase):
    """Optional native modules retain ordinary enable/disable semantics."""

    probe: PulseCfg | None = None


class ModuleCore(ConfigBase):
    """A synthetic experiment cfg is independent of real experiment modules."""

    modules: ModuleSection


class ResetCore(ConfigBase):
    """A required multi-kind module and its nullable counterpart."""

    reset: ResetCfg
    optional: ResetCfg | None = None


@pytest.mark.parametrize("copy", [False, True], ids=["ref", "inline"])
def test_module_node_readonly_preserves_initial_source_and_subtree(copy: bool) -> None:
    class ReadonlyModule(ConfigBase):
        probe: Annotated[PulseCfg, FieldMeta(readonly=True)]

    form = Form(ReadonlyModule, {"probe": Slot("Q.pulses.probe", copy=copy)})
    env = module_env()
    state = FormState(form.default_input(env))
    before = state.input
    current = before.data["probe"]
    state.write({("probe",): current}, env)
    replacement: JsonValue = (
        {"kind": "ref", "name": "Q.pulses.probe"}
        if copy
        else {"kind": "inline", "module": {"type": "pulse"}}
    )
    with pytest.raises(FormInputRejected) as caught:
        state.write({("probe",): replacement}, env)
    assert caught.value.reason == "readonly"
    assert caught.value.current == current
    if copy:
        with pytest.raises(FormInputRejected):
            state.write({("probe", "module", "gain"): 0.9}, env)
    assert state.input.same_as(before)


@pytest.mark.parametrize("nullable", [False, True])
def test_generated_module_options_offer_typed_keys_and_single_inline(
    nullable: bool,
) -> None:
    form = Form(NullableModuleCore if nullable else ModuleSection, {})
    env = module_env()
    options = form.model.options(env).get(("probe",), ())
    assert [option.label for option in options] == (
        (["None"] if nullable else []) + ["Q.pulses.probe", "Inline"]
    )
    key = next(option for option in options if option.label == "Q.pulses.probe")
    assert key.value == {"kind": "ref", "name": "Q.pulses.probe"}
    assert key.fields is None
    inline = options[-1]
    assert inline.value == {"kind": "inline", "module": {"type": "pulse"}}
    state = FormState(form.default_input(env))
    state.select(("probe",), inline.value, env)
    children = state.describe(("probe",), env).children
    assert {field.path[-1] for field in children} >= {"ch", "freq", "gain", "waveform"}
    assert all(field.path[:2] == ("probe", "module") for field in children)


def test_generated_multi_kind_options_keep_editable_kinds_and_filter_keys() -> None:
    form = Form(ResetCore, {})
    env = module_env()
    expected = ["Inline: reset/pulse", "Inline: reset/two_pulse", "Inline: reset/bath"]
    options = form.model.options(env)
    assert [option.label for option in options.get(("reset",), ())] == expected
    assert [option.label for option in options.get(("optional",), ())] == [
        "None",
        *expected,
    ]
    assert [option.value for option in options[("reset",)]] == [
        {"kind": "inline", "module": {"type": kind}}
        for kind in ("reset/pulse", "reset/two_pulse", "reset/bath")
    ]
    state = FormState(form.default_input(env))
    expected_fields = [
        {"pulse_cfg"},
        {"pulse1_cfg", "pulse2_cfg"},
        {"cavity_tone_cfg", "qubit_tone_cfg", "pi2_cfg"},
    ]
    for option, names in zip(options[("reset",)], expected_fields, strict=True):
        state.select(("reset",), option.value, env)
        assert {
            field.path[-1] for field in state.describe(("reset",), env).children
        } == names


def test_module_key_command_copies_captured_content_in_one_write() -> None:
    form = Form(ModuleCore, {})
    env = stored_env()
    state = FormState(
        FormInput(
            form.model,
            {"modules": {"probe": {"kind": "ref", "name": "Q.pulses.probe"}}},
        )
    )
    target = state.describe(("modules", "probe"), env).target
    assert target is not None and target.options is not None
    assert [(item.name, item.label) for item in target.options[0].commands] == [
        ("to_inline", "Convert to inline")
    ]
    notifications: list[str] = []
    state.subscribe(lambda: notifications.append("changed"))

    state.run_command(("modules", "probe"), "to_inline", env)

    copied = expect_object(expect_object(state.input.data["modules"])["probe"])
    assert copied["kind"] == "inline"
    module = expect_object(copied["module"])
    assert (module["ch"], module["freq"], module["gain"]) == (1, 100.0, 0.3)
    assert module["waveform"] == {
        "kind": "inline",
        "module": {"style": "const", "length": 0.1},
    }
    assert notifications == ["changed"]
    observed = form.read_module(state.input, ("modules", "probe"), FormEnv())
    assert not isinstance(observed, FormErrors), observed
    assert observed.mode == "inline" and observed.value["freq"] == 100.0
    assert env.library is not None
    assert env.library.module_spec("Q.pulses.probe").desc == "stored"
    assert observed.value["desc"] is None


def test_generated_inline_selection_preserves_same_type_draft_and_copies_fixed_ref() -> (
    None
):
    from zcu_tools.cfg_model import Template, TemplateDef

    form = Form(ModuleSection, {"probe": Template("pulse")})
    form.bind_templates(
        {
            "pulse": TemplateDef(
                PulseSpec,
                "Pulse",
                {
                    "ch": 2,
                    "nqz": 1,
                    "freq": 90.0,
                    "gain": 0.2,
                    "waveform.style": "const",
                    "waveform.length": 0.4,
                },
            )
        },
        adapter="synthetic",
    )
    env = module_env()
    state = FormState(form.default_input(env))
    options = form.model.options(env).get(("probe",), ())
    assert [option.label for option in options] == ["Q.pulses.probe", "Inline"]
    inline = options[-1]
    assert inline.fields is not None and issubclass(inline.fields, FormModel)
    defaults = inline.fields.defaults(env)
    assert expect_object(defaults["module"])["freq"] == 90.0

    state.select(("probe",), inline.value, env)
    copied = expect_object(expect_object(state.input.data["probe"])["module"])
    assert (copied["freq"], copied["gain"]) == (90.0, 0.2)
    state.write({("probe", "module", "freq"): 250.0}, env)
    state.select(("probe",), options[0].value, env)
    state.select(("probe",), inline.value, env)
    assert (
        expect_object(expect_object(state.input.data["probe"])["module"])["freq"]
        == 100.0
    )


@pytest.mark.parametrize(
    "depth", [0, 1, 2], ids=["module", "waveform", "nested-waveform"]
)
def test_template_kind_reselection_at_each_depth_is_fresh(depth: int) -> None:
    from zcu_tools.cfg_model import (
        Expr,
        PulseResetSpec,
        Template,
        TemplateDef,
        TemplateSeed,
    )

    pulse_seeds: dict[str, TemplateSeed] = {
        "ch": 2,
        "nqz": 1,
        "freq": Expr("100 - 10"),
        "gain": 0.2,
        "waveform.style": "const",
        "waveform.length": Expr("2 * .2"),
    }
    if depth == 1:
        form = Form(ModuleSection, {"probe": Template("authored")})
        definition = TemplateDef(PulseSpec, "Pulse", pulse_seeds)
        root, root_label = "probe", "Inline"
    else:
        form = Form(ResetCore, {"reset": Template("authored")})
        definition = TemplateDef(
            PulseResetSpec,
            "Reset",
            {"pulse_cfg": "Q.pulses.probe"}
            if depth == 0
            else {f"pulse_cfg.{path}": value for path, value in pulse_seeds.items()},
        )
        root, root_label = "reset", "Inline: reset/pulse"
    form.bind_templates({"authored": definition}, adapter="synthetic")
    env = module_env() if depth == 0 else FormEnv()
    state = FormState(form.default_input(env))
    selection_path = (root,)
    if depth:
        if depth == 2:
            selection_path = (*selection_path, "module", "pulse_cfg")
        selection_path = (*selection_path, "module", "waveform")
    path = (*selection_path, "module")
    target = state.describe(selection_path, env).target
    assert target is not None and target.options is not None
    options = target.options

    def current() -> dict[str, JsonValue]:
        value = state.input.data
        for name in path:
            value = expect_object(value[name])
        return value

    def choose(label: str) -> None:
        option = next(item for item in options if item.label == label)
        state.select(selection_path, option.value, env)

    leaf = "length" if depth else "pulse_cfg"
    expected: JsonValue = 0.4 if depth else {"kind": "ref", "name": "Q.pulses.probe"}
    assert current()[leaf] == expected
    state.write({(*path, leaf): 1.7 if depth else {"kind": "inline"}}, env)
    choose("Inline: cosine" if depth else "Inline: reset/two_pulse")
    assert current() == (
        {"style": "cosine", "length": 1.7} if depth else {"type": "reset/two_pulse"}
    )
    # Another non-template kind also starts at type defaults, never the old draft.
    choose("Inline: gauss" if depth else "Inline: reset/bath")
    assert current() == (
        {"style": "gauss", "length": 1.7} if depth else {"type": "reset/bath"}
    )
    choose("Inline: const" if depth else root_label)
    assert current().get(leaf) == (1.7 if depth else expected)


@pytest.mark.parametrize("key", ["Q.pulses.gone", "missing.pulses.probe"])
def test_missing_module_key_reports_captured_library_and_full_key(key: str) -> None:
    form = Form(ModuleCore, {})
    env = module_env()
    state = FormState(
        FormInput(
            form.model,
            {"modules": {"probe": {"kind": "ref", "name": key}}},
        )
    )

    result = state.result(env)

    assert isinstance(result, FormErrors)
    message = result.messages[("modules", "probe", "name")]
    assert key in message
    assert "module" in message.lower()
    assert "captured library" in message.lower()


def module_env() -> FormEnv:
    """A fixed library contains a usable persistent pulse."""
    return FormEnv(
        library=LibrarySnapshot(
            ModuleLibraryDocument(
                format="zcu.module-library",
                format_version="2.0",
                components={
                    "Q": ComponentLibrary(
                        pulses={
                            "probe": PulseSpec.model_validate(
                                {
                                    "ch": 1,
                                    "nqz": 1,
                                    "freq": 100,
                                    "gain": 0.3,
                                    "waveform": {"style": "const", "length": 0.1},
                                }
                            )
                        }
                    )
                },
            )
        )
    )


@pytest.mark.parametrize("nullable", [False, True])
def test_generated_module_copy_and_nullable_result(nullable: bool) -> None:
    form = Form(NullableModuleCore if nullable else ModuleSection, {})
    env = module_env()
    state = FormState(
        FormInput(form.model, {"probe": {"kind": "ref", "name": "Q.pulses.probe"}})
    )
    assert not isinstance(state.result(env), FormErrors)
    state.run_command(("probe",), "to_inline", env)
    value = state.result(env)
    assert not isinstance(value, FormErrors), value
    cfg = form.to_cfg(value, env)
    assert isinstance(cfg, (ModuleSection, NullableModuleCore)), cfg
    assert cfg.probe is not None
    assert cfg.probe.freq == 100.0
    if nullable:
        state.write({("probe",): None}, env)
        assert state.input.data["probe"] is None
        assert not isinstance(state.result(env), FormErrors)


@pytest.mark.parametrize("mode", ["ref", "inline"])
def test_module_reference_or_inline_materializes_to_native_core(mode: str) -> None:
    try:
        form = Form(ModuleCore, {})
    except PydanticSchemaGenerationError as exc:
        pytest.fail(
            f"Form must derive editable module input instead of native QickParam fields: {exc}"
        )
    env = module_env()
    module: JsonValue = (
        {"kind": "ref", "name": "Q.pulses.probe"}
        if mode == "ref"
        else {
            "kind": "inline",
            "module": {
                "type": "pulse",
                "ch": 2,
                "nqz": 1,
                "freq": 200,
                "gain": 0.1,
                "waveform": {
                    "kind": "inline",
                    "module": {"style": "const", "length": 0.2},
                },
            },
        }
    )
    state = FormState(FormInput(form.model, {"modules": {"probe": module}}))
    value = state.result(env)
    assert not isinstance(value, FormErrors), value
    cfg = form.to_cfg(value, env)
    assert isinstance(cfg, ModuleCore), cfg
    assert cfg.modules.probe.ch == (1 if mode == "ref" else 2)
    assert cfg.modules.probe.freq == (100 if mode == "ref" else 200)
    assert cfg.modules.probe.waveform.length == (0.1 if mode == "ref" else 0.2)
    assert state.describe((), env).children


@pytest.mark.parametrize("mode", ["ref", "inline"])
def test_public_module_read_is_detached_and_field_local(mode: str) -> None:
    class InvalidDraft(ModuleCore):
        amount: float

    form = Form(InvalidDraft, {})
    env = module_env()
    raw: JsonValue = (
        {"kind": "ref", "name": "Q.pulses.probe"}
        if mode == "ref"
        else {
            "kind": "inline",
            "module": {
                "type": "pulse",
                "ch": 1,
                "nqz": 1,
                "gain": 0.3,
                "freq": {"expr": "50 * 2"},
                "waveform": {
                    "kind": "inline",
                    "module": {"style": "const", "length": 0.1},
                },
            },
        }
    )
    input = FormInput(form.model, {"modules": {"probe": raw}})
    before = input.data
    assert hasattr(form, "read_module"), "Missing public field-local Form module read"
    observed = form.read_module(input, ("modules", "probe"), env)
    assert not isinstance(observed, FormErrors), observed
    assert observed.value["freq"] == 100.0
    assert observed.mode == ("key" if mode == "ref" else "inline")
    assert observed.source == (
        "library:Q.pulses.probe" if mode == "ref" else "modules.probe"
    )
    observed.value["freq"] = -1.0
    again = form.read_module(input, ("modules", "probe"), env)
    assert not isinstance(again, FormErrors), again
    assert again.value["freq"] == 100.0
    assert input.data == before
    state = FormState(input)
    if mode == "ref":
        state.run_command(("modules", "probe"), "to_inline", env)
    modules = expect_object(state.input.data["modules"])
    assert expect_object(modules["probe"])["kind"] == "inline"
    assert input.data == before


def expect_object(value: JsonValue) -> dict[str, JsonValue]:
    """Narrow a public input observation without altering it."""
    assert isinstance(value, dict)
    return value


def stored_env() -> FormEnv:
    """A stored library entry sets fields the generated inline input cannot edit."""
    return FormEnv(
        library=LibrarySnapshot(
            ModuleLibraryDocument(
                format="zcu.module-library",
                format_version="2.0",
                components={
                    "Q": ComponentLibrary(
                        pulses={
                            "probe": PulseSpec.model_validate(
                                {
                                    "ch": 1,
                                    "nqz": 1,
                                    "freq": 100,
                                    "gain": 0.3,
                                    "desc": "stored",
                                    "waveform": {"style": "const", "length": 0.1},
                                }
                            )
                        }
                    )
                },
            )
        )
    )


@pytest.mark.parametrize("name", ["desc"])
def test_inline_module_rejects_unsupported_field_writes(name: str) -> None:
    form = Form(ModuleCore, {})
    env = module_env()
    state = FormState(
        FormInput(
            form.model,
            {"modules": {"probe": {"kind": "ref", "name": "Q.pulses.probe"}}},
        )
    )
    target = state.describe(("modules", "probe"), env).target
    assert target is not None and target.options is not None
    inline = next(option.value for option in target.options if option.label == "Inline")
    state.select(("modules", "probe"), inline, env)

    with pytest.raises(ValueError, match="Unknown.*field path"):
        state.write({("modules", "probe", "module", name): 1}, env)


@pytest.mark.parametrize("path", ["command", "copy"])
def test_inline_copy_of_library_entry_replaces_unsupported_values_with_defaults(
    path: str,
) -> None:
    from zcu_tools.cfg_model import Slot

    env = stored_env()
    if path == "command":
        form = Form(ModuleCore, {})
        state = FormState(
            FormInput(
                form.model,
                {"modules": {"probe": {"kind": "ref", "name": "Q.pulses.probe"}}},
            )
        )
        state.run_command(("modules", "probe"), "to_inline", env)
    else:
        form = Form(
            ModuleCore, {"modules": {"probe": Slot("Q.pulses.probe", copy=True)}}
        )
        state = FormState(form.default_input(env))

    value = state.result(env)
    assert not isinstance(value, FormErrors), value
    cfg = form.to_cfg(value, env)
    assert isinstance(cfg, ModuleCore), cfg
    probe = cfg.modules.probe
    assert (probe.freq, probe.gain) == (100, 0.3)
    assert probe.desc is None


@pytest.mark.parametrize("missing", ["library", "key"])
def test_required_module_branch_switch_uses_blank_defaults(missing: str) -> None:
    from zcu_tools.cfg_model import Slot, TemplateDef

    form = Form(
        ModuleCore,
        {
            "modules": {
                "probe": Slot(
                    "$qubit.module.probe",
                )
            }
        },
    )
    form.bind_templates(
        {
            "pulse": TemplateDef(
                PulseSpec,
                "Pulse",
                {
                    "ch": 1,
                    "nqz": 1,
                    "freq": 90.0,
                    "gain": 0.2,
                    "waveform.style": "const",
                    "waveform.length": 0.1,
                },
            )
        },
        adapter="synthetic",
    )
    env = FormEnv() if missing == "library" else module_env()
    # FormInput can preserve an invalid draft; ordinary write rejects missing refs.
    input = FormInput(
        form.model, {"modules": {"probe": {"kind": "ref", "name": "Q.pulses.missing"}}}
    )
    before = input.data
    state = FormState(input)
    notifications: list[str] = []
    state.subscribe(lambda: notifications.append("changed"))
    target = state.describe(("modules", "probe"), env).target
    assert target is not None and target.options is not None
    inline = next(option.value for option in target.options if option.label == "Inline")
    with pytest.raises(ValueError, match="library|Missing|missing|Unknown"):
        state.select(("modules", "probe"), inline, env)
    assert notifications == []
    assert state.input.data == input.data == before


class MuxCore(ConfigBase):
    """A synthetic core with one mux readout, independent of lab definitions."""

    readout: MuxReadoutCfg


def test_mux_ref_to_inline_keeps_labels_and_detaches_json_edits() -> None:
    source = MuxReadoutSpec.model_validate(
        {
            "gen_ch": 9,
            "nqz": 2,
            "mixer_freq": 6000.0,
            "length": 2.0,
            "ro_length": 1.5,
            "tones": [{"freq": 6020.0, "gain": 0.1, "ro_ch": 7}],
        }
    )
    env = FormEnv(
        library=LibrarySnapshot(
            ModuleLibraryDocument(
                format="zcu.module-library",
                format_version="2.0",
                components={"MG": ComponentLibrary(readout={"rf": source})},
            )
        )
    )
    form = Form(MuxCore, {})
    state = FormState(
        FormInput(form.model, {"readout": {"kind": "ref", "name": "MG.readout.rf"}})
    )
    ref = form.read_module(state.input, ("readout",), env)
    assert not isinstance(ref, FormErrors), ref
    assert ref.value["tones"] == [
        {"freq": 6020.0, "gain": 0.1, "phase": 0.0, "ro_ch": 7, "label": ""}
    ]
    state.run_command(("readout",), "to_inline", env)
    inline = form.read_module(state.input, ("readout",), FormEnv())
    assert not isinstance(inline, FormErrors), inline
    assert inline.mode == "inline"
    assert inline.value["tones"] == ref.value["tones"]
    state.write(
        {
            ("readout", "module", "tones"): [
                {"freq": 6030.0, "gain": 0.1, "ro_ch": 7, "label": "Edited"}
            ]
        },
        FormEnv(),
    )
    value = state.result(FormEnv())
    assert not isinstance(value, FormErrors), value
    cfg = form.to_cfg(value, FormEnv())
    assert isinstance(cfg, MuxCore), cfg
    assert [(tone.freq, tone.label) for tone in cfg.readout.tones] == [
        (6030.0, "Edited")
    ]
    assert env.library is not None
    assert env.library.module_spec("MG.readout.rf") == source
