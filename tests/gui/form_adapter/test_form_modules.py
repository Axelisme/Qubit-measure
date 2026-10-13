"""Core module fields accept family-limited references and persistent inline Specs."""

from typing import Annotated

import pytest
from pydantic import JsonValue, PydanticSchemaGenerationError
from zcu_tools.cfg_model import ConfigBase, FieldMeta, PulseSpec, ResetCfg
from zcu_tools.gui.form import FormEnv, FormErrors, FormInput, FormState
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


@pytest.mark.parametrize("nullable", [False, True])
def test_generated_module_options_offer_typed_keys_and_single_inline(nullable: bool) -> None:
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
    properties = expect_object(form.model.input_schema(env)["properties"])
    projected = expect_object(properties["probe"])["x-form-options"]
    assert isinstance(projected, list)
    fields = expect_object(expect_object(projected[-1])["fields"])
    module = expect_object(expect_object(fields["properties"])["module"])
    assert set(expect_object(module["properties"])) >= {"ch", "freq", "gain", "waveform"}
    assert "type" not in expect_object(module["properties"])
    assert "kind" not in expect_object(fields["properties"])


def test_generated_multi_kind_options_keep_editable_kinds_and_filter_keys() -> None:
    form = Form(ResetCore, {})
    env = module_env()
    expected = ["Inline: reset/pulse", "Inline: reset/two_pulse", "Inline: reset/bath"]
    options = form.model.options(env)
    assert [option.label for option in options.get(("reset",), ())] == expected
    assert [option.label for option in options.get(("optional",), ())] == ["None", *expected]
    assert [option.value for option in options[("reset",)]] == [
        {"kind": "inline", "module": {"type": kind}}
        for kind in ("reset/pulse", "reset/two_pulse", "reset/bath")
    ]
    schema = form.model.input_schema(env)
    projected = expect_object(expect_object(schema["properties"])["reset"])["x-form-options"]
    assert isinstance(projected, list)
    expected_fields = [{"pulse_cfg"}, {"pulse1_cfg", "pulse2_cfg"}, {"cavity_tone_cfg", "qubit_tone_cfg", "pi2_cfg"}]
    for item, names in zip(projected, expected_fields, strict=True):
        fields = expect_object(expect_object(item)["fields"])
        module = expect_object(expect_object(fields["properties"])["module"])
        assert set(expect_object(module["properties"])) == names


def module_env() -> FormEnv:
    """A fixed library contains a usable persistent pulse."""
    return FormEnv(
        library=LibrarySnapshot(
            ModuleLibraryDocument(
                format="zcu.module-library",
                format_version="1.0",
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
def test_generated_module_schema_merges_key_and_inline(nullable: bool) -> None:
    form = Form(NullableModuleCore if nullable else ModuleSection, {})
    env = module_env()
    schema = form.model.input_schema(env)
    properties = schema["properties"]
    assert isinstance(properties, dict)
    probe = properties["probe"]
    assert isinstance(probe, dict)
    if nullable:
        variants = probe["anyOf"]
        assert isinstance(variants, list)
        assert {"type": "null"} in variants
        probe = next(v for v in variants if isinstance(v, dict) and "oneOf" in v)
    assert probe.get("x-form-key-or-inline") is True
    branches = probe["oneOf"]
    assert isinstance(branches, list)
    ref = next(
        b
        for b in branches
        if isinstance(b, dict) and b["x-form-discriminator-values"] == ["ref"]
    )
    props = ref["properties"]
    assert isinstance(props, dict)
    name = props["name"]
    assert isinstance(name, dict)
    assert name["x-form-library-keys"] == ["Q.pulses.probe"]
    state = FormState(
        FormInput(form.model, {"probe": {"kind": "ref", "name": "Q.pulses.probe"}})
    )
    assert not isinstance(state.result(env), FormErrors)
    state.write(form.module_inline_writes(state.input, ("probe",), env), env)
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
                "waveform": {"style": "const", "length": 0.2},
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
    assert form.model.input_schema(env)


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
                "waveform": {"style": "const", "length": 0.1},
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
    inline = form.module_inline_input(input, ("modules", "probe"), env)
    assert inline["kind"] == "inline"
    assert input.data == before


@pytest.mark.parametrize("fallback", [True, False], ids=["template", "blank"])
@pytest.mark.parametrize("missing", ["library", "key"])
def test_required_module_branch_switch_uses_owner_fallback(
    fallback: bool, missing: str
) -> None:
    from zcu_tools.cfg_model import Slot, Template, TemplateDef

    form = Form(
        ModuleCore,
        {
            "modules": {
                "probe": Slot(
                    "$qubit.module.probe",
                    fallback=Template("pulse") if fallback else None,
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
    state.write({("modules", "probe", "kind"): "inline"}, env)
    assert notifications == ["changed"]
    assert input.data == before
    module = expect_object(expect_object(state.input.data["modules"])["probe"])
    assert module["kind"] == "inline"
    assert expect_object(module["module"])["type"] == "pulse"
    if fallback:
        observed = form.read_module(state.input, ("modules", "probe"), env)
        assert not isinstance(observed, FormErrors), observed
        assert observed.value["freq"] == 90.0
    else:
        state.write({("modules", "probe", "module", "freq"): 125.0}, env)
        assert (
            state.preview(env).values[("modules", "probe", "module", "freq")] == 125.0
        )


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
                format_version="1.0",
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
                                    "mask": [0],
                                    "outsel": 1,
                                    "ro_ch": 4,
                                    "waveform": {"style": "const", "length": 0.1},
                                }
                            )
                        }
                    )
                },
            )
        )
    )


@pytest.mark.parametrize(
    "name", ["desc", "mux_freqs", "mux_gains", "mux_phases", "mask", "outsel", "ro_ch"]
)
def test_inline_module_rejects_unsupported_field_writes(name: str) -> None:
    form = Form(ModuleCore, {})
    env = module_env()
    state = FormState(
        FormInput(
            form.model,
            {"modules": {"probe": {"kind": "ref", "name": "Q.pulses.probe"}}},
        )
    )
    state.write({("modules", "probe", "kind"): "inline"}, env)

    with pytest.raises(ValueError, match="Unknown field path"):
        state.write({("modules", "probe", "module", name): 1}, env)


@pytest.mark.parametrize("path", ["switch", "copy"])
def test_inline_copy_of_library_entry_replaces_unsupported_values_with_defaults(
    path: str,
) -> None:
    from zcu_tools.cfg_model import Slot

    env = stored_env()
    if path == "switch":
        form = Form(ModuleCore, {})
        state = FormState(
            FormInput(
                form.model,
                {"modules": {"probe": {"kind": "ref", "name": "Q.pulses.probe"}}},
            )
        )
        state.write({("modules", "probe", "kind"): "inline"}, env)
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
    assert (probe.desc, probe.mask, probe.outsel, probe.ro_ch) == (None,) * 4


def test_locked_module_field_with_spec_decimals_has_input_schema() -> None:
    from zcu_tools.cfg_model import Slot

    form = Form(
        ModuleCore,
        {"modules": {"probe": Slot("Q.pulses.probe", lock={"freq": 100.0})}},
    )
    env = module_env()

    state = FormState(form.default_input(env))
    value = state.result(env)

    assert isinstance(form.model.input_schema(env)["properties"], dict)
    assert not isinstance(value, FormErrors), value
    cfg = form.to_cfg(value, env)
    assert isinstance(cfg, ModuleCore) and cfg.modules.probe.freq == 100.0
