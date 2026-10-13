"""Core module fields accept family-limited references and persistent inline Specs."""

from typing import Annotated

import pytest
from pydantic import JsonValue, PydanticSchemaGenerationError
from zcu_tools.cfg_model import ConfigBase, FieldMeta, PulseSpec
from zcu_tools.gui.app.measure.adapter import Form
from zcu_tools.gui.form import FormEnv, FormErrors, FormInput, FormState
from zcu_tools.program.v2.modules import PulseCfg
from zcu_tools.resources.context.library import (
    ComponentLibrary,
    LibrarySnapshot,
    ModuleLibraryDocument,
)


class ModuleSection(ConfigBase):
    """The core declares a specific pulse family, not GUI reference modes."""

    probe: Annotated[PulseCfg, FieldMeta(label="Probe", description="Drive the qubit.")]


class ModuleCore(ConfigBase):
    """A synthetic experiment cfg is independent of real experiment modules."""

    modules: ModuleSection


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
