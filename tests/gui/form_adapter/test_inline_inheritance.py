"""Approved inline inheritance is shared by generated and handwritten forms."""

from __future__ import annotations

from collections.abc import Mapping
from itertools import permutations

import pytest
from pydantic import JsonValue, create_model
from zcu_tools.cfg_model import (
    ConstWaveformCfg,
    GaussWaveformCfg,
    PulseSpec,
)
from zcu_tools.gui import form as api
from zcu_tools.gui.form import (
    FormEnv,
    FormErrors,
    FormInput,
    FormModel,
    FormPath,
    FormState,
    Option,
)
from zcu_tools.gui.form_adapter import ModuleInputField, module_form
from zcu_tools.resources.context.library import (
    ComponentLibrary,
    LibrarySnapshot,
    ModuleLibraryDocument,
)


def obj(value: JsonValue) -> dict[str, JsonValue]:
    assert isinstance(value, dict)
    return value


def choose(state: FormState[FormModel], path: tuple[str, ...], label: str, env: FormEnv) -> None:
    target = state.describe(path, env).target
    assert target is not None and target.options is not None
    state.write(
        {
            path: next(
                option.value for option in target.options if option.label == label
            )
        },
        env,
    )


@pytest.mark.parametrize(
    "source,target", list(permutations(("const", "cosine", "gauss", "drag"), 2))
)
def test_explicit_waveform_pairs_preserve_raw_length_and_only_approved_sigma(
    source, target
):
    form = module_form(PulseSpec)
    env = FormEnv()
    draft: dict[str, JsonValue] = {"style": source, "length": {"expr": "4 / 5"}}
    if source in ("gauss", "drag"):
        draft["sigma"] = {"expr": "1 / 5"}
    if source == "drag":
        draft.update(delta=33, alpha=0.8)
    state = FormState(
        FormInput(form.model, {"waveform": {"kind": "inline", "module": draft}})
    )
    choose(state, ("waveform",), f"Inline: {target}", env)
    expected: dict[str, JsonValue] = {"style": target, "length": {"expr": "4 / 5"}}
    if {source, target} == {"gauss", "drag"}:
        expected["sigma"] = {"expr": "1 / 5"}
    assert state.input.data["waveform"] == {"kind": "inline", "module": expected}


def test_ref_inline_copy_is_fixed_and_arb_flat_top_cross_pairs_do_not_inherit():
    form = module_form(PulseSpec)
    env = FormEnv(
        library=LibrarySnapshot(
            ModuleLibraryDocument(
                format="zcu.module-library",
                format_version="2.0",
                components={
                    "Q": ComponentLibrary(
                        waveforms={"shape": GaussWaveformCfg(length=0.8, sigma=0.2)}
                    )
                },
            )
        )
    )
    state = FormState(
        FormInput(
            form.model, {"waveform": {"kind": "ref", "name": "Q.waveforms.shape"}}
        )
    )
    choose(state, ("waveform",), "Inline: gauss", env)
    assert state.input.data["waveform"] == {
        "kind": "inline",
        "module": {"style": "gauss", "length": 0.8, "sigma": 0.2},
    }
    choose(state, ("waveform",), "Inline: flat_top", env)
    assert state.input.data["waveform"] == {
        "kind": "inline",
        "module": {"style": "flat_top"},
    }
    state.write({("waveform", "module", "length"): {"expr": "1 / 0"}}, env)
    before = state.input.data
    choose(state, ("waveform",), "Inline: flat_top", env)
    assert state.input.data == before, "Same-type invalid/missing input must survive"
    choose(state, ("waveform",), "Inline: arb", env)
    assert state.input.data["waveform"] == {
        "kind": "inline",
        "module": {"style": "arb"},
    }
    state.write({("waveform", "module", "data"): "draft-asset"}, env)
    choose(state, ("waveform",), "Inline: arb", env)
    assert obj(obj(state.input.data["waveform"])["module"])["data"] == "draft-asset"
    choose(state, ("waveform",), "Inline: const", env)
    assert state.input.data["waveform"] == {
        "kind": "inline",
        "module": {"style": "const"},
    }


def test_two_fixed_captures_do_not_share_converter_sources():
    form = module_form(PulseSpec)
    first_env = FormEnv(
        library=LibrarySnapshot(
            ModuleLibraryDocument(
                format="zcu.module-library",
                format_version="2.0",
                components={
                    "Q": ComponentLibrary(
                        waveforms={"shape": GaussWaveformCfg(length=0.8, sigma=0.2)}
                    )
                },
            )
        )
    )
    assert first_env.library is not None
    second_env = FormEnv(
        library=first_env.library.replaced_waveform(
            "Q.waveforms.shape", GaussWaveformCfg(length=1.6, sigma=0.4)
        )
    )
    raw = {"waveform": {"kind": "ref", "name": "Q.waveforms.shape"}}
    first = FormState(FormInput(form.model, raw))
    second = FormState(FormInput(form.model, raw))
    choose(first, ("waveform",), "Inline: gauss", first_env)
    choose(second, ("waveform",), "Inline: gauss", second_env)
    assert first.input.data["waveform"] == {
        "kind": "inline",
        "module": {"style": "gauss", "length": 0.8, "sigma": 0.2},
    }
    assert second.input.data["waveform"] == {
        "kind": "inline",
        "module": {"style": "gauss", "length": 1.6, "sigma": 0.4},
    }
    first.describe(("waveform",), second_env)
    first.preview(second_env)
    assert first.input.data["waveform"] == {
        "kind": "inline",
        "module": {"style": "gauss", "length": 0.8, "sigma": 0.2},
    }


def test_handwritten_module_field_uses_same_converter_and_preserves_invalid_drafts():
    field = ModuleInputField(PulseSpec)

    class Handwritten(FormModel):
        @classmethod
        def options(cls, env: FormEnv) -> Mapping[FormPath, tuple[Option, ...]]:
            return {("pulse",): field.options(env)}

        @classmethod
        def converters(cls, env: FormEnv) -> Mapping[FormPath, api.InputConverter]:
            return {("pulse",): field.converter(env)}

    model = create_model(
        "HandwrittenPulse", __base__=Handwritten, pulse=(field.annotation, ...)
    )
    env = FormEnv(
        library=LibrarySnapshot(
            ModuleLibraryDocument(
                format="zcu.module-library",
                format_version="2.0",
                components={
                    "Q": ComponentLibrary(
                        pulses={
                            "probe": PulseSpec(
                                ch=1,
                                nqz=1,
                                freq=100,
                                gain=0.3,
                                waveform=ConstWaveformCfg(length=0.8),
                            )
                        }
                    )
                },
            )
        )
    )
    state = FormState(
        FormInput(model, {"pulse": {"kind": "ref", "name": "Q.pulses.probe"}})
    )
    choose(state, ("pulse",), "Inline", env)
    assert obj(obj(state.input.data["pulse"])["module"])["freq"] == 100
    state.write({("pulse", "module", "freq"): {"expr": "1 / 0"}}, env)
    before = state.input.data
    choose(state, ("pulse",), "Inline", env)
    assert state.input.data == before
    assert isinstance(state.result(env), FormErrors)
