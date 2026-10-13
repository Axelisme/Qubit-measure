"""Untagged str-or-model fields: the JSON type selects a key or an inline model."""

from typing import Annotated, Literal

import pytest
from jsonschema import Draft202012Validator
from pydantic import BaseModel, ConfigDict, Field, JsonValue
from zcu_tools.cfg_model import (
    ConstWaveformCfg,
    DirectReadoutCfg,
    PulseSpec,
    WaveformSpec,
)
from zcu_tools.gui.form import FormEnv, FormErrors, FormInput, FormModel, FormState
from zcu_tools.resources.context.library import (
    ComponentLibrary,
    LibrarySnapshot,
    ModuleLibraryDocument,
)


class Tone(BaseModel):
    """The single inline model of a str-or-model field."""

    model_config = ConfigDict(extra="forbid")

    ch: int
    gain: float = 0.5


class FlatShape(BaseModel):
    """First tagged branch; its tag has no default, like a required Literal."""

    model_config = ConfigDict(extra="forbid")

    style: Literal["flat"]
    length: float = 1.0


class RampShape(BaseModel):
    """Second tagged branch with a required branch-only field."""

    model_config = ConfigDict(extra="forbid")

    style: Literal["ramp"] = "ramp"
    length: float = 1.0
    sigma: float


type Shape = Annotated[FlatShape | RampShape, Field(discriminator="style")]


class KeyOrInlineForm(FormModel):
    """A key-or-single-model field and a key-or-tagged-union field."""

    tone: str | Tone
    shape: str | Shape


def test_nested_expression_in_tagged_inline_branch_decodes_in_result() -> None:
    env = FormEnv()
    state = FormState(
        FormInput(
            KeyOrInlineForm,
            {
                "tone": "Q1.pulses.pi",
                "shape": {
                    "style": "ramp",
                    "length": {"expr": "3 / 2"},
                    "sigma": {"expr": "0.1 * 2"},
                },
            },
        )
    )

    result = state.result(env)

    assert isinstance(result, KeyOrInlineForm)
    assert result.tone == "Q1.pulses.pi"
    assert isinstance(result.shape, RampShape)
    assert result.shape.length == 1.5
    assert result.shape.sigma == pytest.approx(0.2)
    assert state.preview(env).values[("shape", "sigma")] == pytest.approx(0.2)
    assert state.input.data["shape"] == {
        "style": "ramp",
        "length": {"expr": "3 / 2"},
        "sigma": {"expr": "0.1 * 2"},
    }


def test_inline_errors_point_at_branch_fields_without_the_key_alternative() -> None:
    state = FormState(
        FormInput(KeyOrInlineForm, {"tone": {"gain": 0.1}, "shape": {"style": "ramp"}})
    )

    errors = state.result(FormEnv())

    assert isinstance(errors, FormErrors)
    assert set(errors.messages) == {("tone", "ch"), ("shape", "sigma")}


def test_object_write_switches_to_inline_defaults_and_string_write_to_a_key() -> None:
    env = FormEnv()
    state = FormState(
        FormInput(KeyOrInlineForm, {"tone": "Q1.pulses.pi", "shape": "Q1.waveforms.x"})
    )

    state.write({("tone",): {}, ("shape",): {}}, env)

    assert state.input.data == {
        "tone": {"gain": 0.5},
        "shape": {"style": "flat", "length": 1.0},
    }
    state.write({("shape", "style"): "ramp", ("shape", "sigma"): 0.2}, env)
    state.write({("tone", "ch"): 3}, env)
    inline = {
        "tone": {"gain": 0.5, "ch": 3},
        "shape": {"style": "ramp", "length": 1.0, "sigma": 0.2},
    }
    assert state.input.data == inline
    result = state.result(env)
    assert isinstance(result, KeyOrInlineForm)
    assert result.tone == Tone(ch=3)
    assert result.shape == RampShape(sigma=0.2)

    state.write({("tone",): {}}, env)
    assert state.input.data == inline
    with pytest.raises(ValueError, match="Whole model replacement"):
        state.write({("tone",): {"ch": 4}}, env)
    assert state.input.data == inline

    state.write({("tone",): "", ("shape",): ""}, env)
    assert state.input.data == {"tone": "", "shape": ""}


def test_compatible_restore_keeps_saved_keys_and_inline_objects() -> None:
    env = FormEnv()
    state = FormState(
        FormInput(KeyOrInlineForm, {"tone": {"ch": 0}, "shape": "Q1.waveforms.x"})
    )

    dropped = state.write_compatible(
        {"tone": "Q1.pulses.pi", "shape": {"style": "ramp", "sigma": 0.2}}, env
    )

    assert dropped == ()
    assert state.input.data == {
        "tone": "Q1.pulses.pi",
        "shape": {"style": "ramp", "length": 1.0, "sigma": 0.2},
    }
    dropped = state.write_compatible(
        {"tone": {"ch": 3, "gain": 0.2}, "shape": "Q1.waveforms.y"}, env
    )
    assert dropped == ()
    assert state.input.data == {
        "tone": {"ch": 3, "gain": 0.2},
        "shape": "Q1.waveforms.y",
    }


@pytest.mark.parametrize(
    ("data", "valid"),
    [
        ({"tone": "Q1.pulses.pi", "shape": "Q1.waveforms.x"}, True),
        (
            {"tone": {"ch": {"expr": "1 + 1"}}, "shape": {"style": "ramp", "sigma": 1}},
            True,
        ),
        ({"tone": {"expr": "'Q1.pulses.pi'"}}, False),
    ],
)
def test_schema_offers_a_plain_key_or_the_inline_model(
    data: dict[str, JsonValue], *, valid: bool
) -> None:
    validator = Draft202012Validator(KeyOrInlineForm.input_schema(FormEnv()))

    assert validator.is_valid(data) is valid


class Holder(BaseModel):
    """An ordinary nested model whose own field is str-or-model."""

    model_config = ConfigDict(extra="forbid")

    inner: str | Tone = "Q1.pulses.pi"


class MarkerForm(FormModel):
    """Str-or-model fields beside string, nullable and model fields."""

    tone: str | Tone
    shape: str | Shape
    optional: str | Tone | None = None
    holder: Holder = Field(default_factory=Holder)
    name: str = "x"
    maybe: str | None = None
    plain: Tone = Field(default_factory=lambda: Tone(ch=0))
    either: Shape = Field(default_factory=lambda: FlatShape(style="flat"))


def _marked(node: JsonValue, path: tuple[str, ...] = ()) -> set[tuple[str, ...]]:
    """Return the property paths of every schema node marked key-or-inline.

    Defaults are data, not schema, so they are not searched.
    """
    found: set[tuple[str, ...]] = set()
    if isinstance(node, list):
        for item in node:
            found |= _marked(item, path)
    if not isinstance(node, dict):
        return found
    marker = node.get("x-form-key-or-inline")
    if marker is not None:
        assert marker is True, path
        found.add(path)
    for key, child in node.items():
        if key == "properties" and isinstance(child, dict):
            for name, prop in child.items():
                found |= _marked(prop, (*path, name))
        elif key != "default":
            found |= _marked(child, path)
    return found


def test_schema_marks_exactly_the_str_or_model_fields() -> None:
    schema = MarkerForm.input_schema(FormEnv())

    assert _marked(schema) == {
        ("tone",),
        ("shape",),
        ("optional",),
        ("holder", "inner"),
    }


class SpecKeyForm(FormModel):
    """Library-backed fields: a pulse key and a waveform key."""

    drive: str | PulseSpec
    envelope: str | WaveformSpec


def _key_branch(schema: dict[str, JsonValue], name: str) -> dict[str, JsonValue]:
    properties = schema["properties"]
    assert isinstance(properties, dict)
    field = properties[name]
    assert isinstance(field, dict)
    variants = field["anyOf"]
    assert isinstance(variants, list)
    return next(
        variant
        for variant in variants
        if isinstance(variant, dict) and variant.get("type") == "string"
    )


def test_key_branch_lists_captured_library_keys_of_matching_specs() -> None:
    pulse = PulseSpec(ch=0, nqz=1, freq=100.0, gain=0.2, waveform="Q1.waveforms.flat")
    library = LibrarySnapshot(
        ModuleLibraryDocument(
            format="zcu.module-library",
            format_version="1.0",
            components={
                "Q1": ComponentLibrary(
                    pulses={"pi": pulse},
                    readout={
                        "direct": DirectReadoutCfg(
                            ro_ch=0, ro_length=1.0, ro_freq=100.0
                        )
                    },
                    waveforms={"flat": ConstWaveformCfg(length=1.0)},
                )
            },
        )
    )

    captured = SpecKeyForm.input_schema(FormEnv(library=library))
    source_free = SpecKeyForm.input_schema(FormEnv())

    assert _key_branch(captured, "drive")["x-form-library-keys"] == ["Q1.pulses.pi"]
    assert _key_branch(captured, "envelope")["x-form-library-keys"] == [
        "Q1.waveforms.flat"
    ]
    assert "x-form-library-keys" not in _key_branch(source_free, "drive")
