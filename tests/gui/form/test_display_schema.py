"""Presentation hints in the input schema: decimals, hidden, titles and fixed tags."""

import json
from typing import Annotated, Literal

import pytest
from pydantic import BaseModel, ConfigDict, Field, JsonValue
from zcu_tools.cfg_model import DirectReadoutCfg
from zcu_tools.gui.form import (
    Display,
    FormEnv,
    FormErrors,
    FormModel,
    FormState,
    LinRange,
    ModuleInput,
    RangeInput,
)


def node(schema: JsonValue, *names: str) -> dict[str, JsonValue]:
    """Walk nested schema properties by field name; each step is an object.

    A nullable model keeps its object schema as one anyOf alternative.
    """
    for name in names:
        assert isinstance(schema, dict), name
        variants = schema.get("anyOf")
        objects = [
            variant
            for variant in (variants if isinstance(variants, list) else [schema])
            if isinstance(variant, dict) and "properties" in variant
        ]
        assert len(objects) == 1, name
        properties = objects[0]["properties"]
        assert isinstance(properties, dict), name
        schema = properties.get(name)
    assert isinstance(schema, dict), names
    return schema


def branch(field: dict[str, JsonValue], tag: JsonValue) -> dict[str, JsonValue]:
    """Return the tagged oneOf branch of field whose intrinsic tags include tag."""
    branches = field.get("oneOf")
    assert isinstance(branches, list)
    for candidate in branches:
        assert isinstance(candidate, dict)
        tags = candidate.get("x-form-discriminator-values")
        assert isinstance(tags, list)
        if tag in tags:
            return candidate
    raise AssertionError(f"No branch tagged {tag!r}")


class Section(BaseModel):
    """An ordinary nested model whose class name is not a field title."""

    model_config = ConfigDict(extra="forbid")

    level: float = 1.0


class Alpha(BaseModel):
    """First branch of a form-local tagged union."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["alpha"] = "alpha"
    x: float = 1.0


class Beta(BaseModel):
    """Second branch of the same union."""

    model_config = ConfigDict(extra="forbid")

    kind: Literal["beta"] = "beta"


class LockedAlpha(Alpha):
    """A branch subclass that also fixes an ordinary leaf to one value."""

    count: Literal[2] = 2


def test_display_decimals_reach_the_schema_only_where_declared() -> None:
    class DecimalsForm(FormModel):
        delay: Annotated[float, Display(unit="us", decimals=3)] = 1.0
        optional: Annotated[float | None, Display(decimals=0)] = None
        plain: Annotated[float, Display(unit="us")] = 1.0

    schema = DecimalsForm.input_schema(FormEnv())
    assert node(schema, "delay")["decimals"] == 3
    assert node(schema, "optional")["decimals"] == 0
    assert "decimals" not in node(schema, "plain")
    with pytest.raises(ValueError, match="decimals"):
        Display(decimals=-1)

    class IntegerDecimalsForm(FormModel):
        count: Annotated[int, Display(decimals=2)] = 1

    with pytest.raises(ValueError, match="decimals"):
        IntegerDecimalsForm.input_schema(FormEnv())


def test_hidden_field_stays_in_schema_input_and_validation() -> None:
    class HiddenForm(FormModel):
        shown: float = 1.0
        gain: Annotated[float, Display(hidden=True)] = Field(default=0.5, gt=0)

    env = FormEnv()
    schema = HiddenForm.input_schema(env)
    assert node(schema, "gain")["hidden"] is True
    assert "hidden" not in node(schema, "shown")
    state = FormState(HiddenForm.default_input(env))
    state.write({("shown",): 2.0}, env)
    assert state.input.data == {"shown": 2.0, "gain": 0.5}
    state.write({("gain",): -1.0}, env)
    errors = state.result(env)
    assert isinstance(errors, FormErrors)
    assert set(errors.messages) == {("gain",)}


def test_model_sections_without_a_label_are_titled_by_field_name() -> None:
    class TitledForm(FormModel):
        ro_cfg: Section = Field(default_factory=Section)
        labelled: Annotated[Section, Display(label="Readout")] = Field(
            default_factory=Section
        )
        optional_cfg: Section | None = None
        pulse_cfg: str | Section = "Q1.pulses.pi"
        drive: ModuleInput

    schema = TitledForm.input_schema(FormEnv())
    assert node(schema, "ro_cfg")["title"] == "Ro cfg"
    assert node(schema, "labelled")["title"] == "Readout"
    assert node(schema, "optional_cfg")["title"] == "Optional cfg"
    assert node(schema, "pulse_cfg")["title"] == "Pulse cfg"
    drive = node(schema, "drive")
    assert drive["title"] == "Drive"
    assert node(branch(drive, "inline"), "module")["title"] == "Module"


def test_one_value_tags_outside_a_union_are_hidden_and_union_selectors_stay() -> None:
    class TagForm(FormModel):
        choice: Annotated[Alpha | Beta, Field(discriminator="kind")] = Field(
            default_factory=Alpha
        )
        alone: Alpha = Field(default_factory=Alpha)
        locked: LockedAlpha = Field(default_factory=LockedAlpha)
        sweep: LinRange = Field(default_factory=LinRange)
        fluxes: RangeInput = Field(default_factory=LinRange)
        readout: DirectReadoutCfg | None = None
        fixed: Literal["fixed"] = "fixed"

    schema = TagForm.input_schema(FormEnv())
    # Models that a tagged union elsewhere selects by tag: this form's own
    # union, gui.form's RangeInput, and the cfg_model module union.
    for path in (
        ("alone", "kind"),
        ("locked", "kind"),
        ("sweep", "kind"),
        ("readout", "type"),
    ):
        assert node(schema, *path).get("hidden") is True, path
    # Selectors in a union position and one-value ordinary leaves stay visible.
    assert "hidden" not in node(branch(node(schema, "choice"), "alpha"), "kind")
    assert "hidden" not in node(branch(node(schema, "fluxes"), "linrange"), "kind")
    assert "hidden" not in node(schema, "locked", "count")
    assert "hidden" not in node(schema, "fixed")

    class UnionOnlyForm(FormModel):
        choice: Annotated[Alpha | Beta, Field(discriminator="kind")] = Field(
            default_factory=Alpha
        )
        fluxes: RangeInput = Field(default_factory=LinRange)

    assert '"hidden"' not in json.dumps(UnionOnlyForm.input_schema(FormEnv()))
