"""Input schema is a dynamic public contract, not documentation text."""

from collections.abc import Mapping
from typing import Annotated, Literal

import pytest
from jsonschema import Draft202012Validator
from pydantic import BaseModel, Field, JsonValue
from zcu_tools.gui.form import (
    Display,
    FormEnv,
    FormErrors,
    FormInput,
    FormModel,
    FormPath,
    FormState,
)
from zcu_tools.resources.context import MetaDict


class WholeContainerChoicesForm(FormModel):
    """Use JSON options for whole numeric and complex container leaves."""

    samples: tuple[float, ...]
    phasors: tuple[complex, ...]
    offsets: dict[str, complex]

    @classmethod
    def choices(cls, _env: FormEnv, /) -> Mapping[FormPath, tuple[JsonValue, ...]]:
        """Return the finite wire values allowed for each complete container."""
        return {
            ("samples",): ([1.0],),
            ("phasors",): ([{"__complex__": [1.0, 2.0]}],),
            ("offsets",): ({"drive": {"__complex__": [1.0, 2.0]}},),
        }


class SchemaForm(FormModel):
    """Project numeric/complex envelopes, nullable values and dynamic enums."""

    amount: Annotated[float, Display(unit="us")] = Field(default=2.0, gt=0)
    amplitude: complex = 1 + 0j
    mode: str = "alpha"
    optional: float | None = None

    @classmethod
    def choices(cls, env: FormEnv) -> Mapping[FormPath, tuple[JsonValue, ...]]:
        """The same captured choices drive schema and model validation."""
        return {("mode",): ("beta",) if env.md.get("beta_only") else ("alpha",)}


class SharedSection(BaseModel):
    """One nested type is reused at independently constrained field paths."""

    mode: str = "alpha"
    level: float = 2.0


class AlphaBranch(BaseModel):
    """First branch exposes the same choice field as the second branch."""

    kind: Literal["alpha"] = "alpha"
    mode: str = "alpha"


class BetaBranch(BaseModel):
    """Second branch keeps its own discriminator but shares the choice field."""

    kind: Literal["beta"] = "beta"
    mode: str = "beta"


class NestedChoicesForm(FormModel):
    """Top-level choices target reused, nullable and tagged nested fields."""

    restricted: SharedSection = Field(default_factory=SharedSection)
    unrestricted: SharedSection = Field(default_factory=SharedSection)
    optional: SharedSection | None = None
    branch: Annotated[AlphaBranch | BetaBranch, Field(discriminator="kind")] = Field(
        default_factory=AlphaBranch
    )

    @classmethod
    def choices(cls, _env: FormEnv, /) -> Mapping[FormPath, tuple[JsonValue, ...]]:
        """Constrain only the named model-field paths."""
        return {
            ("restricted", "mode"): ("alpha",),
            ("restricted", "level"): (2.0,),
            ("optional", "mode"): ("beta",),
            ("branch", "mode"): ("beta",),
        }


def test_nested_choices_schema_and_result_keep_reused_and_union_fields_isolated() -> (
    None
):
    env = FormEnv(md=MetaDict())
    validator = Draft202012Validator(NestedChoicesForm.input_schema(env))
    for kind in ("alpha", "beta"):
        valid: dict[str, JsonValue] = {
            "restricted": {"mode": "alpha"},
            "unrestricted": {"mode": "beta", "level": 3.0},
            "optional": {"mode": "beta"},
            "branch": {"kind": kind, "mode": "beta"},
        }
        assert validator.is_valid(valid)
        assert isinstance(
            FormState(FormInput(NestedChoicesForm, valid)).result(env),
            NestedChoicesForm,
        )
        invalid_cases: tuple[tuple[str, JsonValue], ...] = (
            ("restricted", {"mode": "beta"}),
            ("optional", {"mode": "alpha"}),
            ("branch", {"kind": kind, "mode": "alpha"}),
        )
        for field, bad in invalid_cases:
            invalid: dict[str, JsonValue] = {**valid, field: bad}
            assert not validator.is_valid(invalid)
            result = FormState(FormInput(NestedChoicesForm, invalid)).result(env)
            assert isinstance(result, FormErrors)
            assert (field, "mode") in result.messages
        expression: dict[str, JsonValue] = {
            **valid,
            "restricted": {"mode": "alpha", "level": {"expr": "2"}},
            "optional": None,
        }
        assert validator.is_valid(expression)
        assert isinstance(
            FormState(FormInput(NestedChoicesForm, expression)).result(env),
            NestedChoicesForm,
        )
        numeric_invalid: dict[str, JsonValue] = {
            **valid,
            "restricted": {"mode": "alpha", "level": 3.0},
        }
        assert not validator.is_valid(numeric_invalid)
        result = FormState(FormInput(NestedChoicesForm, numeric_invalid)).result(env)
        assert isinstance(result, FormErrors)
        assert ("restricted", "level") in result.messages


@pytest.mark.parametrize("path", [("samples", 0), ("offsets", "rack.left")])
def test_choices_reject_container_key_and_index_paths(path: FormPath) -> None:
    class ContainerChoicesForm(FormModel):
        """This invalid declaration targets a container's contents, not a field."""

        samples: tuple[float, ...] = (1.0,)
        offsets: dict[str, float] = Field(default_factory=dict)

        @classmethod
        def choices(cls, _env: FormEnv, /) -> Mapping[FormPath, tuple[JsonValue, ...]]:
            """Return the test's intentionally invalid declared choice path."""
            return {path: (1.0,)}

    env = FormEnv(md=MetaDict())
    with pytest.raises(ValueError, match="path|field|leaf|index|mapping"):
        ContainerChoicesForm.input_schema(env)
    state = FormState(ContainerChoicesForm.default_input(env))
    with pytest.raises(ValueError, match="path|field|leaf|index|mapping"):
        state.result(env)


def test_schema_without_state_describes_partial_expression_complex_and_choices() -> (
    None
):
    env = FormEnv(md=MetaDict())
    schema = SchemaForm.input_schema(env)
    validator = Draft202012Validator(schema)
    assert validator.is_valid({})
    assert validator.is_valid(
        {"amount": {"expr": "seed"}, "amplitude": {"__complex__": [1.0, 2.0]}}
    )
    assert validator.is_valid({"amplitude": {"expr": "1 + 2j"}, "optional": None})
    assert validator.is_valid({"amount": 3.0, "mode": "alpha"})
    assert not validator.is_valid({"amount": -1.0})
    assert not validator.is_valid({"amount": {"expr": 3}})
    assert not validator.is_valid({"amplitude": {"__complex__": [1.0]}})
    assert not validator.is_valid({"mode": "beta"})
    state = FormState(SchemaForm.default_input(env))
    assert state.input_schema(env) == schema


def test_schema_recomputes_dynamic_choices_for_each_request() -> None:
    alpha_env = FormEnv(md=MetaDict())
    beta_md = MetaDict()
    beta_md.update({"beta_only": True})
    beta_env = FormEnv(md=beta_md.snapshot())
    alpha = Draft202012Validator(SchemaForm.input_schema(alpha_env))
    beta = Draft202012Validator(SchemaForm.input_schema(beta_env))
    assert alpha.is_valid({"mode": "alpha"})
    assert not alpha.is_valid({"mode": "beta"})
    assert beta.is_valid({"mode": "beta"})
    assert not beta.is_valid({"mode": "alpha"})


def test_whole_container_choices_share_json_value_membership() -> None:
    env = FormEnv(md=MetaDict())
    valid: dict[str, JsonValue] = {
        "samples": [1.0],
        "phasors": [{"__complex__": [1.0, 2.0]}],
        "offsets": {"drive": {"__complex__": [1.0, 2.0]}},
    }
    validator = Draft202012Validator(WholeContainerChoicesForm.input_schema(env))
    assert validator.is_valid(valid)
    result = FormState(FormInput(WholeContainerChoicesForm, valid)).result(env)
    assert isinstance(result, WholeContainerChoicesForm), result
    assert result.samples == (1.0,)
    assert result.phasors == (1.0 + 2.0j,)
    assert result.offsets == {"drive": 1.0 + 2.0j}

    invalid_values: tuple[tuple[str, JsonValue], ...] = (
        ("samples", [2.0]),
        ("phasors", [{"__complex__": [3.0, 4.0]}]),
        ("offsets", {"drive": {"__complex__": [3.0, 4.0]}}),
    )
    for field, invalid_value in invalid_values:
        invalid = {**valid, field: invalid_value}
        assert not validator.is_valid(invalid)
        error = FormState(FormInput(WholeContainerChoicesForm, invalid)).result(env)
        assert isinstance(error, FormErrors)
        assert set(error.messages) == {(field,)}
