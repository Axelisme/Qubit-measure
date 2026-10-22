"""Dynamic choices constrain result values at independent model-field paths."""

from collections.abc import Mapping
from typing import Annotated, Literal

import pytest
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
from zcu_tools.gui.form_adapter import LinRange

from tests.gui._form_context import captured_ext_env


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


class NullableAlphaBranch(FormModel):
    """A nullable branch owns its choices and tagged complex input."""

    kind: Literal["alpha"] = "alpha"
    mode: str = "local-alpha"
    amplitude: Annotated[complex, Display(unit="V")] = 1 + 2j

    @classmethod
    def choices(cls, _env: FormEnv, /) -> Mapping[FormPath, tuple[JsonValue, ...]]:
        """Allow only this branch's local mode."""
        return {("mode",): ("local-alpha",)}


class NullableBetaBranch(FormModel):
    """The other nullable branch has an independent local mode."""

    kind: Literal["beta"] = "beta"
    mode: str = "local-beta"
    amplitude: complex = 1 + 2j

    @classmethod
    def choices(cls, _env: FormEnv, /) -> Mapping[FormPath, tuple[JsonValue, ...]]:
        """Allow only this branch's local mode."""
        return {("mode",): ("local-beta",)}


class NullableChoicesForm(FormModel):
    """Project both locally constrained branches while also permitting null."""

    branch: Annotated[
        NullableAlphaBranch | NullableBetaBranch | None, Field(discriminator="kind")
    ] = None


class DeepChoicesForm(FormModel):
    """The deepest local hook overrides both ancestors."""

    mode: str = "deep"

    @classmethod
    def choices(cls, _env: FormEnv, /) -> Mapping[FormPath, tuple[JsonValue, ...]]:
        """Constrain the leaf at the deepest model occurrence."""
        return {("mode",): ("deep",)}


class LocalChoicesForm(FormModel):
    """Reused children keep local choices and inherit other parent paths."""

    mode: str = "child"
    level: float = 2.0
    nested: DeepChoicesForm = Field(default_factory=DeepChoicesForm)

    @classmethod
    def choices(cls, env: FormEnv, /) -> Mapping[FormPath, tuple[JsonValue, ...]]:
        """An empty local declaration must also override its parent."""
        return {
            ("mode",): ()
            if (
                env.parameters is not None
                and env.parameters.lookup("general.ext.empty_child").value
            )
            else ("child",),
            ("nested", "mode"): ("middle",),
        }


class EnclosingChoicesForm(FormModel):
    """Only the restricted occurrence inherits the enclosing level constraint."""

    restricted: LocalChoicesForm = Field(default_factory=LocalChoicesForm)
    unrestricted: LocalChoicesForm = Field(default_factory=LocalChoicesForm)

    @classmethod
    def choices(cls, env: FormEnv, /) -> Mapping[FormPath, tuple[JsonValue, ...]]:
        """Use conflicting and nonoverlapping choices at separate occurrences."""
        return {
            ("restricted", "mode"): ()
            if (
                env.parameters is not None
                and env.parameters.lookup("general.ext.empty_parent").value
            )
            else ("parent",),
            ("restricted", "level"): (2.0,),
            ("restricted", "nested", "mode"): ("ancestor",),
        }


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


def test_nested_choices_result_keep_reused_and_union_fields_isolated() -> None:
    env = FormEnv()
    for kind in ("alpha", "beta"):
        valid: dict[str, JsonValue] = {
            "restricted": {"mode": "alpha"},
            "unrestricted": {"mode": "beta", "level": 3.0},
            "optional": {"mode": "beta"},
            "branch": {"kind": kind, "mode": "beta"},
        }
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
            result = FormState(FormInput(NestedChoicesForm, invalid)).result(env)
            assert isinstance(result, FormErrors)
            assert (field, "mode") in result.messages
        # A choice leaf takes a listed value, not an expression.
        expression: dict[str, JsonValue] = {
            **valid,
            "restricted": {"mode": "alpha", "level": {"expr": "2"}},
            "optional": None,
        }
        result = FormState(FormInput(NestedChoicesForm, expression)).result(env)
        assert isinstance(result, FormErrors)
        assert ("restricted", "level") in result.messages
        numeric_invalid: dict[str, JsonValue] = {
            **valid,
            "restricted": {"mode": "alpha", "level": 3.0},
        }
        result = FormState(FormInput(NestedChoicesForm, numeric_invalid)).result(env)
        assert isinstance(result, FormErrors)
        assert ("restricted", "level") in result.messages


@pytest.mark.parametrize("kind", ["alpha", "beta"])
def test_nullable_union_projects_branch_choices_and_complex_input(kind: str) -> None:
    env = FormEnv()
    null_result = FormState(FormInput(NullableChoicesForm, {"branch": None})).result(
        env
    )
    assert isinstance(null_result, NullableChoicesForm)
    assert null_result.branch is None
    amplitudes: tuple[dict[str, JsonValue], ...] = (
        {"__complex__": [1.0, 2.0]},
        {"expr": "1 + 2j"},
    )
    for amplitude in amplitudes:
        valid: dict[str, JsonValue] = {
            "branch": {"kind": kind, "mode": f"local-{kind}", "amplitude": amplitude}
        }
        result = FormState(FormInput(NullableChoicesForm, valid)).result(env)
        assert isinstance(result, NullableChoicesForm)
        assert result.branch is not None
        assert result.branch.amplitude == 1 + 2j
        invalid: dict[str, JsonValue] = {
            "branch": {"kind": kind, "mode": "other", "amplitude": amplitude}
        }
        errors = FormState(FormInput(NullableChoicesForm, invalid)).result(env)
        assert isinstance(errors, FormErrors)
        assert ("branch", "mode") in errors.messages


def test_nullable_union_rejects_invalid_branch_local_choice_path() -> None:
    class InvalidAlphaBranch(NullableAlphaBranch):
        """Declare a nonexistent leaf on one branch only."""

        @classmethod
        def choices(cls, _env: FormEnv, /) -> Mapping[FormPath, tuple[JsonValue, ...]]:
            """Return an intentionally invalid branch-local declaration."""
            return {("missing",): ("value",)}

    class InvalidNullableForm(FormModel):
        """The invalid branch must be checked even when the default is null."""

        branch: Annotated[
            InvalidAlphaBranch | NullableBetaBranch | None, Field(discriminator="kind")
        ] = None

    env = FormEnv()
    active = FormState(FormInput(InvalidNullableForm, {"branch": {"kind": "alpha"}}))
    with pytest.raises(ValueError, match="path|field|leaf"):
        active.result(env)


@pytest.mark.parametrize("empty_parent", [False, True])
@pytest.mark.parametrize("empty_child", [False, True])
def test_deepest_local_choices_override_only_overlapping_parent_paths(
    empty_parent: bool, empty_child: bool
) -> None:
    env = captured_ext_env(empty_parent=empty_parent, empty_child=empty_child)
    valid: dict[str, JsonValue] = {
        "restricted": {"mode": "child", "level": 2.0, "nested": {"mode": "deep"}},
        "unrestricted": {"mode": "child", "level": 3.0, "nested": {"mode": "deep"}},
    }
    result = FormState(FormInput(EnclosingChoicesForm, valid)).result(env)
    if empty_child:
        assert isinstance(result, FormErrors)
        assert ("restricted", "mode") in result.messages
        assert ("unrestricted", "mode") in result.messages
        return
    assert isinstance(result, EnclosingChoicesForm)
    assert result.restricted.mode == result.unrestricted.mode == "child"
    assert result.restricted.nested.mode == result.unrestricted.nested.mode == "deep"
    assert result.restricted.level == 2.0
    assert result.unrestricted.level == 3.0
    invalid_fields: tuple[tuple[str, JsonValue], ...] = (
        ("mode", "parent"),
        ("level", 3.0),
        ("nested", {"mode": "middle"}),
    )
    for field, bad in invalid_fields:
        restricted: dict[str, JsonValue] = {
            "mode": "child",
            "level": 2.0,
            "nested": {"mode": "deep"},
        }
        restricted[field] = bad
        invalid: dict[str, JsonValue] = {**valid, "restricted": restricted}
        errors = FormState(FormInput(EnclosingChoicesForm, invalid)).result(env)
        assert isinstance(errors, FormErrors)
        path = (
            ("restricted", "nested", "mode")
            if field == "nested"
            else ("restricted", field)
        )
        assert path in errors.messages


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

    env = FormEnv()
    with pytest.raises(ValueError, match="path|field|leaf|index|mapping"):
        FormState(ContainerChoicesForm.default_input(env)).describe((), env)
    state = FormState(ContainerChoicesForm.default_input(env))
    with pytest.raises(ValueError, match="path|field|leaf|index|mapping"):
        state.result(env)


def test_public_range_alias_projects_choices_without_reused_field_pollution() -> None:
    class RangeAliasForm(FormModel):
        """Reuse the exported linear range model at two independent paths."""

        restricted: LinRange = Field(
            default_factory=lambda: LinRange(start=0, stop=1, n=3)
        )
        unrestricted: LinRange = Field(
            default_factory=lambda: LinRange(start=0, stop=1, n=3)
        )

        @classmethod
        def choices(cls, _env: FormEnv, /) -> Mapping[FormPath, tuple[JsonValue, ...]]:
            """Restrict n at one occurrence of the public range model."""
            return {("restricted", "n"): (2,)}

    env = FormEnv()
    valid: dict[str, JsonValue] = {
        "restricted": {"kind": "linrange", "start": 0, "stop": 1, "n": 2},
        "unrestricted": {"kind": "linrange", "start": 0, "stop": 1, "n": 3},
    }
    result = FormState(FormInput(RangeAliasForm, valid)).result(env)
    assert isinstance(result, RangeAliasForm)
    assert isinstance(result.restricted, LinRange)
    assert isinstance(result.unrestricted, LinRange)
    assert result.restricted.n == 2
    assert result.unrestricted.n == 3
    # Only the unrestricted occurrence is a text field taking an expression.
    expression: dict[str, JsonValue] = {
        "restricted": {"kind": "linrange", "start": 0, "stop": 1, "n": 2},
        "unrestricted": {"kind": "linrange", "start": 0, "stop": 1, "n": {"expr": "3"}},
    }
    assert isinstance(
        FormState(FormInput(RangeAliasForm, expression)).result(env), RangeAliasForm
    )
    invalid: dict[str, JsonValue] = {
        "restricted": {"kind": "linrange", "start": 0, "stop": 1, "n": 3},
        "unrestricted": {"kind": "linrange", "start": 0, "stop": 1, "n": 3},
    }
    errors = FormState(FormInput(RangeAliasForm, invalid)).result(env)
    assert isinstance(errors, FormErrors)
    assert ("restricted", "n") in errors.messages


def test_nullable_public_range_alias_projects_choices_and_null() -> None:
    class NullableAliasForm(FormModel):
        """Nullable linear ranges retain their nested input contract."""

        fluxes: LinRange | None = None

        @classmethod
        def choices(cls, _env: FormEnv, /) -> Mapping[FormPath, tuple[JsonValue, ...]]:
            """Constrain only the active range branch's n leaf."""
            return {("fluxes", "n"): (2,)}

    env = FormEnv()
    values: tuple[JsonValue, ...] = (
        None,
        {"kind": "linrange", "start": 0, "stop": 1, "n": 2},
    )
    for value in values:
        valid: dict[str, JsonValue] = {"fluxes": value}
        assert isinstance(
            FormState(FormInput(NullableAliasForm, valid)).result(env),
            NullableAliasForm,
        )
    invalid: dict[str, JsonValue] = {
        "fluxes": {"kind": "linrange", "start": 0, "stop": 1, "n": 3}
    }
    errors = FormState(FormInput(NullableAliasForm, invalid)).result(env)
    assert isinstance(errors, FormErrors)
    assert ("fluxes", "n") in errors.messages


def test_whole_container_choices_share_json_value_membership() -> None:
    env = FormEnv()
    valid: dict[str, JsonValue] = {
        "samples": [1.0],
        "phasors": [{"__complex__": [1.0, 2.0]}],
        "offsets": {"drive": {"__complex__": [1.0, 2.0]}},
    }
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
        error = FormState(FormInput(WholeContainerChoicesForm, invalid)).result(env)
        assert isinstance(error, FormErrors)
        assert set(error.messages) == {(field,)}
