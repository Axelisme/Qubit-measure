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
    LinRange,
    RangeInput,
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
            ("mode",): () if env.md.get("empty_child") else ("child",),
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
            ("restricted", "mode"): () if env.md.get("empty_parent") else ("parent",),
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


@pytest.mark.parametrize("kind", ["alpha", "beta"])
def test_nullable_union_projects_branch_choices_and_complex_input(kind: str) -> None:
    env = FormEnv(md=MetaDict())
    validator = Draft202012Validator(NullableChoicesForm.input_schema(env))
    assert validator.is_valid({"branch": None})
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
        assert validator.is_valid(valid)
        result = FormState(FormInput(NullableChoicesForm, valid)).result(env)
        assert isinstance(result, NullableChoicesForm)
        assert result.branch is not None
        assert result.branch.amplitude == 1 + 2j
        invalid: dict[str, JsonValue] = {
            "branch": {"kind": kind, "mode": "other", "amplitude": amplitude}
        }
        assert not validator.is_valid(invalid)
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

    env = FormEnv(md=MetaDict())
    with pytest.raises(ValueError, match="path|field|leaf"):
        InvalidNullableForm.input_schema(env)
    active = FormState(FormInput(InvalidNullableForm, {"branch": {"kind": "alpha"}}))
    with pytest.raises(ValueError, match="path|field|leaf"):
        active.result(env)


@pytest.mark.parametrize("empty_parent", [False, True])
@pytest.mark.parametrize("empty_child", [False, True])
def test_deepest_local_choices_override_only_overlapping_parent_paths(
    empty_parent: bool, empty_child: bool
) -> None:
    md = MetaDict()
    md.update({"empty_parent": empty_parent, "empty_child": empty_child})
    env = FormEnv(md=md.snapshot())
    validator = Draft202012Validator(EnclosingChoicesForm.input_schema(env))
    valid: dict[str, JsonValue] = {
        "restricted": {
            "mode": "child",
            "level": {"expr": "2"},
            "nested": {"mode": "deep"},
        },
        "unrestricted": {"mode": "child", "level": 3.0, "nested": {"mode": "deep"}},
    }
    assert validator.is_valid(valid) is not empty_child
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
        assert not validator.is_valid(invalid)
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


def test_public_range_alias_projects_choices_without_reused_field_pollution() -> None:
    class RangeAliasForm(FormModel):
        """Use the exported alias rather than an equivalent inline union."""

        restricted: RangeInput = Field(default_factory=LinRange)
        unrestricted: RangeInput = Field(default_factory=LinRange)

        @classmethod
        def choices(cls, _env: FormEnv, /) -> Mapping[FormPath, tuple[JsonValue, ...]]:
            """Restrict n at one occurrence of the public alias."""
            return {("restricted", "n"): (2,)}

    env = FormEnv(md=MetaDict())
    validator = Draft202012Validator(RangeAliasForm.input_schema(env))
    counts: tuple[JsonValue, ...] = (2, {"expr": "2"})
    for count in counts:
        valid: dict[str, JsonValue] = {
            "restricted": {"kind": "linrange", "n": count},
            "unrestricted": {"kind": "linrange", "n": 3},
        }
        assert validator.is_valid(valid)
        result = FormState(FormInput(RangeAliasForm, valid)).result(env)
        assert isinstance(result, RangeAliasForm)
        assert isinstance(result.restricted, LinRange)
        assert isinstance(result.unrestricted, LinRange)
        assert result.restricted.n == 2
        assert result.unrestricted.n == 3
    invalid: dict[str, JsonValue] = {
        "restricted": {"kind": "linrange", "n": 3},
        "unrestricted": {"kind": "linrange", "n": 3},
    }
    errors = FormState(FormInput(RangeAliasForm, invalid)).result(env)
    assert isinstance(errors, FormErrors)
    assert ("restricted", "n") in errors.messages
    assert not validator.is_valid(invalid)


def test_nullable_public_range_alias_projects_choices_and_null() -> None:
    class NullableAliasForm(FormModel):
        """Nullable exported aliases retain the same nested input contract."""

        fluxes: RangeInput | None = None

        @classmethod
        def choices(cls, _env: FormEnv, /) -> Mapping[FormPath, tuple[JsonValue, ...]]:
            """Constrain only the active range branch's n leaf."""
            return {("fluxes", "n"): (2,)}

    env = FormEnv(md=MetaDict())
    validator = Draft202012Validator(NullableAliasForm.input_schema(env))
    values: tuple[JsonValue, ...] = (
        None,
        {"kind": "linrange", "n": 2},
        {"kind": "linrange", "n": {"expr": "2"}},
    )
    for value in values:
        valid: dict[str, JsonValue] = {"fluxes": value}
        assert validator.is_valid(valid)
        assert isinstance(
            FormState(FormInput(NullableAliasForm, valid)).result(env),
            NullableAliasForm,
        )
    invalid: dict[str, JsonValue] = {"fluxes": {"kind": "linrange", "n": 3}}
    assert not validator.is_valid(invalid)
    errors = FormState(FormInput(NullableAliasForm, invalid)).result(env)
    assert isinstance(errors, FormErrors)
    assert ("fluxes", "n") in errors.messages


class NumericAliasBranch(FormModel):
    """Two ordinary numeric tags identify the same branch."""

    kind: Literal[1, 2] = 1
    amount: float = 3.0


class NumericOtherBranch(FormModel):
    """Another intrinsic tag remains distinct from dynamic choices."""

    kind: Literal[3] = 3
    other: float = 4.0


class DiscriminatorChoicesForm(FormModel):
    """The host may restrict selector choices without changing branch identity."""

    branch: Annotated[
        NumericAliasBranch | NumericOtherBranch, Field(discriminator="kind")
    ]

    @classmethod
    def choices(cls, env: FormEnv) -> Mapping[FormPath, tuple[JsonValue, ...]]:
        """Allow one alias or no tags using only the captured context."""
        return {("branch", "kind"): () if env.md.get("empty_tags") else (2,)}


def test_discriminator_identity_survives_dynamic_choices_refresh() -> None:
    env = FormEnv(md=MetaDict())
    state = FormState(FormInput(DiscriminatorChoicesForm, {"branch": {"kind": 2}}))
    before = state.input.data
    for empty in (False, True, False):
        env.md.update({"empty_tags": empty})
        schema = DiscriminatorChoicesForm.input_schema(env)
        properties = schema["properties"]
        assert isinstance(properties, dict)
        branch = properties["branch"]
        assert isinstance(branch, dict)
        variants = branch["oneOf"]
        assert isinstance(variants, list)
        identities: list[JsonValue] = []
        for variant in variants:
            assert isinstance(variant, dict)
            identities.append(variant["x-form-discriminator-values"])
            fields = variant["properties"]
            assert isinstance(fields, dict)
            selector = fields["kind"]
            assert isinstance(selector, dict)
            alternatives = selector["anyOf"]
            assert isinstance(alternatives, list)
            direct = alternatives[0]
            assert isinstance(direct, dict)
            if empty:
                assert direct["not"] == {}
            else:
                assert direct["enum"] == [2]
        assert identities == [[1, 2], [3]]
        validator = Draft202012Validator(schema)
        assert validator.is_valid(before) is not empty
        result = state.result(env)
        if empty:
            assert isinstance(result, FormErrors)
            assert ("branch", "kind") in result.messages
        else:
            assert isinstance(result, DiscriminatorChoicesForm)
            assert isinstance(result.branch, NumericAliasBranch)
            assert result.branch.kind == 2
        assert state.input.data == before


type NumericChoice = Annotated[
    NumericAliasBranch | NumericOtherBranch, Field(discriminator="kind")
]


def identity_input() -> dict[str, JsonValue]:
    """Return complete wire input for containers and reused exported range aliases."""
    return {
        "sequence": [{"kind": 2, "amount": 3.0}, {"kind": 3, "other": 4.0}],
        "fixed": [{"kind": 1, "amount": 3.0}, {"kind": 3, "other": 4.0}],
        "mapping": {"drive": {"kind": 2, "amount": 3.0}},
        "restricted": {"kind": "linrange", "n": 2},
        "unrestricted": {"kind": "linrange", "n": 4},
    }


class SchemaIdentityForm(FormModel):
    """Tagged schemas occur inside whole leaves and shared alias definitions."""

    sequence: list[NumericChoice]
    fixed: tuple[NumericChoice, NumericChoice]
    mapping: dict[str, NumericChoice]
    restricted: RangeInput
    unrestricted: RangeInput

    @classmethod
    def choices(cls, env: FormEnv) -> Mapping[FormPath, tuple[JsonValue, ...]]:
        """Refresh whole-leaf membership and only one range occurrence's n."""
        empty = env.md.get("empty_containers")
        data = identity_input()
        return {
            ("sequence",): () if empty else (data["sequence"],),
            ("fixed",): () if empty else (data["fixed"],),
            ("mapping",): () if empty else (data["mapping"],),
            ("restricted", "n"): (3,) if env.md.get("three_only") else (2,),
        }


def schema_object(value: JsonValue) -> dict[str, JsonValue]:
    """Require a public JSON Schema object; fail an assertion otherwise."""
    assert isinstance(value, dict)
    return value


def identity_values(node: dict[str, JsonValue]) -> list[JsonValue]:
    """Read each oneOf branch's public identity, retaining None when absent."""
    branches = node["oneOf"]
    assert isinstance(branches, list)
    return [
        schema_object(branch).get("x-form-discriminator-values") for branch in branches
    ]


def test_container_discriminator_identity_and_retained_discriminator_identity_refresh() -> (
    None
):
    env = FormEnv(md=MetaDict())
    state = FormState(FormInput(SchemaIdentityForm, identity_input()))
    before = state.input.data
    for empty, three_only in (
        (False, False),
        (True, False),
        (False, True),
        (False, False),
    ):
        env.md.update({"empty_containers": empty, "three_only": three_only})
        schema = SchemaIdentityForm.input_schema(env)
        fields = schema_object(schema["properties"])
        definitions = schema_object(schema["$defs"])
        sequence = schema_object(fields["sequence"])
        fixed = schema_object(fields["fixed"])
        mapping = schema_object(fields["mapping"])
        prefix = fixed["prefixItems"]
        assert isinstance(prefix, list)
        observed = {
            "items": identity_values(schema_object(sequence["items"])),
            "prefix0": identity_values(schema_object(prefix[0])),
            "prefix1": identity_values(schema_object(prefix[1])),
            "additionalProperties": identity_values(
                schema_object(mapping["additionalProperties"])
            ),
            "NumericChoice": identity_values(
                schema_object(definitions["NumericChoice"])
            ),
            "RangeInput": identity_values(schema_object(definitions["RangeInput"])),
            "restricted": identity_values(schema_object(fields["restricted"])),
            "unrestricted": identity_values(schema_object(fields["unrestricted"])),
        }
        assert observed == {
            "items": [[1, 2], [3]],
            "prefix0": [[1, 2], [3]],
            "prefix1": [[1, 2], [3]],
            "additionalProperties": [[1, 2], [3]],
            "NumericChoice": [[1, 2], [3]],
            "RangeInput": [["linrange"], ["list"]],
            "restricted": [["linrange"], ["list"]],
            "unrestricted": [["linrange"], ["list"]],
        }
        for location in (
            "items",
            "prefix0",
            "prefix1",
            "additionalProperties",
            "NumericChoice",
        ):
            for tags in observed[location]:
                assert isinstance(tags, list)
                assert all(type(tag) is int for tag in tags)
        for name, node in (
            ("sequence", sequence),
            ("fixed", fixed),
            ("mapping", mapping),
        ):
            if empty:
                assert node["not"] == {}
            else:
                assert node["enum"] == [before[name]]
        restricted = schema_object(fields["restricted"])
        branches = restricted["oneOf"]
        assert isinstance(branches, list)
        linear = schema_object(branches[0])
        linear_fields = schema_object(linear["properties"])
        count_schema = schema_object(linear_fields["n"])
        alternatives = count_schema["anyOf"]
        assert isinstance(alternatives, list)
        allowed_n = 3 if three_only else 2
        assert schema_object(alternatives[0])["enum"] == [allowed_n]
        validator = Draft202012Validator(schema)
        assert validator.is_valid(before) == (not empty and not three_only)
        candidate: dict[str, JsonValue] = {
            **before,
            "restricted": {"kind": "linrange", "n": allowed_n},
        }
        if not empty:
            assert validator.is_valid(candidate)
            candidate_result = FormState(
                FormInput(SchemaIdentityForm, candidate)
            ).result(env)
            assert isinstance(candidate_result, SchemaIdentityForm)
            assert isinstance(candidate_result.restricted, LinRange)
            assert isinstance(candidate_result.unrestricted, LinRange)
            assert candidate_result.restricted.n == allowed_n
            assert candidate_result.unrestricted.n == 4
        result = state.result(env)
        if empty or three_only:
            assert isinstance(result, FormErrors)
            if empty:
                assert {
                    ("sequence",),
                    ("fixed",),
                    ("mapping",),
                } <= result.messages.keys()
            if three_only:
                assert ("restricted", "n") in result.messages
        else:
            assert isinstance(result, SchemaIdentityForm)
            assert isinstance(result.sequence[0], NumericAliasBranch)
            assert result.sequence[0].kind == 2
            assert result.fixed[0].kind == 1
            assert result.mapping["drive"].kind == 2
        assert state.input.data == before
    invalid: dict[str, JsonValue] = {
        **before,
        "unrestricted": {"kind": "linrange", "n": 0},
    }
    assert not Draft202012Validator(schema).is_valid(invalid)
    errors = FormState(FormInput(SchemaIdentityForm, invalid)).result(env)
    assert isinstance(errors, FormErrors)
    assert ("unrestricted", "n") in errors.messages


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
