"""Tagged input schemas preserve native identity and branch-local constraints."""

from collections.abc import Callable, Mapping
from typing import Annotated, Literal

import pytest
from jsonschema import Draft202012Validator
from pydantic import Field, JsonValue
from pydantic.json_schema import SkipJsonSchema
from zcu_tools.gui.form import (
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


def identity_snapshot(schema: dict[str, JsonValue]) -> dict[str, list[JsonValue]]:
    """Read identities at the eight public positions of SchemaIdentityForm's schema."""
    fields = schema_object(schema["properties"])
    definitions = schema_object(schema["$defs"])
    sequence = schema_object(fields["sequence"])
    fixed = schema_object(fields["fixed"])
    mapping = schema_object(fields["mapping"])
    prefix = fixed["prefixItems"]
    assert isinstance(prefix, list)
    return {
        "items": identity_values(schema_object(sequence["items"])),
        "prefix0": identity_values(schema_object(prefix[0])),
        "prefix1": identity_values(schema_object(prefix[1])),
        "additionalProperties": identity_values(
            schema_object(mapping["additionalProperties"])
        ),
        "NumericChoice": identity_values(schema_object(definitions["NumericChoice"])),
        "RangeInput": identity_values(schema_object(definitions["RangeInput"])),
        "restricted": identity_values(schema_object(fields["restricted"])),
        "unrestricted": identity_values(schema_object(fields["unrestricted"])),
    }


def range_count_options(node: dict[str, JsonValue]) -> JsonValue:
    """Read the ordinary n options from the first branch of a RangeInput schema."""
    branches = node["oneOf"]
    assert isinstance(branches, list)
    fields = schema_object(schema_object(branches[0])["properties"])
    count = schema_object(fields["n"])
    alternatives = count["anyOf"]
    assert isinstance(alternatives, list)
    return schema_object(alternatives[0])["enum"]


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
        observed = identity_snapshot(schema)
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
        for name in ("sequence", "fixed", "mapping"):
            node = schema_object(fields[name])
            if empty:
                assert node["not"] == {}
            else:
                assert node["enum"] == [before[name]]
        allowed_n = 3 if three_only else 2
        assert range_count_options(schema_object(fields["restricted"])) == [allowed_n]
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


class StringNumericTagBranch(FormModel):
    """The string tag 1 is not the numeric tag 1."""

    kind: Literal["1"] = "1"
    text_only: float = 4.0


class TypedTagCollisionForm(FormModel):
    """Typed intrinsic tags must not be grouped by their string representation."""

    branch: Annotated[
        NumericAliasBranch | StringNumericTagBranch, Field(discriminator="kind")
    ]


def test_discriminator_identity_keeps_numeric_and_string_tags_distinct() -> None:
    env = FormEnv(md=MetaDict())
    schema = TypedTagCollisionForm.input_schema(env)
    fields = schema_object(schema["properties"])
    branches = schema_object(fields["branch"])["oneOf"]
    assert isinstance(branches, list)
    observed: dict[str, JsonValue] = {}
    for value in branches:
        branch = schema_object(value)
        properties = schema_object(branch["properties"])
        name = "amount" if "amount" in properties else "text_only"
        observed[name] = branch.get("x-form-discriminator-values")
    assert observed == {"amount": [1, 2], "text_only": ["1"]}
    for tag in (1, 2, "1"):
        data: dict[str, JsonValue] = {"branch": {"kind": tag}}
        assert Draft202012Validator(schema).is_valid(data)
        result = FormState(FormInput(TypedTagCollisionForm, data)).result(env)
        assert isinstance(result, TypedTagCollisionForm)
        assert result.branch.kind == tag
        assert type(result.branch.kind) is type(tag)


class OmittedChoiceForm(FormModel):
    """Omitting one schema branch must leave normal typed aliases editable."""

    branch: Annotated[
        NumericAliasBranch | SkipJsonSchema[NumericOtherBranch],
        Field(discriminator="kind"),
    ]


class InvalidJsonChoiceBranch(NumericOtherBranch):
    """A callable choice cannot be emitted as a native JSON Schema branch."""

    callback: Callable[[], None]


class InvalidJsonChoiceForm(FormModel):
    """Native schema generation skips the invalid choice, not the entire field."""

    branch: Annotated[
        NumericAliasBranch | InvalidJsonChoiceBranch, Field(discriminator="kind")
    ]


@pytest.mark.parametrize("model", [OmittedChoiceForm, InvalidJsonChoiceForm])
def test_skipped_discriminator_choice_keeps_normal_field_and_identity(
    model: type[OmittedChoiceForm | InvalidJsonChoiceForm],
) -> None:
    env = FormEnv(md=MetaDict())
    schema = model.input_schema(env)
    fields = schema_object(schema["properties"])
    branch = schema_object(fields["branch"])
    assert identity_values(branch) == [[1, 2]]
    for tag in (1, 2):
        data: dict[str, JsonValue] = {"branch": {"kind": tag, "amount": 7.0}}
        assert Draft202012Validator(schema).is_valid(data)
        state = FormState(FormInput(model, data))
        result = state.result(env)
        assert isinstance(result, (OmittedChoiceForm, InvalidJsonChoiceForm))
        assert isinstance(result.branch, NumericAliasBranch)
        assert result.branch.kind == tag
        assert type(result.branch.kind) is int
        assert result.branch.amount == 7.0
        assert state.input.data == data


class SkippedChoiceWithChoices(NumericOtherBranch):
    """A skipped choice's kind restriction belongs only to that choice."""

    @classmethod
    def choices(cls, _env: FormEnv, /) -> Mapping[FormPath, tuple[JsonValue, ...]]:
        """Restrict the skipped model to its own intrinsic tag."""
        return {("kind",): (3,)}


class InvalidChoiceWithChoices(SkippedChoiceWithChoices):
    """Native generation skips a callable branch with different local choices."""

    callback: Callable[[], None]


class RetainedChoiceWithChoices(NumericAliasBranch):
    """The emitted numeric branch owns dynamic amount choices."""

    @classmethod
    def choices(cls, env: FormEnv, /) -> Mapping[FormPath, tuple[JsonValue, ...]]:
        """Read empty or singleton amount choices from the captured environment."""
        if env.md.get("empty_amounts", False):
            return {("amount",): ()}
        return {("amount",): (9.0,) if env.md.get("nine_only", False) else (7.0,)}


type AliasTags = Literal[1, 2]


class AliasTagChoice(RetainedChoiceWithChoices):
    """A named Literal tag keeps the numeric branch's local choices."""

    kind: AliasTags = 1


class UnionTagChoice(RetainedChoiceWithChoices):
    """A union of Literals identifies the same branch with both numeric tags."""

    kind: Literal[1] | Literal[2] = 1


class AliasTagForm(FormModel):
    """Native named tag aliases remain usable by schema and state."""

    branch: Annotated[
        AliasTagChoice | StringNumericTagBranch, Field(discriminator="kind")
    ]


class UnionTagForm(FormModel):
    """Native unions of Literal tags remain usable by schema and state."""

    branch: Annotated[
        UnionTagChoice | StringNumericTagBranch, Field(discriminator="kind")
    ]


@pytest.mark.parametrize("model", [AliasTagForm, UnionTagForm])
def test_literal_tag_declaration_keeps_schema_and_result_agreement(
    model: type[AliasTagForm | UnionTagForm],
) -> None:
    env = FormEnv(md=MetaDict())
    state = FormState(FormInput(model, {"branch": {"kind": 2, "amount": 7.0}}))
    before = state.input.data
    for empty, nine_only in (
        (False, False),
        (True, False),
        (False, True),
        (False, False),
    ):
        env.md.update({"empty_amounts": empty, "nine_only": nine_only})
        schema = model.input_schema(env)
        fields = schema_object(schema["properties"])
        identities = identity_values(schema_object(fields["branch"]))
        assert identities == [[1, 2], ["1"]]
        numeric_tags = identities[0]
        string_tags = identities[1]
        assert isinstance(numeric_tags, list) and isinstance(string_tags, list)
        assert all(type(tag) is int for tag in numeric_tags)
        assert all(type(tag) is str for tag in string_tags)
        validator = Draft202012Validator(schema)
        for tag in (1, 2):
            for amount in (7.0, 9.0):
                data: dict[str, JsonValue] = {"branch": {"kind": tag, "amount": amount}}
                accepted = not empty and amount == (9.0 if nine_only else 7.0)
                assert validator.is_valid(data) is accepted
                candidate = FormState(FormInput(model, data))
                result = candidate.result(env)
                if accepted:
                    assert isinstance(result, model)
                    assert isinstance(result.branch, (AliasTagChoice, UnionTagChoice))
                    assert result.branch.kind == tag
                    assert type(result.branch.kind) is int
                    assert result.branch.amount == amount
                else:
                    assert isinstance(result, FormErrors)
                    assert ("branch", "amount") in result.messages
                assert candidate.input.data == data
        string_data: dict[str, JsonValue] = {"branch": {"kind": "1", "text_only": 4.0}}
        assert validator.is_valid(string_data)
        string_state = FormState(FormInput(model, string_data))
        string_result = string_state.result(env)
        assert isinstance(string_result, model)
        assert isinstance(string_result.branch, StringNumericTagBranch)
        assert string_result.branch.kind == "1"
        assert type(string_result.branch.kind) is str
        assert string_result.branch.text_only == 4.0
        assert string_state.input.data == string_data
        current = state.result(env)
        if empty or nine_only:
            assert isinstance(current, FormErrors)
            assert ("branch", "amount") in current.messages
        else:
            assert isinstance(current, model)
            assert isinstance(current.branch, (AliasTagChoice, UnionTagChoice))
            assert current.branch.kind == 2
            assert type(current.branch.kind) is int
            assert current.branch.amount == 7.0
        assert state.input.data == before


class SkipFirstOwnerForm(FormModel):
    """Skipping the first choice must retain both following model owners."""

    branch: Annotated[
        SkipJsonSchema[SkippedChoiceWithChoices]
        | RetainedChoiceWithChoices
        | StringNumericTagBranch,
        Field(discriminator="kind"),
    ]


class InvalidFirstOwnerForm(FormModel):
    """An invalid first schema choice must not shift following model owners."""

    branch: Annotated[
        InvalidChoiceWithChoices | RetainedChoiceWithChoices | StringNumericTagBranch,
        Field(discriminator="kind"),
    ]


class SkipMiddleOwnerForm(FormModel):
    """Skipping a middle choice must retain the final string-tag owner."""

    branch: Annotated[
        RetainedChoiceWithChoices
        | SkipJsonSchema[SkippedChoiceWithChoices]
        | StringNumericTagBranch,
        Field(discriminator="kind"),
    ]


class InvalidMiddleOwnerForm(FormModel):
    """An invalid middle schema choice must not shift the final model owner."""

    branch: Annotated[
        RetainedChoiceWithChoices | InvalidChoiceWithChoices | StringNumericTagBranch,
        Field(discriminator="kind"),
    ]


@pytest.mark.parametrize(
    "model",
    [
        SkipFirstOwnerForm,
        InvalidFirstOwnerForm,
        SkipMiddleOwnerForm,
        InvalidMiddleOwnerForm,
    ],
)
def test_skipped_choice_keeps_branch_ownership_through_choices_refresh(
    model: type[
        SkipFirstOwnerForm
        | InvalidFirstOwnerForm
        | SkipMiddleOwnerForm
        | InvalidMiddleOwnerForm
    ],
) -> None:
    env = FormEnv(md=MetaDict())
    state = FormState(FormInput(model, {"branch": {"kind": 2, "amount": 7.0}}))
    before = state.input.data
    string_state = FormState(
        FormInput(model, {"branch": {"kind": "1", "text_only": 4.0}})
    )
    string_before = string_state.input.data
    for empty, nine_only in (
        (False, False),
        (True, False),
        (False, True),
        (False, False),
    ):
        env.md.update({"empty_amounts": empty, "nine_only": nine_only})
        schema = model.input_schema(env)
        fields = schema_object(schema["properties"])
        identities = identity_values(schema_object(fields["branch"]))
        assert identities == [[1, 2], ["1"]]
        numeric_tags = identities[0]
        string_tags = identities[1]
        assert isinstance(numeric_tags, list) and isinstance(string_tags, list)
        assert all(type(tag) is int for tag in numeric_tags)
        assert all(type(tag) is str for tag in string_tags)
        validator = Draft202012Validator(schema)
        allowed = 9.0 if nine_only else 7.0
        for tag in (1, 2):
            for amount in (7.0, 9.0):
                data: dict[str, JsonValue] = {"branch": {"kind": tag, "amount": amount}}
                accepted = not empty and amount == allowed
                assert validator.is_valid(data) is accepted
                candidate = FormState(FormInput(model, data))
                result = candidate.result(env)
                if accepted:
                    assert isinstance(result, model)
                    assert isinstance(result.branch, RetainedChoiceWithChoices)
                    assert result.branch.kind == tag
                    assert type(result.branch.kind) is int
                    assert result.branch.amount == amount
                else:
                    assert isinstance(result, FormErrors)
                    assert ("branch", "amount") in result.messages
                assert candidate.input.data == data
        assert validator.is_valid(string_before)
        string_result = string_state.result(env)
        assert isinstance(string_result, model)
        assert isinstance(string_result.branch, StringNumericTagBranch)
        assert string_result.branch.kind == "1"
        assert type(string_result.branch.kind) is str
        assert string_result.branch.text_only == 4.0
        assert string_state.input.data == string_before
        assert validator.is_valid(before) == (not empty and not nine_only)
        current = state.result(env)
        if empty or nine_only:
            assert isinstance(current, FormErrors)
            assert ("branch", "amount") in current.messages
        else:
            assert isinstance(current, model)
            assert isinstance(current.branch, RetainedChoiceWithChoices)
            assert current.branch.kind == 2
        assert state.input.data == before
