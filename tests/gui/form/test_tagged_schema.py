"""Tagged input preserves native identity and branch-local constraints."""

from collections.abc import Callable, Mapping
from typing import Annotated, Literal

import pytest
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
)

from tests.gui._form_context import captured_ext_env


class NumericAliasBranch(FormModel):
    """Two ordinary numeric tags identify the same branch."""

    kind: Literal[1, 2] = 1
    amount: float = 3.0


class NumericOtherBranch(FormModel):
    """Another intrinsic tag remains distinct from dynamic choices."""

    kind: Literal[3] = 3
    other: float = 4.0


class SingleTaggedBranchForm(FormModel):
    """A single branch still declares a structural selector, not a value leaf."""

    branch: Annotated[NumericAliasBranch, Field(discriminator="kind")]


@pytest.mark.parametrize("selector", [{"expr": "1"}, {"__complex__": [1.0, 0.0]}])
def test_compatible_restore_drops_selector_envelopes_but_keeps_other_leaves(
    selector: JsonValue,
) -> None:
    env = FormEnv()
    state = FormState(
        FormInput(SingleTaggedBranchForm, {"branch": {"kind": 2, "amount": 3.0}})
    )
    assert state.write_compatible(
        {"branch": {"kind": selector, "amount": 8.0}}, env
    ) == (("branch", "kind"),)
    assert state.input.data == {"branch": {"kind": 2, "amount": 8.0}}
    assert state.result(env) == SingleTaggedBranchForm(
        branch=NumericAliasBranch(kind=2, amount=8.0)
    )


class DiscriminatorChoicesForm(FormModel):
    """The host may restrict selector choices without changing branch identity."""

    branch: Annotated[
        NumericAliasBranch | NumericOtherBranch, Field(discriminator="kind")
    ]

    @classmethod
    def choices(cls, env: FormEnv) -> Mapping[FormPath, tuple[JsonValue, ...]]:
        """Allow one alias or no tags using only the captured context."""
        return {
            ("branch", "kind"): ()
            if (
                env.parameters is not None
                and env.parameters.lookup("general.ext.empty_tags").value
            )
            else (2,)
        }


def test_discriminator_identity_survives_dynamic_choices_refresh() -> None:
    env = FormEnv()
    state = FormState(FormInput(DiscriminatorChoicesForm, {"branch": {"kind": 2}}))
    before = state.input.data
    for empty in (False, True, False):
        env = captured_ext_env(empty_tags=empty)
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
    """Tagged schemas occur in whole leaves alongside reused linear ranges."""

    sequence: list[NumericChoice]
    fixed: tuple[NumericChoice, NumericChoice]
    mapping: dict[str, NumericChoice]
    restricted: LinRange
    unrestricted: LinRange

    @classmethod
    def choices(cls, env: FormEnv) -> Mapping[FormPath, tuple[JsonValue, ...]]:
        """Refresh whole-leaf membership and only one range occurrence's n."""
        empty = (
            env.parameters is not None
            and env.parameters.lookup("general.ext.empty_containers").value
        )
        data = identity_input()
        return {
            ("sequence",): () if empty else (data["sequence"],),
            ("fixed",): () if empty else (data["fixed"],),
            ("mapping",): () if empty else (data["mapping"],),
            ("restricted", "n"): (3,)
            if (
                env.parameters is not None
                and env.parameters.lookup("general.ext.three_only").value
            )
            else (2,),
        }


def test_container_discriminator_identity_and_retained_discriminator_identity_refresh() -> (
    None
):
    env = FormEnv()
    state = FormState(FormInput(SchemaIdentityForm, identity_input()))
    before = state.input.data
    for empty, three_only in (
        (False, False),
        (True, False),
        (False, True),
        (False, False),
    ):
        env = captured_ext_env(empty_containers=empty, three_only=three_only)
        allowed_n = 3 if three_only else 2
        count = state.describe(("restricted", "n"), env).target
        assert count is not None and count.choices == (allowed_n,)
        candidate: dict[str, JsonValue] = {
            **before,
            "restricted": {"kind": "linrange", "n": allowed_n},
        }
        if not empty:
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
    env = FormEnv()
    for tag in (1, 2, "1"):
        data: dict[str, JsonValue] = {"branch": {"kind": tag}}
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
    env = FormEnv()
    for tag in (1, 2):
        data: dict[str, JsonValue] = {"branch": {"kind": tag, "amount": 7.0}}
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
        if (
            env.parameters is not None
            and env.parameters.lookup("general.ext.empty_amounts").value
        ):
            return {("amount",): ()}
        return {
            ("amount",): (9.0,)
            if (
                env.parameters is not None
                and env.parameters.lookup("general.ext.nine_only").value
            )
            else (7.0,)
        }


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
def test_literal_tag_declaration_keeps_result_agreement(
    model: type[AliasTagForm | UnionTagForm],
) -> None:
    env = FormEnv()
    state = FormState(FormInput(model, {"branch": {"kind": 2, "amount": 7.0}}))
    before = state.input.data
    for empty, nine_only in (
        (False, False),
        (True, False),
        (False, True),
        (False, False),
    ):
        env = captured_ext_env(empty_amounts=empty, nine_only=nine_only)
        for tag in (1, 2):
            for amount in (7.0, 9.0):
                data: dict[str, JsonValue] = {"branch": {"kind": tag, "amount": amount}}
                accepted = not empty and amount == (9.0 if nine_only else 7.0)
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
    env = FormEnv()
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
        env = captured_ext_env(empty_amounts=empty, nine_only=nine_only)
        allowed = 9.0 if nine_only else 7.0
        for tag in (1, 2):
            for amount in (7.0, 9.0):
                data: dict[str, JsonValue] = {"branch": {"kind": tag, "amount": amount}}
                accepted = not empty and amount == allowed
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
        string_result = string_state.result(env)
        assert isinstance(string_result, model)
        assert isinstance(string_result.branch, StringNumericTagBranch)
        assert string_result.branch.kind == "1"
        assert type(string_result.branch.kind) is str
        assert string_result.branch.text_only == 4.0
        assert string_state.input.data == string_before
        current = state.result(env)
        if empty or nine_only:
            assert isinstance(current, FormErrors)
            assert ("branch", "amount") in current.messages
        else:
            assert isinstance(current, model)
            assert isinstance(current.branch, RetainedChoiceWithChoices)
            assert current.branch.kind == 2
        assert state.input.data == before
