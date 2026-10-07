"""Draft editing, validation and comparison through FormState."""

from collections.abc import Mapping
from typing import Annotated, Literal, Self

import pytest
from pydantic import BaseModel, Field, JsonValue, model_validator
from zcu_tools.gui.form import (
    ExplicitList,
    FormEnv,
    FormErrors,
    FormInput,
    FormModel,
    FormPath,
    FormState,
    FormWritePath,
    LinRange,
    RangeInput,
    field_error,
)
from zcu_tools.resources.context import MetaDict


class PlainSection(BaseModel):
    """Nested ordinary Pydantic models need no form hooks."""

    level: float = 2.0
    required: float


class DraftForm(FormModel):
    """Input fields cover scalar, nullable, sequence and literal mapping keys."""

    amount: float = Field(default=2.0, gt=0)
    optional: float | None = None
    amplitude: complex = 1 + 0j
    samples: tuple[float, ...] = (1.0,)
    offsets: dict[str, float] = Field(default_factory=dict)


class Bounds(FormModel):
    """A cross-field validator directs one error and leaves others model-wide."""

    start: float = 1.0
    stop: float = 2.0

    @model_validator(mode="after")
    def validate_order(self) -> Self:
        """Attribute ordering failure to stop within this model."""
        if self.stop <= self.start:
            raise field_error(("stop",), "Stop must exceed start")
        if self.start + self.stop > 100:
            raise ValueError("Total bounds exceed the model-wide limit")
        return self


class BoundsForm(FormModel):
    """Wrap a cross-field model to observe parent-path prefixing."""

    bounds: Bounds = Field(default_factory=Bounds)


class StaticBranch(FormModel):
    """A branch without a factory keeps required fields absent."""

    kind: Literal["static"] = "static"
    shared: float = 1.0
    static_only: int = 7


class ContextBranch(FormModel):
    """A branch factory reads captured context and previous input."""

    kind: Literal["context"] = "context"
    shared: float = 9.0
    context_only: float = 3.0
    required: float

    @classmethod
    def defaults(
        cls, env: FormEnv, previous: Mapping[str, JsonValue]
    ) -> dict[str, JsonValue]:
        """Seed the new branch before core carries same-name fields."""
        return {
            "kind": "context",
            "shared": 9.0,
            "context_only": env.md.get("seed", 3.0),
            "required": previous.get("static_only", 0),
        }


class RequiredBranch(FormModel):
    """Static fallback has a deliberately unfilled required field."""

    kind: Literal["required"] = "required"
    shared: float = 1.0
    required: float


class BranchForm(FormModel):
    """Declare discriminator switching and the shared range submodels."""

    branch: Annotated[
        StaticBranch | ContextBranch | RequiredBranch, Field(discriminator="kind")
    ] = Field(default_factory=StaticBranch)
    fluxes: Annotated[LinRange | ExplicitList, Field(discriminator="kind")] = Field(
        default_factory=LinRange
    )


class NullableBranchForm(FormModel):
    """Nullable tagged objects require activation before child editing."""

    branch: Annotated[
        StaticBranch | ContextBranch | None, Field(discriminator="kind")
    ] = Field(default_factory=StaticBranch)


class PlainForm(FormModel):
    """Use a nested BaseModel without replacing its owner."""

    section: PlainSection


class NullableForm(FormModel):
    """Optional sections retain the distinction between null and nested input."""

    section: Bounds | None = None


class ChoiceForm(FormModel):
    """Dynamic choices use the environment supplied to each result call."""

    mode: Literal["alpha", "beta"] = "alpha"

    @classmethod
    def choices(cls, env: FormEnv) -> Mapping[FormPath, tuple[JsonValue, ...]]:
        """Permit beta only in the supplied captured context."""
        if env.md.get("empty_choices"):
            return {("mode",): ()}
        return {
            ("mode",): ("alpha", "beta") if env.md.get("allow_beta") else ("alpha",)
        }


def environment(seed: float = 4.0) -> FormEnv:
    """Make isolated memory-only sources for this test's calls."""
    md = MetaDict()
    md.update({"seed": seed})
    return FormEnv(md=md.snapshot())


def test_static_defaults_leave_required_keys_absent_and_keep_nullable_null() -> None:
    env = environment()
    assert PlainForm.default_input(env).data == {}
    partial = PlainForm.defaults(env, {})
    assert "section" not in partial
    nullable = NullableForm.default_input(env)
    assert nullable.data == {"section": None}
    assert isinstance(FormState(nullable).result(env), NullableForm)


def test_absent_non_nullable_nested_model_accepts_child_leaf_writes() -> None:
    env = environment()
    state = FormState(PlainForm.default_input(env))
    state.write({("section", "level"): 5.0}, env)
    assert state.input.data == {"section": {"level": 5.0}}
    incomplete = state.result(env)
    assert isinstance(incomplete, FormErrors)
    assert ("section", "required") in incomplete.messages
    state.write({("section", "required"): 3.0}, env)
    complete = state.result(env)
    assert isinstance(complete, PlainForm)
    assert complete.section == PlainSection(level=5.0, required=3.0)

    defaults = FormState(PlainForm.default_input(env))
    defaults.write({("section", "required"): 3.0}, env)
    assert defaults.input.data == {"section": {"level": 2.0, "required": 3.0}}
    result = defaults.result(env)
    assert isinstance(result, PlainForm)
    assert result.section == PlainSection(level=2.0, required=3.0)


def test_nullable_section_activation_child_edit_and_deactivation() -> None:
    env = environment()
    state = FormState(NullableForm.default_input(env))
    with pytest.raises(ValueError, match="section|object|null|path"):
        state.write({("section", "stop"): 5.0}, env)
    assert state.input.data == {"section": None}
    state.write({("section",): {"start": 2.0, "stop": 4.0}}, env)
    active = state.result(env)
    assert isinstance(active, NullableForm)
    assert active.section == Bounds(start=2.0, stop=4.0)
    state.write({("section", "stop"): 5.0}, env)
    edited = state.result(env)
    assert isinstance(edited, NullableForm)
    assert edited.section == Bounds(start=2.0, stop=5.0)
    state.write({("section",): None}, env)
    inactive = state.result(env)
    assert isinstance(inactive, NullableForm)
    assert inactive.section is None
    assert state.input.data["section"] is None


def test_result_evaluates_input_against_each_env_without_replacing_original() -> None:
    state = FormState(
        FormInput(
            DraftForm,
            {
                "amount": {"expr": "seed * 2"},
                "optional": None,
                "amplitude": {"__complex__": [1.0, -2.0]},
                "samples": [1.0, {"expr": "seed"}],
                "offsets": {"rack.left": {"expr": "seed + 1"}},
            },
        )
    )
    first_env = environment(4.0)
    before_md = dict(first_env.md.items())
    first = state.result(first_env)
    second = state.result(environment(6.0))
    assert isinstance(first, DraftForm)
    assert isinstance(second, DraftForm)
    assert (first.amount, second.amount) == (8.0, 12.0)
    assert first.optional is None
    assert first.amplitude == 1 - 2j
    assert first.samples == (1.0, 4.0)
    assert first.offsets == {"rack.left": 5.0}
    assert state.input.data["amount"] == {"expr": "seed * 2"}
    assert dict(first_env.md.items()) == before_md


def test_write_changes_and_reset_preserve_input_value_and_whole_sequence() -> None:
    env = environment()
    base = DraftForm.default_input(env)
    state = FormState(base)
    observed: list[dict[str, JsonValue]] = []
    unsubscribe = state.subscribe(lambda: observed.append(state.input.data))
    batch: dict[FormWritePath, JsonValue] = {
        ("amount",): {"expr": "seed / 2"},
        ("samples",): [2.0, 3.0],
        ("offsets",): {"rack.left": {"expr": "seed"}, "expr": 1.0},
    }
    state.write(batch, env)
    batch[("samples",)] = []

    changes = state.changes(env)
    assert not isinstance(changes, FormErrors)
    by_path = {change.path: change for change in changes}
    assert set(by_path) == {("amount",), ("samples",), ("offsets",)}
    assert by_path[("amount",)].input == {"expr": "seed / 2"}
    assert by_path[("amount",)].value == 2.0
    assert by_path[("amount",)].original == "seed / 2"
    assert by_path[("samples",)].input == [2.0, 3.0]
    assert by_path[("samples",)].value == (2.0, 3.0)
    assert by_path[("samples",)].original is None
    assert by_path[("offsets",)].input == {
        "rack.left": {"expr": "seed"},
        "expr": 1.0,
    }
    assert by_path[("offsets",)].value == {"rack.left": 4.0, "expr": 1.0}
    assert by_path[("offsets",)].original is None
    detached_mapping = by_path[("offsets",)].input
    assert isinstance(detached_mapping, dict)
    detached_mapping.clear()
    assert state.input.data["offsets"] == {
        "rack.left": {"expr": "seed"},
        "expr": 1.0,
    }
    assert state.base.data == base.data
    assert len(observed) == 1
    assert observed[0] == state.input.data
    state.write({("amount",): {"expr": "seed / 2"}}, env)
    assert len(observed) == 1

    exposed = state.input.data
    exposed["amount"] = 1000.0
    assert state.input.data["amount"] == {"expr": "seed / 2"}
    state.reset(base)
    assert state.changes(env) == ()
    assert state.input.data == base.data
    assert len(observed) == 2
    unsubscribe()
    unsubscribe()
    state.write({("amount",): 3.0}, env)
    assert len(observed) == 2


@pytest.mark.parametrize(
    ("value", "path"),
    [
        (-1.0, ("amount",)),
        ({"expr": "unknown_seed"}, ("amount",)),
        ({"expr": 3}, ("amount",)),
    ],
)
def test_invalid_drafts_remain_editable_and_never_yield_partial_changes(
    value: JsonValue, path: FormWritePath
) -> None:
    env = environment()
    state = FormState(DraftForm.default_input(env))
    state.write({path: value, ("samples",): [3.0]}, env)
    assert state.input.data["amount"] == value
    result = state.result(env)
    changes = state.changes(env)
    assert isinstance(result, FormErrors)
    assert isinstance(changes, FormErrors)
    assert path in result.messages
    assert changes.messages == result.messages
    state.write({("amount",): 4.0}, env)
    assert isinstance(state.result(env), DraftForm)


def test_error_paths_distinguish_missing_null_nested_index_and_model_wide() -> None:
    env = environment()
    missing = FormState(FormInput(PlainForm, {"section": {"level": 1.0}})).result(env)
    null = FormState(FormInput(PlainForm, {"section": {"required": None}})).result(env)
    assert isinstance(missing, FormErrors)
    assert isinstance(null, FormErrors)
    assert ("section", "required") in missing.messages
    assert ("section", "required") in null.messages
    assert missing.messages != null.messages

    state = FormState(BoundsForm.default_input(env))
    state.write({("bounds", "stop"): 0.5}, env)
    directed = state.result(env)
    assert isinstance(directed, FormErrors)
    assert set(directed.messages) == {("bounds", "stop")}
    state.write({("bounds", "start"): 60.0, ("bounds", "stop"): 70.0}, env)
    undirected = state.result(env)
    assert isinstance(undirected, FormErrors)
    assert set(undirected.messages) == {("bounds",)}

    indexed = FormState(FormInput(DraftForm, {"samples": [1.0, "bad"]})).result(env)
    assert isinstance(indexed, FormErrors)
    assert ("samples", 1) in indexed.messages


def test_root_model_wide_error_uses_empty_path() -> None:
    env = environment()
    result = FormState(FormInput(Bounds, {"start": 60.0, "stop": 70.0})).result(env)
    assert isinstance(result, FormErrors)
    assert set(result.messages) == {()}


@pytest.mark.parametrize(
    "batch",
    [
        {("amount",): 3.0, ("unknown",): 4.0},
        {("amount",): 3.0, ("samples", "0"): 4.0},
        {("offsets",): {}, ("offsets", "rack.left"): 4.0},
    ],
)
def test_structural_write_failure_does_not_publish_a_partial_batch(
    batch: Mapping[FormWritePath, JsonValue],
) -> None:
    env = environment()
    state = FormState(DraftForm.default_input(env))
    before = state.input.data
    notifications: list[int] = []
    state.subscribe(lambda: notifications.append(1))
    with pytest.raises(ValueError, match="path|Path|unknown|Unknown|overlap|sequence"):
        state.write(batch, env)
    assert state.input.data == before
    assert state.base.data == before
    assert notifications == []


def test_mapping_replacement_is_a_leaf_but_key_and_model_patches_are_rejected() -> None:
    env = environment()
    mapping = FormState(
        FormInput(DraftForm, {"offsets": {"rack.left": 3.0, "remove": 1.0}})
    )
    before = mapping.input.data
    with pytest.raises(ValueError, match="leaf|mapping|replacement|path"):
        mapping.write({("amount",): 4.0, ("offsets", "rack.left"): 5.0}, env)
    assert mapping.input.data == before

    mapping.write({("offsets",): {"renamed": 3.0}}, env)
    result = mapping.result(env)
    assert isinstance(result, DraftForm)
    assert result.offsets == {"renamed": 3.0}
    changes = mapping.changes(env)
    assert not isinstance(changes, FormErrors)
    assert len(changes) == 1
    assert changes[0].path == ("offsets",)
    assert changes[0].input == {"renamed": 3.0}
    assert changes[0].value == {"renamed": 3.0}
    assert changes[0].original is None

    mapping.write({("offsets",): {}}, env)
    cleared = mapping.result(env)
    assert isinstance(cleared, DraftForm)
    assert cleared.offsets == {}
    changes = mapping.changes(env)
    assert not isinstance(changes, FormErrors)
    assert len(changes) == 1
    assert changes[0].path == ("offsets",)
    assert changes[0].input == {}
    assert changes[0].value == {}

    model = FormState(BoundsForm.default_input(env))
    before_model = model.input.data
    with pytest.raises(ValueError, match="leaf|model|replacement|path"):
        model.write({("bounds",): {"start": 2.0, "stop": 4.0}}, env)
    assert model.input.data == before_model


def test_changes_describe_active_branch_leaves_and_nullable_null() -> None:
    env = environment()
    branch = FormState(BranchForm.default_input(env))
    branch.write({("branch", "kind"): "context"}, env)
    changes = branch.changes(env)
    assert not isinstance(changes, FormErrors)
    by_path = {change.path: change for change in changes}
    assert set(by_path) == {
        ("branch", "kind"),
        ("branch", "context_only"),
        ("branch", "required"),
    }
    assert by_path[("branch", "kind")].value == "context"
    assert by_path[("branch", "required")].value == 7.0

    nullable = FormState(
        FormInput(NullableForm, {"section": {"start": 2.0, "stop": 4.0}})
    )
    nullable.write({("section",): None}, env)
    changes = nullable.changes(env)
    assert not isinstance(changes, FormErrors)
    assert len(changes) == 1
    assert changes[0].path == ("section",)
    assert changes[0].input is None
    assert changes[0].value is None
    assert changes[0].original is None


def test_discriminator_first_factory_inheritance_and_explicit_content_order() -> None:
    env = environment(8.0)
    state = FormState(BranchForm.default_input(env))
    state.write({("branch", "shared"): 5.0}, env)
    notifications: list[dict[str, JsonValue]] = []
    state.subscribe(lambda: notifications.append(state.input.data))
    # Non-discriminator content deliberately precedes kind in the batch.
    state.write({("branch", "context_only"): 12.0, ("branch", "kind"): "context"}, env)
    result = state.result(env)
    assert isinstance(result, BranchForm)
    assert isinstance(result.branch, ContextBranch)
    assert result.branch.shared == 5.0
    assert result.branch.context_only == 12.0
    assert result.branch.required == 7.0
    assert len(notifications) == 1

    state.write({("branch", "kind"): "static"}, env)
    state.write({("branch", "kind"): "context"}, env)
    result = state.result(env)
    assert isinstance(result, BranchForm)
    assert isinstance(result.branch, ContextBranch)
    assert result.branch.context_only == 8.0
    assert result.branch.shared == 5.0

    state.write({("branch", "kind"): "static"}, env)
    state.write({("branch", "kind"): "required"}, env)
    result = state.result(env)
    assert isinstance(result, FormErrors)
    assert ("branch", "required") in result.messages


def test_range_input_persists_as_range_and_switches_to_whole_explicit_list() -> None:
    env = environment()
    state = FormState(BranchForm.default_input(env))
    state.write(
        {("fluxes", "start"): 0.3, ("fluxes", "stop"): 0.5, ("fluxes", "n"): 3}, env
    )
    saved = FormInput.from_json(BranchForm, state.input.to_json())
    assert saved.data["fluxes"] == {
        "kind": "linrange",
        "start": 0.3,
        "stop": 0.5,
        "n": 3,
    }
    state.write({("fluxes", "values"): [0.5, 0.3], ("fluxes", "kind"): "list"}, env)
    result = state.result(env)
    assert isinstance(result, BranchForm)
    assert isinstance(result.fluxes, ExplicitList)
    assert result.fluxes.values == (0.5, 0.3)
    state.write({("fluxes", "kind"): "linrange"}, env)
    result = state.result(env)
    assert isinstance(result, BranchForm)
    assert isinstance(result.fluxes, LinRange)
    assert result.fluxes.n == LinRange().n


def test_dynamic_choices_revalidate_with_the_supplied_context() -> None:
    allowed_md = MetaDict()
    allowed_md.update({"allow_beta": True})
    allowed = FormEnv(md=allowed_md.snapshot())
    denied = FormEnv(md=MetaDict())
    state = FormState(ChoiceForm.default_input(allowed))
    state.write({("mode",): "beta"}, allowed)
    assert isinstance(state.result(allowed), ChoiceForm)
    rejected = state.result(denied)
    assert isinstance(rejected, FormErrors)
    assert ("mode",) in rejected.messages
    assert state.input.data["mode"] == "beta"


def test_empty_dynamic_choices_reject_every_present_value() -> None:
    md = MetaDict()
    md.update({"empty_choices": True})
    env = FormEnv(md=md.snapshot())
    state = FormState(FormInput(ChoiceForm, {"mode": "alpha"}))
    result = state.result(env)
    assert isinstance(result, FormErrors)
    assert ("mode",) in result.messages


def test_compatible_restore_applies_branch_discriminator_before_saved_leaves() -> None:
    env = environment()
    state = FormState(BranchForm.default_input(env))
    dropped = state.write_compatible(
        {
            "branch": {"shared": 6.0, "required": 12.0, "kind": "context"},
            "fluxes": {"values": [0.3, 0.4], "kind": "list"},
        },
        env,
    )
    assert dropped == ()
    result = state.result(env)
    assert isinstance(result, BranchForm)
    assert result.branch == ContextBranch(shared=6.0, required=12.0, context_only=4.0)
    assert result.fluxes == ExplicitList(values=(0.3, 0.4))


@pytest.mark.parametrize("kind", ["static", "context"])
def test_compatible_restore_without_tag_overlays_current_branch_once(kind: str) -> None:
    env = environment()
    state = FormState(BranchForm.default_input(env))
    state.write({("branch", "kind"): kind}, env)
    before = state.input.data
    current = before["branch"]
    assert isinstance(current, dict)
    notifications: list[dict[str, JsonValue]] = []
    state.subscribe(lambda: notifications.append(state.input.data))
    assert state.write_compatible({"branch": {}}, env) == ()
    assert state.input.data == before
    assert notifications == []
    branch_only = "static_only" if kind == "static" else "context_only"
    saved: dict[str, JsonValue] = {
        "shared": 3.0,
        branch_only: "bad",
        "removed": 9,
    }
    dropped = state.write_compatible({"branch": saved}, env)
    assert set(dropped) == {("branch", branch_only), ("branch", "removed")}
    restored = state.input.data
    assert restored == {**before, "branch": {**current, "shared": 3.0}}
    assert notifications == [restored]
    result = state.result(env)
    assert isinstance(result, BranchForm)
    assert result.branch.kind == kind
    assert result.branch.shared == 3.0


@pytest.mark.parametrize("current", [{}, {"kind": "unknown"}])
@pytest.mark.parametrize("saved", [{}, {"shared": 3.0}])
def test_compatible_restore_cannot_replace_an_unselected_model_object(
    current: dict[str, JsonValue], saved: dict[str, JsonValue]
) -> None:
    env = environment()
    state = FormState(FormInput(BranchForm, {"branch": current}))
    before = state.input.data
    notifications: list[dict[str, JsonValue]] = []
    state.subscribe(lambda: notifications.append(state.input.data))
    assert state.write_compatible({"branch": saved}, env) == (("branch",),)
    assert state.input.data == before
    assert notifications == []
    result = state.result(env)
    assert isinstance(result, FormErrors)
    assert ("branch",) in result.messages


class NullableRangeAliasForm(FormModel):
    """Expose the public range alias as a nullable active model section."""

    fluxes: RangeInput | None = Field(default_factory=LinRange)


def test_nullable_range_alias_evaluates_expressions_and_canonical_errors() -> None:
    env = environment()
    data: dict[str, JsonValue] = {
        "fluxes": {"kind": "linrange", "start": {"expr": "seed"}, "stop": 5.0, "n": 2}
    }
    state = FormState(FormInput(NullableRangeAliasForm, data))
    result = state.result(env)
    assert isinstance(result, NullableRangeAliasForm)
    assert isinstance(result.fluxes, LinRange)
    assert result.fluxes.start == 4.0
    assert state.input.data == data
    invalid: dict[str, JsonValue] = {
        "fluxes": {"kind": "linrange", "n": {"expr": "1.5"}}
    }
    errors = FormState(FormInput(NullableRangeAliasForm, invalid)).result(env)
    assert isinstance(errors, FormErrors)
    assert ("fluxes", "n") in errors.messages
    assert () not in errors.messages


def test_nullable_range_alias_edits_and_restores_the_current_branch() -> None:
    env = environment()
    state = FormState(NullableRangeAliasForm.default_input(env))
    state.write(
        {
            ("fluxes", "start"): {"expr": "seed"},
            ("fluxes", "stop"): 5.0,
            ("fluxes", "n"): 2,
        },
        env,
    )
    prior = state.input.data["fluxes"]
    assert isinstance(prior, dict)
    assert state.write_compatible({"fluxes": {"n": 3}}, env) == ()
    assert state.input.data["fluxes"] == {**prior, "n": 3}
    result = state.result(env)
    assert isinstance(result, NullableRangeAliasForm)
    assert isinstance(result.fluxes, LinRange)
    assert result.fluxes.start == 4.0
    assert result.fluxes.n == 3


def test_null_tagged_section_rejects_discriminator_child_write_until_activation() -> (
    None
):
    env = environment()
    state = FormState(NullableBranchForm.default_input(env))
    state.write({("branch",): None}, env)
    before = state.input.data
    base = state.base.data
    notifications: list[dict[str, JsonValue]] = []
    state.subscribe(lambda: notifications.append(state.input.data))
    with pytest.raises(ValueError, match="null|activat"):
        state.write({("branch", "kind"): "context"}, env)
    assert state.input.data == before
    assert state.base.data == base
    assert notifications == []
    state.write({("branch",): {"kind": "static"}}, env)
    state.write({("branch", "kind"): "context"}, env)
    result = state.result(env)
    assert isinstance(result, NullableBranchForm)
    assert isinstance(result.branch, ContextBranch)
    assert result.branch.kind == "context"
    assert len(notifications) == 2


def test_compatible_restore_preserves_nullable_branch_deactivation() -> None:
    env = environment()
    state = FormState(NullableBranchForm.default_input(env))
    notifications: list[dict[str, JsonValue]] = []
    state.subscribe(lambda: notifications.append(state.input.data))
    assert state.write_compatible({"branch": None}, env) == ()
    assert notifications == [{"branch": None}]
    result = state.result(env)
    assert isinstance(result, NullableBranchForm)
    assert result.branch is None
    assert state.write_compatible({"branch": {}}, env) == (("branch",),)
    assert state.input.data == {"branch": None}
    assert notifications == [{"branch": None}]


def test_compatible_restore_keeps_well_formed_unresolved_expressions() -> None:
    env = environment()
    state = FormState(DraftForm.default_input(env))
    dropped = state.write_compatible({"amount": {"expr": "missing"}}, env)
    assert dropped == ()
    assert state.input.data["amount"] == {"expr": "missing"}
    result = state.result(env)
    assert isinstance(result, FormErrors)
    assert ("amount",) in result.messages


def test_compatible_restore_drops_types_not_range_or_cross_field_errors() -> None:
    env = environment()
    state = FormState(BoundsForm.default_input(env))
    dropped = state.write_compatible(
        {"bounds": {"start": 3.0, "stop": 2.0, "removed": 9}, "old_root": 1}, env
    )
    assert set(dropped) == {("bounds", "removed"), ("old_root",)}
    result = state.result(env)
    assert isinstance(result, FormErrors)
    assert ("bounds", "stop") in result.messages

    scalar = FormState(DraftForm.default_input(env))
    dropped = scalar.write_compatible(
        {
            "amount": -1.0,
            "samples": [1.0, "bad"],
            "optional": None,
            "amplitude": {"__complex__": [2.0, 3.0]},
            "offsets": {"rack.left": {"expr": "seed"}},
        },
        env,
    )
    assert set(dropped) == {("samples",)}
    assert scalar.input.data["samples"] == DraftForm.default_input(env).data["samples"]
    assert scalar.input.data["amount"] == -1.0
    assert scalar.input.data["optional"] is None
    assert scalar.input.data["amplitude"] == {"__complex__": [2.0, 3.0]}
    errors = scalar.result(env)
    assert isinstance(errors, FormErrors)
    assert ("amount",) in errors.messages


def test_reset_rejects_a_different_model_without_changing_state() -> None:
    env = environment()
    state = FormState[FormModel](DraftForm.default_input(env))
    before = state.input.data
    with pytest.raises(ValueError, match="model|Model|class|type"):
        state.reset(ChoiceForm.default_input(env))
    assert state.input.data == before
    assert state.base.data == before


def test_callback_failure_propagates_after_installation_without_rollback() -> None:
    env = environment()
    state = FormState(DraftForm.default_input(env))

    def reject_notification() -> None:
        """Represent a host callback failure, not a form validation failure."""
        raise RuntimeError("observer failed")

    unsubscribe = state.subscribe(reject_notification)
    with pytest.raises(RuntimeError, match="observer failed"):
        state.write({("amount",): 3.0}, env)
    assert state.input.data["amount"] == 3.0
    unsubscribe()
    state.write({("amount",): 4.0}, env)
    assert state.input.data["amount"] == 4.0
