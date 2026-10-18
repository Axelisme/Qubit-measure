"""Generic option selection, projection and commands at public form seams."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Literal

import pytest
from pydantic import BaseModel, Field, JsonValue
from pydantic_core import PydanticCustomError
from zcu_tools.gui import form as api
from zcu_tools.gui.form import (
    FormEnv,
    FormErrors,
    FormInput,
    FormModel,
    FormPath,
    FormState,
)

from tests.gui._form_converters import RebuildingForm


class Speed(StrEnum):
    slow = "slow"
    fast = "fast"


class First(FormModel):
    kind: Literal["first"] = "first"
    shared: float = 2.0


class Second(FormModel):
    kind: Literal["second"] = "second"
    shared: float = 7.0
    required: float

    @classmethod
    def defaults(cls, env: FormEnv) -> dict[str, JsonValue]:
        del env
        return {"kind": "second", "shared": {"expr": "2 * 4"}}


class Picks(RebuildingForm):
    branch_models = {("branch",): (First, Second)}

    count: Literal[1, 2] = 1
    speed: Speed = Speed.slow
    device: str = "a"
    flag: bool = False
    enabled: First | None = None
    branch: First | Second = Field(default_factory=First, discriminator="kind")

    @classmethod
    def choices(cls, env: FormEnv) -> Mapping[FormPath, tuple[JsonValue, ...]]:
        return {("device",): tuple(env.devices) or ("a", "b")}


@pytest.mark.parametrize("dynamic", [False, True], ids=["static-C", "missing-old-key"])
def test_saved_unlisted_selection_is_visible_invalid_input_not_a_restore_failure(
    dynamic: bool,
) -> None:
    hooks_enabled = False

    class HookSelection(FormModel):
        key: str

        @classmethod
        def options(cls, env: FormEnv) -> Mapping[FormPath, tuple[api.Option, ...]]:
            assert hooks_enabled, "Restore must not execute declaration hooks"
            keys = (
                (() if env.library is None else env.library.module_keys())
                if dynamic
                else ("A", "B")
            )
            return {("key",): tuple(api.Option(key, key) for key in keys)}

    selection = "old-key" if dynamic else "C"
    authored = FormInput(HookSelection, {"key": "A"}, selections={("key",): selection})
    restored = FormInput.from_json(HookSelection, authored.to_json())
    hooks_enabled = True
    state = FormState(restored)
    errors = state.result(FormEnv())
    assert isinstance(errors, FormErrors)
    assert ("key",) in errors.messages
    assert "option" in errors.messages[("key",)].lower()
    assert state.preview(FormEnv()).errors == errors
    assert state.input.same_as(authored)


def test_hook_options_do_not_restrict_raw_value_without_selection_override() -> None:
    class Ref(BaseModel):
        kind: Literal["ref"] = "ref"
        name: str

    class RefForm(FormModel):
        source: Ref

        @classmethod
        def options(cls, _env: FormEnv) -> Mapping[FormPath, tuple[api.Option, ...]]:
            return {
                ("source",): (api.Option("Listed", {"kind": "ref", "name": "listed"}),)
            }

    raw: dict[str, JsonValue] = {"source": {"kind": "ref", "name": "unlisted"}}
    ordinary = FormState(FormInput(RefForm, raw))
    assert isinstance(ordinary.result(FormEnv()), RefForm)
    explicit = FormState(
        FormInput(RefForm, raw, selections={("source",): raw["source"]})
    )
    errors = explicit.result(FormEnv())
    assert isinstance(errors, FormErrors)
    assert ("source",) in errors.messages
    assert explicit.input.data == raw


def test_reselecting_scalar_replaces_stale_override_and_tracks_new_choice() -> None:
    listed = ["A", "B"]

    class ChangingChoices(FormModel):
        key: str

        @classmethod
        def options(cls, _env: FormEnv) -> Mapping[FormPath, tuple[api.Option, ...]]:
            return {("key",): tuple(api.Option(key, key) for key in listed)}

    env = FormEnv()
    state = FormState(
        FormInput(ChangingChoices, {"key": "A"}, selections={("key",): "C"})
    )
    assert isinstance(state.result(env), FormErrors)
    state.select(("key",), "B", env)
    assert isinstance(state.result(env), ChangingChoices)
    assert state.input.data == {"key": "B"}
    assert state.input.selections == {("key",): "B"}
    listed.remove("B")
    errors = state.result(env)
    assert isinstance(errors, FormErrors)
    assert ("key",) in errors.messages
    assert state.input.data == {"key": "B"}


def obj(value: JsonValue) -> dict[str, JsonValue]:
    assert isinstance(value, dict)
    return value


def test_option_matches_typed_leaves_first_match_and_missing_without_mutation() -> None:
    assert hasattr(api, "Option"), "core must expose named Option"
    Option = api.Option
    options = (
        Option("partial", {"kind": "first"}),
        Option("specific", {"kind": "first", "shared": 2}),
    )
    state = FormState(Picks.default_input(FormEnv()))
    current = state.input.data["branch"]
    assert (
        next(option.label for option in options if option.matches(current)) == "partial"
    )
    assert Option("null", None).matches(None)
    assert Option("number", 2).matches(2)
    assert not Option("number", 2).matches("2")
    assert not Option("number", 1).matches(True)
    assert not Option("null", None).matches({})
    assert not any(option.matches({"kind": "missing"}) for option in options)
    assert state.input.data["branch"] == current
    source: dict[str, JsonValue] = {"kind": "first"}
    detached = Option("detached", source)
    source["kind"] = "second"
    assert detached.matches(current)
    exported = detached.value
    assert isinstance(exported, dict)
    exported["kind"] = "second"
    assert detached.matches(current)
    with pytest.raises(ValueError, match="finite"):
        Option("bad", float("inf"))
    with pytest.raises(ValueError, match="fields"):
        Option("bad fields", "key", fields=First)


def test_branch_selection_starts_from_fresh_defaults_and_commits_once() -> None:
    env = FormEnv()
    state = FormState(
        FormInput(Picks, {"branch": {"kind": "first", "shared": {"expr": "99"}}})
    )
    observed: list[dict[str, JsonValue]] = []
    state.subscribe(lambda: observed.append(state.input.data))
    state.select(("branch",), {"kind": "second"}, env)
    assert state.input.data == {"branch": {"kind": "second", "shared": 8}}
    assert observed == [state.input.data]
    result = state.result(env)
    assert isinstance(result, FormErrors)
    assert result.messages == {("branch", "required"): "Field required"}
    state.write({("branch",): {"kind": "first", "shared": 6}}, env)
    assert state.input.data == {"branch": {"kind": "first", "shared": 6}}
    before = state.input.data
    with pytest.raises(ValueError, match="[Uu]nknown"):
        state.write({("branch",): {"kind": "second", "unknown": 1}}, env)
    assert state.input.data == before
    assert len(observed) == 2


class Source(BaseModel):
    kind: Literal["source"] = "source"
    name: str = "original"


class Destination(BaseModel):
    kind: Literal["snapshot"] = "snapshot"
    name: str = "default"


class CommandForm(FormModel):
    value: Source | Destination = Field(default_factory=Source, discriminator="kind")

    @classmethod
    def options(cls, _env: FormEnv) -> Mapping[FormPath, tuple[api.Option, ...]]:
        def convert(
            _old: api.FormCommandSnapshot, _context: api.FormCommandContext
        ) -> api.FormCommandResult:
            return api.FormCommandResult(
                {"kind": "snapshot"}, Destination, {"name": "copied"}
            )

        def fail(
            _old: api.FormCommandSnapshot, _context: api.FormCommandContext
        ) -> api.FormCommandResult:
            raise ValueError("unavailable source")

        return {
            ("value",): (
                api.Option(
                    "Original",
                    {"kind": "source", "name": "original"},
                    fields=Source,
                    commands=(
                        api.OptionCommand("convert", "Convert", convert),
                        api.OptionCommand("fail", "Fail", fail),
                    ),
                ),
                api.Option("Snapshot", {"kind": "snapshot"}, fields=Destination),
            )
        }


def test_declared_options_project_commands_and_execute_one_atomic_write() -> None:
    assert hasattr(api, "OptionCommand"), "core must expose option command contract"
    env = FormEnv()
    state = FormState(CommandForm.default_input(env))
    target = state.describe(("value",), env).target
    assert target is not None and target.options is not None
    assert [(item.name, item.label) for item in target.options[0].commands] == [
        ("convert", "Convert"),
        ("fail", "Fail"),
    ]
    events: list[dict[str, JsonValue]] = []
    state.subscribe(lambda: events.append(state.input.data))
    with pytest.raises(ValueError, match="unavailable source"):
        state.run_command(("value",), "fail", env)
    assert events == []
    state.run_command(("value",), "convert", env)
    assert state.input.data == {"value": {"kind": "snapshot", "name": "copied"}}
    assert events == [state.input.data]
    with pytest.raises(ValueError, match="command"):
        state.run_command(("value",), "convert", env)
    assert len(events) == 1


class LocalOptions(FormModel):
    empty: Literal[1, 2] = 1
    local: int = 1
    parent: int = 1

    @classmethod
    def options(cls, _env: FormEnv, /) -> Mapping[FormPath, tuple[api.Option, ...]]:
        return {("empty",): (), ("local",): (api.Option("Local", 1),)}


class NestedOptions(FormModel):
    child: LocalOptions = Field(default_factory=LocalOptions)
    sibling: LocalOptions = Field(default_factory=LocalOptions)

    @classmethod
    def options(cls, _env: FormEnv, /) -> Mapping[FormPath, tuple[api.Option, ...]]:
        return {
            ("child", "empty"): (api.Option("Parent empty", 2),),
            ("child", "local"): (api.Option("Parent local", 2),),
            ("child", "parent"): (api.Option("Parent retained", 1),),
            ("sibling", "parent"): (api.Option("Sibling", 2),),
        }


def test_nested_options_keep_deepest_empty_and_nonoverlapping_parent_declarations() -> (
    None
):
    env = FormEnv()
    state = FormState(NestedOptions.default_input(env))
    child = {item.path[-1]: item for item in state.describe(("child",), env).children}
    sibling = {
        item.path[-1]: item for item in state.describe(("sibling",), env).children
    }
    assert child["empty"].options == ()
    for field, label, value in (
        (child["local"], "Local", 1),
        (child["parent"], "Parent retained", 1),
        (sibling["parent"], "Sibling", 2),
    ):
        assert field.options is not None
        assert [(item.label, item.value) for item in field.options] == [(label, value)]


class ActiveBranch(FormModel):
    kind: Literal["active"] = "active"
    value: int = 1

    @classmethod
    def options(cls, _env: FormEnv, /) -> Mapping[FormPath, tuple[api.Option, ...]]:
        return {
            ("value",): (
                api.Option(
                    "One",
                    1,
                    commands=(
                        api.OptionCommand(
                            "active",
                            "Active",
                            lambda _old, _context: api.FormCommandResult(None, None, 2),
                        ),
                    ),
                ),
            )
        }


class InactiveBranch(FormModel):
    kind: Literal["inactive"] = "inactive"
    value: int = 1

    @classmethod
    def options(cls, _env: FormEnv, /) -> Mapping[FormPath, tuple[api.Option, ...]]:
        return {
            ("value",): (
                api.Option(
                    "One",
                    1,
                    commands=(
                        api.OptionCommand(
                            "inactive",
                            "Inactive",
                            lambda _old, _context: api.FormCommandResult(None, None, 3),
                        ),
                    ),
                ),
            )
        }


class BranchCommands(RebuildingForm):
    branch_models = {("branch",): (ActiveBranch, InactiveBranch)}

    branch: ActiveBranch | InactiveBranch = Field(
        default_factory=ActiveBranch, discriminator="kind"
    )


def test_commands_only_use_the_active_branch_declaration() -> None:
    assert hasattr(api, "OptionCommand"), "core must expose option command contract"
    env = FormEnv()
    state = FormState(BranchCommands.default_input(env))
    before = state.input.data
    with pytest.raises(ValueError, match="command"):
        state.run_command(("branch", "value"), "inactive", env)
    assert state.input.data == before
    state.run_command(("branch", "value"), "active", env)
    assert state.input.data == {"branch": {"kind": "active", "value": 2}}
    state.select(("branch",), {"kind": "inactive"}, env)
    state.run_command(("branch", "value"), "inactive", env)
    assert state.input.data == {"branch": {"kind": "inactive", "value": 3}}


class SharedCommandChild(FormModel):
    value: int = 1


class ActiveSharedBranch(FormModel):
    kind: Literal["active"] = "active"
    child: SharedCommandChild = Field(default_factory=SharedCommandChild)

    @classmethod
    def options(cls, env: FormEnv) -> Mapping[FormPath, tuple[api.Option, ...]]:
        return {("child", "value"): ActiveBranch.options(env)[("value",)]}


class InactiveSharedBranch(FormModel):
    kind: Literal["inactive"] = "inactive"
    child: SharedCommandChild = Field(default_factory=SharedCommandChild)

    @classmethod
    def options(cls, env: FormEnv) -> Mapping[FormPath, tuple[api.Option, ...]]:
        return {("child", "value"): InactiveBranch.options(env)[("value",)]}


class SharedBranchCommands(RebuildingForm):
    branch_models = {("branch",): (ActiveSharedBranch, InactiveSharedBranch)}

    branch: ActiveSharedBranch | InactiveSharedBranch = Field(
        default_factory=ActiveSharedBranch, discriminator="kind"
    )


def test_shared_child_commands_follow_active_ancestor_branch() -> None:
    env = FormEnv()
    state = FormState(SharedBranchCommands.default_input(env))
    before = state.input.data
    events: list[dict[str, JsonValue]] = []
    state.subscribe(lambda: events.append(state.input.data))
    with pytest.raises(ValueError, match="command"):
        state.run_command(("branch", "child", "value"), "inactive", env)
    assert state.input.data == before
    assert state.base.data == before
    assert events == []
    state.run_command(("branch", "child", "value"), "active", env)
    assert state.input.data == {"branch": {"kind": "active", "child": {"value": 2}}}
    assert events == [state.input.data]
    state.select(("branch",), {"kind": "inactive"}, env)
    with pytest.raises(ValueError, match="command"):
        state.run_command(("branch", "child", "value"), "active", env)
    state.run_command(("branch", "child", "value"), "inactive", env)
    assert state.input.data == {"branch": {"kind": "inactive", "child": {"value": 3}}}
    assert len(events) == 3


def test_array_options_match_whole_typed_leaves_not_partial_objects() -> None:
    array = api.Option("Array", [{"id": 1}])
    assert array.matches([{"id": 1}])
    assert not array.matches([{"id": 1, "extra": 2}])
    assert not array.matches([{"id": True}])
    assert not array.matches([{"id": 1.0}])
    assert not array.matches([{"id": 1}, {"id": 2}])
    assert not array.matches({"id": 1})
    assert api.Option("Partial object", {"id": 1}).matches({"id": 1, "extra": 2})
    nested = api.Option("Nested array", {"items": [{"id": 1}]})
    assert nested.matches({"items": [{"id": 1}], "extra": 2})
    assert not nested.matches({"items": [{"id": 1, "extra": 2}]})


class UnknownOptionPath(FormModel):
    value: int = 1

    @classmethod
    def options(cls, _env: FormEnv, /) -> Mapping[FormPath, tuple[api.Option, ...]]:
        return {("missing",): (api.Option("One", 1),)}


def test_unknown_declared_option_path_fails_before_state_changes() -> None:
    assert hasattr(api, "Option"), "core must expose named Option"
    env = FormEnv()
    state = FormState(UnknownOptionPath.default_input(env))
    before = state.input.data
    with pytest.raises(ValueError, match="option.*path"):
        state.describe((), env)
    with pytest.raises(ValueError, match="option.*path"):
        state.run_command(("value",), "unused", env)
    assert state.input.data == before


class NullablePicks(FormModel):
    number: Literal[1, 2] | None = None
    speed: Speed | None = None


def test_nullable_literal_and_enum_options_include_typed_null() -> None:
    env = FormEnv()
    state = FormState(NullablePicks.default_input(env))
    for name, nonnull in (("number", [1, 2]), ("speed", ["slow", "fast"])):
        target = state.describe((name,), env).target
        assert target is not None and target.options is not None
        values = [option.value for option in target.options]
        assert None in values
        assert [value for value in values if value is not None] == nonnull


class UnavailableBranch(FormModel):
    kind: Literal["unavailable"] = "unavailable"
    source: str

    @classmethod
    def defaults(cls, env: FormEnv) -> dict[str, JsonValue]:
        del env
        raise api.field_error(("source",), "Unavailable captured source")


class FailureContainer(RebuildingForm):
    branch_models = {("branch",): (First, UnavailableBranch)}

    branch: First | UnavailableBranch = Field(
        default_factory=First, discriminator="kind"
    )


class SourceFailureForm(FormModel):
    outer: FailureContainer = Field(default_factory=FailureContainer)
    flag: bool = False


def test_selection_source_failure_preserves_parent_path_and_atomicity() -> None:
    env = FormEnv()
    state = FormState(SourceFailureForm.default_input(env))
    before = state.input.data
    events: list[dict[str, JsonValue]] = []
    state.subscribe(lambda: events.append(state.input.data))
    with pytest.raises(
        PydanticCustomError, match="Unavailable captured source"
    ) as caught:
        state.select(("outer", "branch"), {"kind": "unavailable"}, env)
    assert (caught.value.context or {}).get("path") == ("outer", "branch", "source")
    assert state.input.data == before
    assert state.base.data == before
    assert events == []


class AliasedBranch(FormModel):
    kind: Literal["a", "alias"] = "a"
    shared: float = 1.0
    a_only: float = 3.0


class PlainBranch(FormModel):
    kind: Literal["b"] = "b"
    shared: float = 9.0
    b_only: float = 4.0


class StringAliasForm(RebuildingForm):
    branch_models = {("branch",): (AliasedBranch, PlainBranch)}

    branch: AliasedBranch | PlainBranch = Field(discriminator="kind")


class NumericAliasedBranch(FormModel):
    kind: Literal[1, 2] = 1
    shared: float = 1.0
    a_only: float = 3.0


class NumericPlainBranch(FormModel):
    kind: Literal[3] = 3
    shared: float = 9.0
    b_only: float = 4.0


class NumericAliasForm(RebuildingForm):
    branch_models = {("branch",): (NumericAliasedBranch, NumericPlainBranch)}

    branch: NumericAliasedBranch | NumericPlainBranch = Field(discriminator="kind")


@pytest.mark.parametrize(
    ("model", "tags", "alias"),
    [(StringAliasForm, ["a", "alias", "b"], "alias"), (NumericAliasForm, [1, 2, 3], 2)],
    ids=["string", "numeric"],
)
def test_every_intrinsic_tag_is_a_typed_option_sharing_its_branch_fields(
    model: type[FormModel], tags: list[JsonValue], alias: JsonValue
) -> None:
    env = FormEnv()
    state = FormState(FormInput(model, {"branch": {"kind": alias, "shared": 6.0}}))
    description = state.describe(("branch",), env)
    target = description.target
    assert target is not None and target.options is not None
    values = [option.value for option in target.options]
    assert values == [{"kind": tag} for tag in tags]
    assert [type(obj(value)["kind"]) for value in values] == [type(t) for t in tags]
    assert [option.label for option in target.options] == [str(tag) for tag in tags]
    assert target.selected == tags.index(alias)
    assert {child.path[-1] for child in description.children} == {"shared", "a_only"}
    state.select(("branch",), {"kind": tags[-1]}, env)
    assert {child.path[-1] for child in state.describe(("branch",), env).children} == {
        "shared",
        "b_only",
    }
    result = state.result(env)
    assert not isinstance(result, FormErrors)
    assert obj(result.model_dump(mode="json")["branch"])["kind"] == tags[-1]
