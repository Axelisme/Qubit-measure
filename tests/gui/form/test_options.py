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
    def defaults(
        cls, env: FormEnv, previous: Mapping[str, JsonValue]
    ) -> dict[str, JsonValue]:
        del env
        return (
            {"kind": "second", "shared": 7.0, "required": len(previous), "derived": 1}
            if previous
            else {"kind": "second", "shared": {"expr": "2 * 4"}}
        )


class Picks(FormModel):
    count: Literal[1, 2] = 1
    speed: Speed = Speed.slow
    device: str = "a"
    flag: bool = False
    enabled: First | None = None
    branch: First | Second = Field(default_factory=First, discriminator="kind")

    @classmethod
    def choices(cls, env: FormEnv) -> Mapping[FormPath, tuple[JsonValue, ...]]:
        return {("device",): tuple(env.devices) or ("a", "b")}


def obj(value: JsonValue) -> dict[str, JsonValue]:
    assert isinstance(value, dict)
    return value


def test_schema_offers_typed_default_options_without_consumer_union_inference() -> None:
    properties = obj(Picks.input_schema(FormEnv())["properties"])
    assert obj(properties["count"]).get("x-form-options") == [
        {"label": "1", "value": 1},
        {"label": "2", "value": 2},
    ]
    assert obj(properties["speed"]).get("x-form-options") == [
        {"label": "slow", "value": "slow"},
        {"label": "fast", "value": "fast"},
    ]
    assert obj(properties["device"]).get("x-form-options") == [
        {"label": "a", "value": "a"},
        {"label": "b", "value": "b"},
    ]
    options = obj(properties["branch"]).get("x-form-options")
    assert isinstance(options, list)
    assert [obj(option)["value"] for option in options] == [
        {"kind": "first"},
        {"kind": "second"},
    ]
    assert [obj(option)["label"] for option in options] == ["first", "second"]
    fields = obj(obj(options[0])["fields"])
    assert set(obj(fields["properties"])) == {"shared"}
    assert "x-form-options" not in obj(properties["flag"])
    assert "x-form-options" not in obj(properties["enabled"])


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


@pytest.mark.parametrize(
    "batch", [{("branch",): {"kind": "second"}}, {("branch", "kind"): "second"}]
)
def test_branch_selection_starts_from_fresh_defaults_and_commits_once(
    batch: Mapping[tuple[str, ...], JsonValue],
) -> None:
    env = FormEnv()
    state = FormState(
        FormInput(Picks, {"branch": {"kind": "first", "shared": {"expr": "99"}}})
    )
    observed: list[dict[str, JsonValue]] = []
    state.subscribe(lambda: observed.append(state.input.data))
    state.write(batch, env)
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
    def options(cls, env: FormEnv) -> Mapping[FormPath, tuple[api.Option, ...]]:
        def convert(captured: FormEnv) -> JsonValue:
            if captured is not env:
                raise ValueError("wrong capture")
            return {"kind": "snapshot", "name": "copied"}

        def fail(_captured: FormEnv) -> JsonValue:
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
    schema = CommandForm.input_schema(env)
    choices = obj(obj(schema["properties"])["value"])["x-form-options"]
    assert isinstance(choices, list)
    assert obj(choices[0])["commands"] == [
        {"name": "convert", "label": "Convert"},
        {"name": "fail", "label": "Fail"},
    ]
    assert set(obj(obj(obj(choices[1])["fields"])["properties"])) == {"name"}
    state = FormState(CommandForm.default_input(env))
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
    assert hasattr(api, "Option"), "core must expose named Option"
    props = obj(NestedOptions.input_schema(FormEnv())["properties"])
    child = obj(obj(props["child"])["properties"])
    sibling = obj(obj(props["sibling"])["properties"])
    assert obj(child["empty"])["x-form-options"] == []
    assert obj(child["local"])["x-form-options"] == [{"label": "Local", "value": 1}]
    assert obj(child["parent"])["x-form-options"] == [
        {"label": "Parent retained", "value": 1}
    ]
    assert obj(sibling["parent"])["x-form-options"] == [
        {"label": "Sibling", "value": 2}
    ]


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
                    commands=(api.OptionCommand("active", "Active", lambda _env: 2),),
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
                        api.OptionCommand("inactive", "Inactive", lambda _env: 3),
                    ),
                ),
            )
        }


class BranchCommands(FormModel):
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
    state.write({("branch", "kind"): "inactive"}, env)
    state.run_command(("branch", "value"), "inactive", env)
    assert state.input.data == {"branch": {"kind": "inactive", "value": 3}}


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
        UnknownOptionPath.input_schema(env)
    with pytest.raises(ValueError, match="option.*path"):
        state.run_command(("value",), "unused", env)
    assert state.input.data == before


class OuterInline(FormModel):
    kind: Literal["inline"] = "inline"
    module: First | Second = Field(discriminator="kind")
    mode: Literal["fast", "slow"] = "fast"


class NestedSelection(FormModel):
    value: Source | OuterInline = Field(discriminator="kind")

    @classmethod
    def options(cls, _env: FormEnv, /) -> Mapping[FormPath, tuple[api.Option, ...]]:
        return {
            ("value",): (
                api.Option(
                    "Inline first",
                    {"kind": "inline", "module": {"kind": "first"}},
                    fields=OuterInline,
                ),
            )
        }


def test_option_fields_project_nested_selected_branch_without_consumer_inference() -> (
    None
):
    assert hasattr(api, "Option"), "core must expose named Option"
    options = obj(obj(NestedSelection.input_schema(FormEnv())["properties"])["value"])[
        "x-form-options"
    ]
    assert isinstance(options, list)
    fields = obj(obj(options[0])["fields"])
    properties = obj(fields["properties"])
    module = obj(properties["module"])
    assert set(obj(module.get("properties"))) == {"shared"}
    assert set(properties) == {"module", "mode"}
    assert obj(properties["mode"])["x-form-options"] == [
        {"label": "fast", "value": "fast"},
        {"label": "slow", "value": "slow"},
    ]


class NullablePicks(FormModel):
    number: Literal[1, 2] | None = None
    speed: Speed | None = None


def test_nullable_literal_and_enum_options_include_typed_null() -> None:
    props = obj(NullablePicks.input_schema(FormEnv())["properties"])
    for name, nonnull in (("number", [1, 2]), ("speed", ["slow", "fast"])):
        options = obj(props[name])["x-form-options"]
        assert isinstance(options, list)
        values = [obj(option)["value"] for option in options]
        assert None in values
        assert [value for value in values if value is not None] == nonnull


class UnavailableBranch(FormModel):
    kind: Literal["unavailable"] = "unavailable"
    source: str

    @classmethod
    def defaults(
        cls, env: FormEnv, previous: Mapping[str, JsonValue]
    ) -> dict[str, JsonValue]:
        del env, previous
        raise api.field_error(("source",), "Unavailable captured source")


class FailureContainer(FormModel):
    branch: First | UnavailableBranch = Field(
        default_factory=First, discriminator="kind"
    )


class SourceFailureForm(FormModel):
    outer: FailureContainer = Field(default_factory=FailureContainer)
    flag: bool = False


def test_partial_branch_source_failure_preserves_parent_path_and_batch_atomicity() -> (
    None
):
    env = FormEnv()
    state = FormState(SourceFailureForm.default_input(env))
    before = state.input.data
    events: list[dict[str, JsonValue]] = []
    state.subscribe(lambda: events.append(state.input.data))
    with pytest.raises(
        PydanticCustomError, match="Unavailable captured source"
    ) as caught:
        state.write(
            {("flag",): True, ("outer", "branch"): {"kind": "unavailable"}}, env
        )
    assert (caught.value.context or {}).get("path") == ("outer", "branch", "source")
    assert state.input.data == before
    assert state.base.data == before
    assert events == []
