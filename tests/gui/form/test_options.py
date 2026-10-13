"""Generic option selection, projection and commands at public form seams."""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Literal

import pytest
from pydantic import BaseModel, Field, JsonValue
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
    with pytest.raises(ValueError):
        Option("bad", float("inf"))
    with pytest.raises(ValueError):
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
    with pytest.raises(ValueError):
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
