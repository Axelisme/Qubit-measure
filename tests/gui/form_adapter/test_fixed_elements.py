"""The declaring adapter owns fixed identities and ordinary participant errors."""

from collections.abc import Mapping
from typing import Annotated

import pytest
from pydantic import Field, JsonValue, create_model
from zcu_tools.cfg_model import ConfigBase
from zcu_tools.gui import form as core
from zcu_tools.gui import form_adapter as adapter
from zcu_tools.gui.app.measure.services.tab_form import TabFormResource, TabFormSources


class Element(core.FormModel):
    label: str
    gain: float = Field(gt=0)


def owner_model(*, duplicate: bool = False) -> type[core.FormModel]:
    """Build a synthetic declaration only after checking its new public API."""
    assert hasattr(adapter, "FixedElementsForm"), "Missing declaring-owner model"
    assert hasattr(adapter, "FixedElements"), "Missing owner fixed-elements capability"
    assert hasattr(core, "FormInputRejected"), "Missing recoverable typed-path refusal"

    class Owner(adapter.FixedElementsForm):
        fixed_elements = (
            adapter.FixedElements(("tones",), "label", ("participants",)),
        )
        tones: list[Element]
        participants: list[str]

        @classmethod
        def defaults(cls, env: core.FormEnv) -> dict[str, JsonValue]:
            return {
                "tones": [
                    {"label": "A", "gain": 1},
                    {"label": "A" if duplicate else "B", "gain": 2},
                ],
                "participants": ["B"],
            }

    return Owner


def owner_state() -> core.FormState[core.FormModel]:
    model = owner_model()
    return core.FormState(model.default_input(core.FormEnv()))


@pytest.mark.parametrize(
    "path,value",
    [
        (("tones", 0, "label"), "C"),
        (("tones",), []),
        (("tones",), [{"label": "A", "gain": 1}]),
        (("tones",), [{"label": "B", "gain": 2}, {"label": "A", "gain": 1}]),
    ],
)
def test_owner_refuses_identity_mutations_atomically_with_requested_path(path, value):
    state = owner_state()
    before, base = state.input, state.base
    notifications = []
    state.subscribe(lambda: notifications.append(state.input))
    with pytest.raises(core.FormInputRejected, match="identity") as caught:
        state.write({path: value, ("participants",): ["A"]}, core.FormEnv())
    assert caught.value.path == path
    assert state.input.same_as(before)
    assert state.base.same_as(base)
    assert state.input.selections == before.selections
    assert notifications == []


def test_owner_allows_invalid_values_but_checks_every_element_and_participant():
    state = owner_state()
    env = core.FormEnv()
    state.write({("tones", 0, "gain"): -1, ("participants",): ["B"]}, env)
    errors = state.result(env)
    assert isinstance(errors, core.FormErrors)
    assert ("tones", 0, "gain") in errors.messages
    state.write({("tones", 0, "gain"): 3}, env)
    for value, error_path in [
        ([], ("participants",)),
        (["A", "A"], ("participants", 1)),
        (["Q4"], ("participants", 0)),
    ]:
        state.write({("participants",): value}, env)
        assert state.input.data["participants"] == value
        errors = state.result(env)
        assert isinstance(errors, core.FormErrors)
        assert error_path in errors.messages
    state.write({("participants",): ["B", "A"]}, env)
    result = state.result(env)
    assert isinstance(result, core.FormModel)
    assert result.model_dump()["participants"] == ["B", "A"]
    assert [item["label"] for item in result.model_dump()["tones"]] == ["A", "B"]


def test_owner_rejects_duplicate_capture_names_with_original_indices():
    model = owner_model(duplicate=True)
    with pytest.raises(ValueError, match=r"A.*0.*1"):
        model.default_input(core.FormEnv())


def test_restored_identity_becomes_the_boundary_until_reset():
    state = owner_state()
    env = core.FormEnv()
    state.write_compatible(
        {"tones": [{"label": "saved", "gain": 5}], "participants": ["saved"]}, env
    )
    state.write({("tones", 0, "gain"): 7}, env)
    assert state.input.data["tones"] == [{"label": "saved", "gain": 7}]
    with pytest.raises(core.FormInputRejected, match="identity"):
        state.write({("tones", 0, "label"): "A"}, env)
    state.reset(state.input.model.default_input(env))
    assert state.input.data["participants"] == ["B"]


def test_resource_uses_declaring_owner_for_whole_list_and_recipe_edits():
    model = owner_model()
    cfg = create_model(
        "Config",
        __base__=ConfigBase,
        tones=(list[Element], ...),
        participants=(list[str], ...),
    )
    initial = model.default_input(core.FormEnv()).data
    resource = TabFormResource(
        lambda _ctx: adapter.Form(
            cfg,
            initial,
            fixed_elements=(
                adapter.FixedElements(("tones",), "label", ("participants",)),
            ),
        ),
        capture=lambda _aliases: TabFormSources(core.FormEnv()),
        mutation_allowed=lambda: True,
    )
    events = []
    resource.subscribe(lambda: events.append(resource.input))
    before = resource.input
    try:
        with pytest.raises(core.FormInputRejected, match="identity") as caught:
            resource.edit_recipe_fields(
                (adapter.LogicalEdit(("tones", 0, "label"), "C"),)
            )
        assert caught.value.path == ("tones", 0, "label")
        assert resource.input.same_as(before)
        assert events == []
        resource.write({("tones", 0, "gain"): 4, ("participants",): ["Q4"]})
        assert resource.input.data["participants"] == ["Q4"]
        assert isinstance(resource.result(), core.FormErrors)
        with pytest.raises(ValueError, match="not valid"):
            resource.accept()
    finally:
        resource.revoke()


@pytest.mark.parametrize("operation", ["command", "converter", "install"])
def test_owner_checks_complete_command_converter_and_prepared_replacements(operation):
    owner = owner_model()

    def replace(_snapshot, _context):
        return core.FormCommandResult(None, None, [])

    class Commands(owner):
        tones: Annotated[
            list[Element], core.InputCommand("clear", "Clear", (), replace)
        ]

        @classmethod
        def options(
            cls, _env: core.FormEnv
        ) -> Mapping[core.FormPath, tuple[core.Option, ...]]:
            return {("tones",): (core.Option("Clear", []),)}

        @classmethod
        def converters(cls, _env: core.FormEnv):
            return {
                ("tones",): core.InputConverter(
                    lambda *_args: core.FormConversionResult(None, [])
                )
            }

    state = core.FormState(Commands.default_input(core.FormEnv()))
    before, base = state.input, state.base
    events = []
    state.subscribe(lambda: events.append(state.input))
    with pytest.raises(core.FormInputRejected, match="identity"):
        if operation == "command":
            state.run_command(("tones",), "clear", core.FormEnv())
        elif operation == "converter":
            state.write({("tones",): []}, core.FormEnv())
        else:
            state.install_input(
                core.FormInput(Commands, {"tones": [], "participants": ["B"]})
            )
    assert state.input.same_as(before)
    assert state.base.same_as(base)
    assert state.input.selections == before.selections
    assert events == []
