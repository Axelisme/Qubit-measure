"""The declaring adapter owns fixed identities and ordinary participant errors."""

from collections.abc import Mapping
from typing import Annotated

import pytest
from pydantic import Field, JsonValue, create_model
from zcu_tools.cfg_model import ConfigBase, Hidden, Locked, Slot, Template
from zcu_tools.gui import form as core
from zcu_tools.gui import form_adapter as adapter
from zcu_tools.gui.app.measure.services.form_contracts import (
    CfgPreconditionError,
    CfgPreconditionReason,
)
from zcu_tools.gui.app.measure.services.tab_form import TabFormResource, TabFormSources


class Element(core.FormModel):
    label: str
    gain: float = Field(gt=0)


def test_native_model_list_generates_indexed_modules_sweeps_and_lowers():
    from tests.gui.form_adapter._indexed_models import (
        element,
        indexed_env,
        indexed_form,
    )

    env = indexed_env()
    definition = indexed_form()
    try:
        state = core.FormState(definition.default_input(env))
    except TypeError as exc:
        pytest.fail(f"Native model-list seeds must generate editable inputs: {exc}")
    assert element(state.input.data, 0)["drive"] == {
        "kind": "ref",
        "name": "Q.pulses.drive",
    }
    sweep = state.describe(("tones", 1, "sweep"), env)
    assert sweep.target is not None and sweep.target.commands
    sibling = element(state.input.data, 0)
    state.run_command(("tones", 1, "drive"), "to_inline", env)
    state.write({("tones", 1, "drive", "module", "gain"): {"expr": "3 / 10"}}, env)
    state.run_command(("tones", 1, "sweep"), "to_center", env)
    assert element(state.input.data, 0) == sibling
    result = state.result(env)
    assert not isinstance(result, core.FormErrors), result
    cfg = definition.to_cfg(result, env)
    assert not isinstance(cfg, core.FormErrors), cfg
    assert cfg.tones[1].drive.gain == 0.3
    assert cfg.tones[1].sweep.start == 10
    assert cfg.tones[1].sweep.stop == 14
    assert cfg.participants == ["B", "A"]


def test_native_model_list_defaults_are_projected_without_losing_values():
    from tests.gui.form_adapter._indexed_models import element, native_defaults_form

    definition = native_defaults_form()
    env = core.FormEnv()
    state = core.FormState(definition.default_input(env))
    raw = element(state.input.data, 0)
    assert raw["sweep"] == {"kind": "linrange", "start": 3.0, "stop": 7.0, "n": 3}
    assert raw["drive"]["kind"] == "inline"
    result = state.result(env)
    assert not isinstance(result, core.FormErrors), result
    cfg = definition.to_cfg(result, env)
    assert not isinstance(cfg, core.FormErrors), cfg
    assert cfg.tones[0].drive.freq == 42
    assert cfg.tones[0].sweep.step == 2


def test_native_field_meta_exposes_readonly_names_and_picker_labels():
    from inspect import signature

    from zcu_tools.cfg_model import FieldMeta

    from tests.gui.form_adapter._indexed_models import displayed_form, indexed_env

    assert "readonly" in signature(FieldMeta).parameters, "Missing native readonly hint"
    assert "element_label" in signature(FieldMeta).parameters, (
        "Missing native picker hint"
    )
    env = indexed_env()
    definition = displayed_form()
    state = core.FormState(definition.default_input(env))
    target = state.describe(("tones",), env).target
    assert target is not None and target.elements is not None
    assert [(item.index, item.name) for item in target.elements] == [(0, "A"), (1, "B")]
    name = state.describe(("tones", 0, "name"), env).target
    assert name is not None and name.readonly


@pytest.mark.parametrize("node", [
    Locked(1), Hidden(),
    Slot("Q.pulses.drive", lock={"ch": 0}), Template("pulse", lock={"ch": 0}),
])
def test_native_position_seeds_reject_declaration_changes_at_typed_position(node):
    from tests.gui.form_adapter._indexed_models import NativeGroup, indexed_tree

    tree = indexed_tree()
    tree["tones"][1]["drive"] = node
    with pytest.raises(ValueError, match="position.*declaration") as caught:
        adapter.Form(NativeGroup, tree)
    assert "('tones', 1, 'drive')" in str(caught.value)


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
            del env
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
    cases: list[tuple[JsonValue, core.FormPath]] = [
        ([], ("participants",)),
        (["A", "A"], ("participants", 1)),
        (["Q4"], ("participants", 0)),
    ]
    for value, error_path in cases:
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
    state.restore_input(
        core.FormInput(
            state.input.model,
            {"tones": [{"label": "saved", "gain": 5}], "participants": ["saved"]},
        ),
        env,
    )
    state.write({("tones", 0, "gain"): 7}, env)
    assert state.input.data["tones"] == [{"label": "saved", "gain": 7}]
    with pytest.raises(core.FormInputRejected, match="identity"):
        state.write({("tones", 0, "label"): "A"}, env)
    state.reset(state.input.model.default_input(env))
    assert state.input.data["participants"] == ["B"]


def test_resource_uses_declaring_owner_for_whole_list_and_recipe_edits():
    model = owner_model()
    initial = model.default_input(core.FormEnv()).data
    cfg = create_model(
        "Config",
        __base__=ConfigBase,
        tones=(list[Element], initial["tones"]),
        participants=(list[str], initial["participants"]),
    )
    resource = TabFormResource(
        lambda _ctx: adapter.Form(
            cfg,
            {},
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
        with pytest.raises(CfgPreconditionError, match="not valid") as caught:
            resource.accept()
        assert caught.value.reason is CfgPreconditionReason.NOT_VALID
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
        def converters(
            cls, _env: core.FormEnv
        ) -> Mapping[core.FormPath, core.InputConverter]:
            return {
                ("tones",): core.InputConverter(
                    lambda *_args: core.FormConversionResult(None, [])
                )
            }

    state = core.FormState(Commands.default_input(core.FormEnv()))
    before, base = state.input, state.base
    events = []
    state.subscribe(lambda: events.append(state.input))

    def mutate() -> None:
        if operation == "command":
            state.run_command(("tones",), "clear", core.FormEnv())
        elif operation == "converter":
            state.select(("tones",), [], core.FormEnv())
        else:
            state.install_input(
                core.FormInput(Commands, {"tones": [], "participants": ["B"]})
            )

    with pytest.raises(core.FormInputRejected, match="identity"):
        mutate()
    assert state.input.same_as(before)
    assert state.base.same_as(base)
    assert state.input.selections == before.selections
    assert events == []
