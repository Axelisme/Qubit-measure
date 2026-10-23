"""The declaring adapter owns fixed identities and ordinary participant errors."""

from collections.abc import Mapping
from typing import Annotated

import pytest
from pydantic import Field, JsonValue, create_model
from zcu_tools.cfg_model import ConfigBase, Hidden
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
    assert cfg.participants == [1, 0]


def test_native_model_list_defaults_are_projected_without_losing_values():
    from tests.gui.form_adapter._indexed_models import element, native_defaults_form

    definition = native_defaults_form()
    env = core.FormEnv()
    state = core.FormState(definition.default_input(env))
    raw = element(state.input.data, 0)
    assert raw["sweep"] == {"kind": "linrange", "start": 3.0, "stop": 7.0, "n": 3}
    drive = raw["drive"]
    assert isinstance(drive, dict) and drive["kind"] == "inline"
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


def test_readonly_list_positions_keep_individual_values_and_restore_saved_values():
    from zcu_tools.cfg_model import FieldMeta

    class Tone(ConfigBase):
        name: Annotated[str, FieldMeta(readonly=True)]

    class Group(ConfigBase):
        tones: list[Tone]

    env = core.FormEnv()
    definition = adapter.Form(Group, {"tones": [{"name": "A"}, {"name": "B"}]})
    state = core.FormState(definition.default_input(env))
    state.write({("tones", 0, "name"): "A", ("tones", 1, "name"): "B"}, env)
    before = state.input
    with pytest.raises(core.FormInputRejected) as caught:
        state.write({("tones", 1, "name"): "A"}, env)
    assert caught.value.reason == "readonly"
    assert caught.value.current == "B"
    assert state.input.same_as(before)
    saved = before.data
    tones = saved["tones"]
    assert isinstance(tones, list) and isinstance(tones[0], dict)
    tones[0]["name"] = "stored-A"
    state.restore_input(core.FormInput(definition.model, saved), env)
    assert state.input.data["tones"] == tones
    result = state.result(env)
    assert not isinstance(result, core.FormErrors), result
    cfg = definition.to_cfg(result, env)
    assert not isinstance(cfg, core.FormErrors), cfg
    assert [tone.name for tone in cfg.tones] == ["stored-A", "B"]


@pytest.mark.parametrize(
    "node",
    [
        Hidden(),
    ],
)
def test_native_position_seeds_reject_declaration_changes_at_typed_position(node):
    from tests.gui.form_adapter._indexed_models import NativeGroup

    with pytest.raises(ValueError, match="position.*declaration") as caught:
        adapter.Form(NativeGroup, {"tones": [{}, {"drive": node}]})
    assert "('tones', 1, 'drive')" in str(caught.value)


def test_hidden_module_list_keeps_values_and_sources_per_position():
    from zcu_tools.cfg_model import PulseSpec, Range, Slot, Template, TemplateDef

    from tests.gui.form_adapter._indexed_models import NativeGroup, indexed_env

    env = indexed_env()
    definition = adapter.Form(
        NativeGroup,
        {
            "tones": [
                {
                    "name": "A",
                    "drive": Template(
                        "pulse",
                        set={"freq": Hidden(11.0), "waveform.length": Hidden(0.2)},
                    ),
                    "sweep": Range.between(1, 2, 3),
                },
                {
                    "name": "B",
                    "drive": Slot(
                        "Q.pulses.drive",
                        copy=True,
                        set={"freq": Hidden(22.0), "waveform.length": Hidden(0.4)},
                    ),
                    "sweep": Range.between(3, 4, 3),
                },
            ],
            "participants": [1, 0],
        },
        fixed_elements=(adapter.FixedElements(("tones",), "name", ("participants",)),),
    )
    definition.bind_templates(
        {
            "pulse": TemplateDef(
                PulseSpec,
                "Pulse",
                {"ch": 1, "nqz": 1, "gain": 0.3, "waveform.style": "const"},
            )
        },
        adapter="synthetic",
    )
    state = core.FormState(definition.default_input(env))
    for index in (0, 1):
        for suffix in (("freq",), ("waveform", "module", "length")):
            target = state.describe(
                ("tones", index, "drive", "module", *suffix), env
            ).target
            assert target is not None and target.hidden
    result = state.result(env)
    assert not isinstance(result, core.FormErrors), result
    cfg = definition.to_cfg(result, env)
    assert not isinstance(cfg, core.FormErrors), cfg
    assert [(tone.drive.freq, tone.drive.waveform.length) for tone in cfg.tones] == [
        (11.0, 0.2),
        (22.0, 0.4),
    ]
    assert [(tone.drive.ch, tone.drive.gain) for tone in cfg.tones] == [
        (1, 0.3),
        (0, 0.2),
    ]
    assert env.library is not None
    assert env.library.module_spec("Q.pulses.drive").model_dump()["freq"] == 100


@pytest.mark.parametrize("inactive_index", [0, 1])
def test_nullable_module_list_keeps_hidden_policy_with_explicit_none(inactive_index):
    from zcu_tools.cfg_model import FormNode, PulseCfg, PulseSpec, Template, TemplateDef

    class Tone(ConfigBase):
        drive: PulseCfg | None = None

    class Group(ConfigBase):
        tones: list[Tone]

    tones: list[FormNode] = [
        {"drive": Template("pulse", set={"freq": Hidden(11.0)})},
        {"drive": Template("pulse", set={"freq": Hidden(22.0)})},
    ]
    tones[inactive_index] = {"drive": None}
    definition = adapter.Form(Group, {"tones": tones})
    try:
        definition.bind_templates(
            {
                "pulse": TemplateDef(
                    PulseSpec,
                    "Pulse",
                    {
                        "ch": 1,
                        "nqz": 1,
                        "gain": 0.3,
                        "waveform.style": "const",
                        "waveform.length": 0.2,
                    },
                )
            },
            adapter="synthetic",
        )
    except ValueError as error:
        pytest.fail(
            f"Explicit nullable None must not conflict with Hidden policy: {error}"
        )
    env = core.FormEnv()
    state = core.FormState(definition.default_input(env))
    active_index = 1 - inactive_index
    target = state.describe(
        ("tones", active_index, "drive", "module", "freq"), env
    ).target
    assert target is not None and target.hidden
    result = state.result(env)
    assert not isinstance(result, core.FormErrors), result
    cfg = definition.to_cfg(result, env)
    assert not isinstance(cfg, core.FormErrors), cfg
    assert cfg.tones[inactive_index].drive is None
    drive = cfg.tones[active_index].drive
    assert drive is not None and drive.freq == (22.0 if active_index else 11.0)
    state.write({("tones", active_index, "drive"): None}, env)
    result = state.result(env)
    assert not isinstance(result, core.FormErrors), result
    cfg = definition.to_cfg(result, env)
    assert not isinstance(cfg, core.FormErrors), cfg
    assert [tone.drive for tone in cfg.tones] == [None, None]


@pytest.mark.parametrize("second_path", ["gain", None])
def test_hidden_module_list_rejects_mismatched_paths(second_path):
    from zcu_tools.cfg_model import Slot

    from tests.gui.form_adapter._indexed_models import NativeGroup

    definition = adapter.Form(
        NativeGroup,
        {
            "tones": [
                {"drive": Slot("Q.pulses.drive", set={"freq": Hidden(0.0)})},
                {
                    "drive": Slot(
                        "Q.pulses.drive",
                        set={} if second_path is None else {second_path: Hidden(0.0)},
                    )
                },
            ]
        },
    )
    with pytest.raises(ValueError, match="Hidden.*paths|hidden.*paths") as caught:
        definition.bind_templates({}, adapter="synthetic")
    assert "tones" in str(caught.value)


def test_indexed_module_read_rejects_inactive_ancestry_with_original_path():
    from zcu_tools.cfg_model import PulseCfg

    class Section(ConfigBase):
        drive: PulseCfg | None = None

    class Tone(ConfigBase):
        name: str
        section: Section | None = None

    class Group(ConfigBase):
        tones: list[Tone]

    definition = adapter.Form(Group, {"tones": [{"name": "A"}]})
    env = core.FormEnv()
    state = core.FormState(definition.default_input(env))
    state.write({("tones", 0, "section"): None}, env)
    path = ("tones", 0, "section", "drive")
    with pytest.raises(core.FormPathError) as caught:
        definition.read_module(state.input, path, env)
    assert caught.value.path == path
    with pytest.raises(core.FormPathError) as caught:
        definition.module_input_kind(state.input, path)
    assert caught.value.path == path


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
        participants: list[int]

        @classmethod
        def defaults(cls, env: core.FormEnv) -> dict[str, JsonValue]:
            del env
            return {
                "tones": [
                    {"label": "A", "gain": 1},
                    {"label": "A" if duplicate else "B", "gain": 2},
                ],
                "participants": [1],
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
        state.write({path: value, ("participants",): [0]}, core.FormEnv())
    assert caught.value.path == path
    assert state.input.same_as(before)
    assert state.base.same_as(base)
    assert state.input.selections == before.selections
    assert notifications == []


def test_owner_allows_invalid_values_but_checks_every_element_and_participant():
    state = owner_state()
    env = core.FormEnv()
    state.write({("tones", 0, "gain"): -1, ("participants",): [1]}, env)
    errors = state.result(env)
    assert isinstance(errors, core.FormErrors)
    assert ("tones", 0, "gain") in errors.messages
    state.write({("tones", 0, "gain"): 3}, env)
    cases: list[tuple[JsonValue, core.FormPath]] = [
        ([], ("participants",)),
        ([0, 0], ("participants", 1)),
        ([4], ("participants", 0)),
    ]
    for value, error_path in cases:
        state.write({("participants",): value}, env)
        assert state.input.data["participants"] == value
        errors = state.result(env)
        assert isinstance(errors, core.FormErrors)
        assert error_path in errors.messages
    state.write({("participants",): [1, 0]}, env)
    result = state.result(env)
    assert isinstance(result, core.FormModel)
    assert result.model_dump()["participants"] == [1, 0]
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
            {"tones": [{"label": "saved", "gain": 5}], "participants": [0]},
        ),
        env,
    )
    state.write({("tones", 0, "gain"): 7}, env)
    assert state.input.data["tones"] == [{"label": "saved", "gain": 7}]
    with pytest.raises(core.FormInputRejected, match="identity"):
        state.write({("tones", 0, "label"): "A"}, env)
    state.reset(state.input.model.default_input(env))
    assert state.input.data["participants"] == [1]


def test_resource_uses_declaring_owner_for_whole_list_and_recipe_edits():
    model = owner_model()
    initial = model.default_input(core.FormEnv()).data
    cfg = create_model(
        "Config",
        __base__=ConfigBase,
        tones=(list[Element], initial["tones"]),
        participants=(list[int], initial["participants"]),
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
        resource.write({("tones", 0, "gain"): 4, ("participants",): [4]})
        assert resource.input.data["participants"] == [4]
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
                core.FormInput(Commands, {"tones": [], "participants": [1]})
            )

    with pytest.raises(core.FormInputRejected, match="identity"):
        mutate()
    assert state.input.same_as(before)
    assert state.base.same_as(base)
    assert state.input.selections == before.selections
    assert events == []


@pytest.mark.parametrize(
    "participants, path",
    [
        ([0, 0], ("participants", 1)),
        ([2], ("participants", 0)),
        ([-1], ("participants", 0)),
        ([True], ("participants", 0)),
    ],
)
def test_participant_errors_keep_original_index_path(
    participants: list[int], path: core.FormPath
) -> None:
    # Reuse the ordinary generated declaration, but supply a simple valid model
    # so the assertion isolates index admission from missing drive/sweep values.
    class Element(core.FormModel):
        name: str

    class Named(adapter.FixedElementsForm):
        tones: list[Element]
        participants: list[int]

    Named.fixed_elements = (
        adapter.FixedElements(("tones",), "name", ("participants",)),
    )
    state = core.FormState(
        core.FormInput(
            Named,
            {
                "tones": [{"name": "A"}, {"name": "B"}],
                "participants": [value for value in participants],
            },
        )
    )
    result = state.result(core.FormEnv())
    assert isinstance(result, core.FormErrors)
    assert path in result.messages
