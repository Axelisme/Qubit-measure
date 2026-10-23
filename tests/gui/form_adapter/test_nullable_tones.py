"""Public generated-form contracts for ordinary nullable tone drafts."""

from typing import Annotated

from pydantic import Field
from zcu_tools.cfg_model import ConfigBase, FieldMeta, FormNode, PulseCfg
from zcu_tools.gui.form import FormEnv, FormErrors, FormState
from zcu_tools.gui.form_adapter import FixedElements, Form
from zcu_tools.program.v2.sweep import SweepCfg


class Bounds(ConfigBase):
    """Finite experiment-specific bounds, independent of participation."""

    start: float
    stop: float


class Tone(ConfigBase):
    """Readout stays required while pulse and both sweep shapes accept null."""

    name: Annotated[str, FieldMeta(readonly=True)]
    readout: float = Field(gt=0)
    drive: PulseCfg | None = None
    sweep: Bounds | None = None
    delay: SweepCfg | None = None


class Group(ConfigBase):
    """Keep complete ordered tones and original participant indices."""

    tones: Annotated[list[Tone], FieldMeta(element_label="name")]
    participants: list[int]


def definition() -> Form[Group]:
    """Author explicit nulls, including null overriding nonempty sibling seeds."""
    tones: list[FormNode] = [
        {"name": "A", "readout": 6000, "sweep": {"start": 1, "stop": 2}},
        {"name": "B", "readout": 6100, "drive": None, "sweep": None, "delay": None},
    ]
    return Form(
        Group,
        {"tones": tones, "participants": [0]},
        fixed_elements=(FixedElements(("tones",), "name", ("participants",)),),
    )


def test_explicit_nullable_defaults_roundtrip_and_participant_edits_preserve_values() -> (
    None
):
    form, env = definition(), FormEnv()
    state = FormState(form.default_input(env))
    result = state.result(env)
    assert not isinstance(result, FormErrors), result
    cfg = form.to_cfg(result, env)
    assert not isinstance(cfg, FormErrors), cfg
    assert cfg.tones[1].sweep is None
    assert cfg.tones[1].delay is None
    assert cfg.tones[1].drive is None
    assert [tone.name for tone in cfg.tones] == ["A", "B"]
    original = state.input.data["tones"]
    state.write({("participants",): [1, 0]}, env)
    assert state.input.data["tones"] == original
    state.write({("tones", 1, "sweep"): {"start": 3, "stop": 4}}, env)
    edited = state.input.data["tones"]
    state.write({("participants",): [0]}, env)
    assert state.input.data["tones"] == edited
    state.reset(form.default_input(env))
    result = state.result(env)
    assert not isinstance(result, FormErrors), result
    cfg = form.to_cfg(result, env)
    assert not isinstance(cfg, FormErrors), cfg
    assert cfg.tones[1].sweep is None


def test_unselected_nullable_values_and_readout_still_validate() -> None:
    form, env = definition(), FormEnv()
    state = FormState(form.default_input(env))
    state.write({("tones", 1, "readout"): None}, env)
    errors = state.preview(env).errors
    assert errors is not None and ("tones", 1, "readout") in errors.messages
    state.write(
        {
            ("tones", 1, "readout"): 6100,
            ("tones", 1, "sweep"): {"start": "bad", "stop": 4},
        },
        env,
    )
    errors = state.preview(env).errors
    assert errors is not None and ("tones", 1, "sweep", "start") in errors.messages
