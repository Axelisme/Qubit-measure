"""Native nullable fields and original-index participation share one input."""

from typing import Annotated, Self

import pytest
from pydantic import Field, model_validator
from zcu_tools.cfg_model import ConfigBase, FieldMeta, PulseCfg, Range
from zcu_tools.gui import form as core
from zcu_tools.gui import form_adapter as adapter
from zcu_tools.program.v2.sweep import SweepCfg


class Tone(ConfigBase):
    """Readout identity is unconditional; drive and sweep can be inapplicable."""

    name: Annotated[str, FieldMeta(readonly=True)]
    readout: float = Field(gt=0)
    drive: PulseCfg | None = None
    sweep: SweepCfg | None = None


class Group(ConfigBase):
    """Require native drive/sweep only at the original participant indices."""

    tones: Annotated[list[Tone], FieldMeta(element_label="name")]
    participants: list[int]

    @model_validator(mode="after")
    def validate_active(self) -> Self:
        """Direct a missing active setting to its original tone, never reorder."""
        for index in self.participants:
            for name in ("drive", "sweep"):
                if getattr(self.tones[index], name) is None:
                    raise core.field_error(("tones", index, name), "Active setting required")
        return self


def definition() -> adapter.Form[Group]:
    """Seed no sources: the inactive second tone deliberately has empty settings."""
    try:
        declaration = adapter.FixedElements(
            ("tones",), "name", ("participants",),
            participant_fields=(("drive",), ("sweep",)),
        )
    except TypeError as exc:
        pytest.fail(f"Adapter must declare participant-owned nullable fields: {exc}")
    return adapter.Form(Group, {
        "tones": [
            {"name": "A", "readout": 6000, "sweep": Range.between(1, 2, 3)},
            {"name": "B", "readout": 6100, "sweep": Range.between(None, None, None)},
        ],
        "participants": [0],
    }, fixed_elements=(declaration,))


def test_inactive_module_and_sweep_lower_to_none_without_losing_original_tones() -> None:
    form = definition()
    env = core.FormEnv()
    state = core.FormState(form.default_input(env))
    state.write({("tones", 0, "drive"): {
        "kind": "inline", "module": {"type": "pulse", "ch": 0, "nqz": 1,
        "freq": 100, "gain": 0.2, "waveform": {"kind": "inline", "module": {"style": "const", "length": 0.1}}},
    }}, env)
    raw = state.input.to_json()
    result = state.result(env)
    assert not isinstance(result, core.FormErrors), result
    cfg = form.to_cfg(result, env)
    assert not isinstance(cfg, core.FormErrors), cfg
    assert [tone.name for tone in cfg.tones] == ["A", "B"]
    assert cfg.participants == [0]
    assert cfg.tones[1].drive is None and cfg.tones[1].sweep is None
    assert state.input.to_json() == raw
    for name in ("drive", "sweep"):
        target = state.describe(("tones", 1, name), env).target
        assert target is not None and target.disabled
    state.write({("participants",): [0, 1]}, env)
    result = state.result(env)
    if isinstance(result, core.FormErrors):
        errors = result
    else:
        errors = form.to_cfg(result, env)
    assert isinstance(errors, core.FormErrors)
    assert any(path[:2] == ("tones", 1) for path in errors.messages)


def test_unselected_readout_identity_is_still_required() -> None:
    form = definition()
    env = core.FormEnv()
    state = core.FormState(form.default_input(env))
    state.write({("participants",): [1], ("tones", 0, "readout"): None}, env)
    errors = state.preview(env).errors
    assert errors is not None and ("tones", 0, "readout") in errors.messages


@pytest.mark.parametrize("participants, path", [([0, 0], ("participants", 1)), ([2], ("participants", 0)), ([-1], ("participants", 0)), ([True], ("participants", 0))])
def test_participant_errors_keep_original_index_path(participants: list[int], path: core.FormPath) -> None:
    # Reuse the ordinary generated declaration, but supply a simple valid model
    # so the assertion isolates index admission from missing drive/sweep values.
    class Element(core.FormModel):
        name: str

    class Named(adapter.FixedElementsForm):
        tones: list[Element]
        participants: list[int]

    Named.fixed_elements = (adapter.FixedElements(("tones",), "name", ("participants",)),)
    state = core.FormState(core.FormInput(Named, {"tones": [{"name": "A"}, {"name": "B"}], "participants": participants}))
    result = state.result(core.FormEnv())
    assert isinstance(result, core.FormErrors)
    assert path in result.messages
