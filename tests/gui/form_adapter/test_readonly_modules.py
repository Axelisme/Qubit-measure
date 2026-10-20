"""Readonly descendants constrain their values, not ancestor source choices."""

from typing import Annotated, ClassVar, Literal

import pytest
from pydantic import Field
from zcu_tools.cfg_model import ConfigBase
from zcu_tools.gui.form import (
    Display,
    FormInputRejected,
    FormModel,
    FormPath,
    FormState,
)
from zcu_tools.gui.form_adapter import Form, ModuleRef

from tests.gui._form_converters import RebuildingForm
from tests.gui.form_adapter._indexed_models import indexed_env


class InlinePulse(FormModel):
    kind: Literal["inline"] = "inline"
    freq: Annotated[float, Display(readonly=True)] = 100.0
    gain: float = 0.2


class SourceExtras(RebuildingForm):
    branch_models: ClassVar[dict[FormPath, tuple[type[FormModel], ...]]] = {
        ("pulse",): (ModuleRef, InlinePulse)
    }
    pulse: ModuleRef | InlinePulse = Field(
        default_factory=InlinePulse, discriminator="kind"
    )


class ExperimentCfg(ConfigBase):
    rounds: int = 1


def lower_source(_extras: SourceExtras, cfg: ExperimentCfg) -> ExperimentCfg:
    """Keep this synthetic experiment independent of the author-only source."""
    return cfg


def test_readonly_inline_child_allows_ancestor_library_selection() -> None:
    definition = Form(ExperimentCfg, {}, extras=SourceExtras, lower=lower_source)
    env = indexed_env()
    state = FormState(definition.default_input(env))
    with pytest.raises(FormInputRejected) as rejected:
        state.write({("extras", "pulse", "freq"): 125.0}, env)
    assert rejected.value.reason == "readonly"
    state.select(("extras", "pulse"), {"kind": "ref"}, env)
    state.write({("extras", "pulse", "name"): "Q.pulses.drive"}, env)
    assert state.input.data["extras"] == {
        "pulse": {"kind": "ref", "name": "Q.pulses.drive"}
    }
