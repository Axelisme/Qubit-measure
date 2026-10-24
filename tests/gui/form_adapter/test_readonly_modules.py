"""Readonly descendants constrain their values, not ancestor source choices."""

from typing import Annotated, ClassVar, Literal

import pytest
from pydantic import Field
from zcu_tools.cfg_model import ConfigBase
from zcu_tools.gui.form import (
    Display,
    FormInput,
    FormInputRejected,
    FormModel,
    FormPath,
    FormState,
)
from zcu_tools.gui.form_adapter import Form, ModuleRef

from tests.gui._form_converters import RebuildingForm
from tests.gui.form_adapter._indexed_models import indexed_env
from tests.gui.form_adapter._input_models import specialize_input


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


def test_readonly_inline_child_allows_ancestor_library_selection() -> None:
    definition = Form(ExperimentCfg, {})
    input_model = specialize_input(definition, SourceExtras)
    env = indexed_env()
    state = FormState(
        FormInput(input_model, {"pulse": InlinePulse().model_dump(mode="json")})
    )
    with pytest.raises(FormInputRejected) as rejected:
        state.write({("pulse", "freq"): 125.0}, env)
    assert rejected.value.reason == "readonly"
    state.select(("pulse",), {"kind": "ref"}, env)
    state.write({("pulse", "name"): "Q.pulses.drive"}, env)
    assert state.input.data["pulse"] == {"kind": "ref", "name": "Q.pulses.drive"}
