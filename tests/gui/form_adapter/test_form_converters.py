"""Declared input converters and generated module converters share fixed sources."""

from collections.abc import Mapping
from dataclasses import replace
from typing import Annotated, Literal

from pydantic import Field, JsonValue
from zcu_tools.cfg_model import ConfigBase, PulseCfg, Slot
from zcu_tools.gui.form import FormEnv, FormModel, FormPath, FormState, InputConverter
from zcu_tools.gui.form_adapter import Form

from tests.gui._form_converters import fresh_branches
from tests.gui.app.measure._cfg_fakes import parameter_capture
from tests.gui.form_adapter._indexed_models import indexed_env


class Linear(FormModel):
    kind: Literal["linear"] = "linear"
    start: float = 1.0


class Captured(FormModel):
    kind: Literal["captured"] = "captured"
    start: float

    @classmethod
    def defaults(cls, env: FormEnv) -> dict[str, JsonValue]:
        if env.parameters is None:
            raise ValueError("Captured branch needs fixed parameters")
        return {"start": env.parameters.value("general.ext.seed")}


class SamplingInput(FormModel):
    sampling: Annotated[Linear | Captured, Field(discriminator="kind")] = Linear()

    @classmethod
    def converters(cls, env: FormEnv) -> Mapping[FormPath, InputConverter]:
        return {("sampling",): fresh_branches(env, (Linear, Captured))}


class NativeInput(ConfigBase, SamplingInput):
    drive: PulseCfg


def test_declared_root_converter_and_generated_module_converter_both_use_fixed_env() -> (
    None
):
    env = replace(indexed_env(), parameters=parameter_capture(seed=17.0))
    form = Form(NativeInput, {"drive": Slot("Q.pulses.drive")})
    state = FormState(form.default_input(env))
    state.select(("sampling",), {"kind": "captured"}, env)
    assert state.input.data["sampling"] == {"kind": "captured", "start": 17.0}
    state.select(("drive",), {"kind": "inline", "module": {"type": "pulse"}}, env)
    drive = state.input.data["drive"]
    assert isinstance(drive, dict) and drive["kind"] == "inline"
    module = drive["module"]
    assert isinstance(module, dict) and module["freq"] == 100.0
