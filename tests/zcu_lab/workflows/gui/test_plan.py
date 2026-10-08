"""Native plan conversion at the public form and T1 cfg seams."""

import pytest
from pydantic import JsonValue
from zcu_tools.gui.form import FormEnv, FormErrors, FormInput, FormState
from zcu_tools.program.v2 import SweepCfg

from tests.zcu_lab.workflows._sim_inputs import Inputs, make_plan, seed_from_inputs
from zcu_lab.workflows.gui.plan import T1PlanForm
from zcu_lab.workflows.t1 import T1Tunables, t1_cfg


@pytest.mark.parametrize(
    ("fluxes", "expected"),
    [
        ({"kind": "linrange", "start": 0.1, "stop": 0.3, "n": 3}, (0.1, 0.2, 0.3)),
        ({"kind": "linrange", "start": 0.3, "stop": 0.1, "n": 3}, (0.3, 0.2, 0.1)),
        ({"kind": "linrange", "start": 0.1, "stop": 0.3, "n": 1}, (0.1,)),
        ({"kind": "linrange", "start": 0.1, "stop": 0.1, "n": 3}, (0.1, 0.1, 0.1)),
        ({"kind": "list", "values": [0.3, 0.1, 0.3]}, (0.3, 0.1, 0.3)),
    ],
)
def test_form_setpoints_and_templates_reach_native_t1_cfg(
    inputs: Inputs,
    fluxes: dict[str, JsonValue],
    expected: tuple[float, ...],
) -> None:
    template = make_plan(inputs)
    data = FormInput.from_json(T1PlanForm, template.model_dump_json()).data
    data.pop("snapshot")
    data["fluxes"] = fluxes
    state = FormState(FormInput(T1PlanForm, data))
    result = state.result(FormEnv())
    assert isinstance(result, T1PlanForm)
    native = result.to_cfg(template.snapshot)
    assert native.fluxes == pytest.approx(expected)
    assert native.calibration == template.calibration
    assert native.snapshot == template.snapshot
    cfg = t1_cfg(native, T1Tunables(), seed_from_inputs(inputs), 20.0)
    assert isinstance(cfg.sweep.length, SweepCfg)
    assert cfg.sweep.length.expts == native.points
    assert cfg.uniform
    assert state.input.data == data


def test_required_templates_remain_absent_and_invalid_fields_are_directed(
    inputs: Inputs,
) -> None:
    env = FormEnv()
    missing = FormState(T1PlanForm.default_input(env)).result(env)
    assert isinstance(missing, FormErrors)
    assert {("flux_dev",), ("fluxes",), ("calibration",), ("t1",)} <= set(
        missing.messages
    )
    template = make_plan(inputs)
    data = FormInput.from_json(T1PlanForm, template.model_dump_json()).data
    data.pop("snapshot")
    data["fluxes"] = {"kind": "list", "values": [1.0]}
    data["points"] = 4
    state = FormState(FormInput(T1PlanForm, data))
    invalid = state.result(env)
    assert isinstance(invalid, FormErrors)
    assert ("points",) in invalid.messages
    state.write({("points",): 5, ("t1", "uniform"): False}, env)
    nonuniform = state.result(env)
    assert isinstance(nonuniform, FormErrors)
    assert ("t1", "uniform") in nonuniform.messages
    state.write({("t1", "uniform"): True}, env)
    assert isinstance(state.result(env), T1PlanForm)
