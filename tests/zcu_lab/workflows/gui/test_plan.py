"""Reference conversion and runtime T1 generation at public form seams."""

from collections.abc import Iterator

import numpy as np
import pytest
from pydantic import JsonValue
from zcu_tools.cfg_model import ConstWaveformCfg, PulseSpec, WaveformSamples
from zcu_tools.gui.form import FormEnv, FormErrors, FormInput, FormState
from zcu_tools.gui.session.context_types import EntryDraft
from zcu_tools.gui.session.parameters import ParameterContext, RoleBinding
from zcu_tools.program.v2 import SweepCfg
from zcu_tools.resources.context.library import LibrarySnapshot, ModuleLibraryDocument
from zcu_tools.resources.entry import RoleSpec, component_registry
from zcu_tools.resources.entry.schema import PointDocument

from tests.program.v2.support.waveform_assets import InMemoryWaveformAsset
from tests.zcu_lab.workflows._sim_inputs import Inputs, make_plan, seed_from_inputs
from zcu_lab.components import QubitSchema
from zcu_lab.workflows.gui.plan import T1PlanForm
from zcu_lab.workflows.t1 import T1Plan, T1Tunables, t1_cfg

from ._plan_inputs import inline_input, library_env


@pytest.fixture
def captured_pi_env(inputs: Inputs) -> Iterator[FormEnv]:
    """Use an isolated registered qubit model and fixed x180 module capture."""
    before = component_registry.kinds()
    kind = "workflows/test-qubit"
    component_registry.register(kind, QubitSchema)
    try:
        library = library_env(inputs).library
        assert library is not None
        spec = library.module_spec("Q1.pulses.pi")
        assert isinstance(spec, PulseSpec)
        library = library.replaced_module(
            "Q1.pulses.pi",
            spec.model_copy(update={"waveform": ConstWaveformCfg(length=0.12)}),
        )
        parameters = ParameterContext(
            point=PointDocument.model_validate(
                {
                    "format": "zcu.parameter-container",
                    "format_version": "1.0",
                    "general": {"created_at": "2026-10-09T00:00:00Z"},
                    "components": {
                        "Q1": {"kind": kind, "module": {"x180": "Q1.pulses.pi"}}
                    },
                }
            ),
            library=library,
            target=EntryDraft(entry_id="entry"),
            roles=RoleBinding("Q1", {"qubit": RoleSpec(kind)}, {"qubit": "Q1"}),
        )
        yield FormEnv(parameters=parameters, library=library)
    finally:
        component_registry.unregister(kind)
        assert component_registry.kinds() == before


def test_pi_default_uses_captured_x180_and_remains_user_editable(
    inputs: Inputs, captured_pi_env: FormEnv
) -> None:
    env = captured_pi_env
    default = T1PlanForm.default_input(env)
    assert default.data["pi_length_us"] == {"expr": "qubit.module.x180.waveform.length"}
    data = inline_input(inputs).data
    data["pi_length_us"] = default.data["pi_length_us"]
    state = FormState(FormInput(T1PlanForm, data))
    result = state.result(env)
    assert isinstance(result, T1PlanForm)
    assert result.pi_length_us == pytest.approx(0.12)
    state.write({("pi_length_us",): 0.25}, env)
    changed = state.result(env)
    assert isinstance(changed, T1PlanForm)
    assert changed.pi_length_us == 0.25
    assert T1PlanForm.defaults(env, {"pi_length_us": 0.25}).get("pi_length_us") is None


@pytest.mark.parametrize("slot", [None, "Q1.pulses.absent", "unbound-role"])
def test_missing_or_dangling_x180_leaves_pi_required(
    inputs: Inputs, captured_pi_env: FormEnv, slot: str | None
) -> None:
    parameters = captured_pi_env.parameters
    assert parameters is not None
    point = parameters.document()
    qubit = point.components["Q1"]
    assert isinstance(qubit, QubitSchema)
    qubit.module = {} if slot is None else {"x180": slot}
    env = FormEnv(
        parameters=ParameterContext(
            point=point,
            library=parameters.library,
            target=parameters.target,
            roles=None if slot == "unbound-role" else parameters.roles,
        ),
        library=parameters.library,
    )
    data = inline_input(inputs).data
    del data["pi_length_us"]
    data.update(T1PlanForm.default_input(env).data)
    result = FormState(FormInput(T1PlanForm, data)).result(env)
    assert isinstance(result, FormErrors)
    assert ("pi_length_us",) in result.messages


@pytest.mark.parametrize("value", [None, 0.0, -0.1])
def test_invalid_pi_input_blocks_submission(
    inputs: Inputs, value: float | None
) -> None:
    data = inline_input(inputs).data
    data["pi_length_us"] = value
    result = FormState(FormInput(T1PlanForm, data)).result(FormEnv())
    assert isinstance(result, FormErrors)
    assert ("pi_length_us",) in result.messages


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
def test_form_setpoints_and_modules_reach_runtime_t1_cfg(
    inputs: Inputs, fluxes: dict[str, JsonValue], expected: tuple[float, ...]
) -> None:
    template = make_plan(inputs)
    data = inline_input(inputs).data
    data["fluxes"] = fluxes
    state = FormState(FormInput(T1PlanForm, data))
    env = FormEnv()
    result = state.result(env)
    assert isinstance(result, T1PlanForm)
    native = result.to_cfg(env, template.snapshot)
    assert isinstance(native, T1Plan)
    assert native.fluxes == pytest.approx(expected)
    assert native.qub_pulse == template.qub_pulse
    assert native.readout == template.readout
    assert native.snapshot == template.snapshot
    cfg = t1_cfg(native, T1Tunables(sweep_factor=4.0), seed_from_inputs(inputs), 20.0)
    assert isinstance(cfg.sweep.length, SweepCfg)
    assert cfg.sweep.length.expts == native.points
    assert cfg.sweep.length.stop == 80.0
    assert cfg.uniform
    assert state.input.data == data


def test_missing_selections_and_point_errors_remain_directed(inputs: Inputs) -> None:
    env = FormEnv()
    missing = FormState(T1PlanForm.default_input(env)).result(env)
    assert isinstance(missing, FormErrors)
    assert {("flux_dev",), ("fluxes",), ("qub_pulse",), ("readout",)} <= set(
        missing.messages
    )
    data = inline_input(inputs).data
    data["points"] = 4
    state = FormState(FormInput(T1PlanForm, data))
    invalid = state.result(env)
    assert isinstance(invalid, FormErrors)
    assert ("points",) in invalid.messages
    state.write({("points",): 5}, env)
    assert isinstance(state.result(env), T1PlanForm)


def test_refs_materialize_and_inline_switch_detaches_the_selected_drive(
    inputs: Inputs,
) -> None:
    env = library_env(inputs)
    data = inline_input(inputs).data
    data["qub_pulse"] = {"kind": "ref", "name": "Q1.pulses.pi"}
    data["readout"] = {"kind": "ref", "name": "Q1.readout.ro"}
    state = FormState(FormInput(T1PlanForm, data))
    result = state.result(env)
    assert isinstance(result, T1PlanForm)
    native = result.to_cfg(env, make_plan(inputs).snapshot)
    assert isinstance(native, T1Plan)
    assert native.qub_pulse == inputs.rabi.modules.qub_pulse
    assert native.readout == inputs.readout.modules.readout
    state.write({("qub_pulse", "kind"): "inline"}, env)
    state.write({("qub_pulse", "module", "gain"): 0.5}, env)
    assert env.library is not None
    spec = env.library.module_spec("Q1.pulses.pi")
    assert isinstance(spec, PulseSpec)
    changed = FormEnv(
        library=env.library.replaced_module(
            "Q1.pulses.pi", spec.model_copy(update={"gain": 0.75})
        )
    )
    result = state.result(changed)
    assert isinstance(result, T1PlanForm)
    detached = result.to_cfg(changed, make_plan(inputs).snapshot)
    assert isinstance(detached, T1Plan)
    assert detached.qub_pulse.gain == 0.5


@pytest.mark.parametrize(
    "failure", ["missing_ref", "wrong_readout_family", "missing_waveform"]
)
def test_bad_module_selections_block_plan_with_consuming_paths(
    inputs: Inputs, failure: str
) -> None:
    env = library_env(inputs)
    data = inline_input(inputs).data
    if failure == "missing_ref":
        data["qub_pulse"] = {"kind": "ref", "name": "Q1.pulses.absent"}
    elif failure == "wrong_readout_family":
        data["readout"] = {"kind": "ref", "name": "Q1.pulses.pi"}
    else:
        raw = inputs.rabi.modules.qub_pulse.model_dump(mode="json")
        raw["waveform"] = "Q1.waveforms.absent"
        data["qub_pulse"] = {"kind": "inline", "module": raw}
    result = FormState(FormInput(T1PlanForm, data)).result(env)
    if isinstance(result, T1PlanForm):
        result = result.to_cfg(env, make_plan(inputs).snapshot)
    assert isinstance(result, FormErrors)
    field = "readout" if failure == "wrong_readout_family" else "qub_pulse"
    assert any(path and path[0] == field for path in result.messages)


@pytest.mark.parametrize("selection", ["drive", "readout", "readout_ramp"])
def test_asset_backed_selection_is_rejected_at_consuming_field(
    inputs: Inputs, selection: str
) -> None:
    env = FormEnv(
        library=LibrarySnapshot(
            ModuleLibraryDocument(format="zcu.module-library", format_version="1.0"),
            assets=InMemoryWaveformAsset(
                "present",
                WaveformSamples(
                    time=np.array([0.0, 0.05]),
                    idata=np.array([0.0, 1.0]),
                    qdata=None,
                ),
            ),
        )
    )
    data = inline_input(inputs).data
    waveform = {"style": "arb", "data": "present"}
    field = "qub_pulse" if selection == "drive" else "readout"
    if selection == "drive":
        raw = inputs.rabi.modules.qub_pulse.model_dump(mode="json")
        raw["waveform"] = waveform
    else:
        raw = inputs.readout.modules.readout.model_dump(mode="json")
        raw["pulse_cfg"]["waveform"] = (
            {"style": "flat_top", "length": 1.0, "raise_waveform": waveform}
            if selection == "readout_ramp"
            else waveform
        )
    data[field] = {"kind": "inline", "module": raw}
    form = FormState(FormInput(T1PlanForm, data)).result(env)
    assert isinstance(form, T1PlanForm), form
    result = form.to_cfg(env, make_plan(inputs).snapshot)
    assert isinstance(result, FormErrors)
    assert (field,) in result.messages
    assert "arb asset" in result.messages[(field,)].lower()
