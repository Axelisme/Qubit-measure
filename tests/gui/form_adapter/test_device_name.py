"""Instrument-name annotations use captured suggestions, not legal-value choices."""

from typing import Annotated

from zcu_tools.cfg_model import ConfigBase, DeviceName, FieldMeta
from zcu_tools.device import BaseDeviceInfo
from zcu_tools.gui.form import FormEnv, FormErrors, FormModel, FormState
from zcu_tools.gui.form_adapter import DeviceName as FormDeviceName
from zcu_tools.gui.form_adapter import Form


class DeviceCfg(ConfigBase):
    flux: Annotated[DeviceName, FieldMeta(label="Flux source")] = "offline"
    rf: DeviceName | None = None


class DevicePlan(FormModel):
    flux: FormDeviceName = "offline"
    rf: FormDeviceName | None = None


def validated_cfg(form: Form[DeviceCfg], state: FormState[FormModel], env: FormEnv):
    value = state.result(env)
    assert not isinstance(value, FormErrors), value
    return form.to_cfg(value, env)


def test_generated_device_names_use_capture_and_keep_nullable_and_unknown_values():
    env = FormEnv(
        devices={
            "coil": BaseDeviceInfo(address="offline-coil", type="fake"),
            "pump": BaseDeviceInfo(address="offline-pump", type="fake"),
        }
    )
    form = Form(DeviceCfg, {})
    state = FormState(form.default_input(env))
    flux, rf = state.describe((), env).children
    assert flux.title == "Flux source"
    assert flux.options is not None
    assert [item.value for item in flux.options] == ["coil", "pump"]
    assert flux.choices is None
    assert rf.options is not None
    assert [item.value for item in rf.options] == [None, "coil", "pump"]
    assert validated_cfg(form, state, env) == DeviceCfg()
    state.write({("rf",): "unregistered"}, env)
    assert validated_cfg(form, state, env) == DeviceCfg(rf="unregistered")
    state.select(("rf",), None, env)
    assert validated_cfg(form, state, env) == DeviceCfg()
    plan = FormState(DevicePlan.default_input(env))
    options = plan.describe((), env).children[0].options
    assert options is not None
    assert [item.value for item in options] == ["coil", "pump"]
    optional = plan.describe(("rf",), env).target
    assert optional is not None and optional.options is not None
    assert [item.value for item in optional.options] == [None, "coil", "pump"]
    assert optional.commands[0].name == "toggle_text"
    plan.run_command(("rf",), "toggle_text", env)
    assert plan.result(env) == DevicePlan()
