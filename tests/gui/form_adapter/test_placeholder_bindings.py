"""Placeholder defaults are fixed host captures, not process-global policy."""

from dataclasses import replace

import pytest
from zcu_tools import cfg_model
from zcu_tools.gui.form import FormEnv, FormErrors, FormState
from zcu_tools.gui.form_adapter import Form


class SyntheticCore(cfg_model.ConfigBase):
    probe: cfg_model.PulseCfg


def pulse_template() -> cfg_model.TemplateDef:
    return cfg_model.TemplateDef(
        cfg_model.PulseSpec,
        "Synthetic",
        {
            "ch": 0,
            "nqz": 1,
            "freq": cfg_model.Expr("{drive} + {sensor}"),
            "gain": 0.2,
            "waveform.style": "const",
            "waveform.length": 0.1,
        },
    )


def frequency(cfg: cfg_model.ConfigBase | FormErrors) -> float:
    """Read the synthetic pulse result through either public Form projection."""
    assert not isinstance(cfg, FormErrors), cfg
    if isinstance(cfg, SyntheticCore):
        value = cfg.probe.freq
    else:
        assert isinstance(cfg, cfg_model.PulseSpec), cfg
        value = cfg.freq
    assert isinstance(value, (float, int)), value
    return float(value)


@pytest.mark.parametrize("nested", [False, True], ids=["spec", "core"])
def test_template_bindings_override_defaults_and_keep_fixed_capture(
    nested: bool,
) -> None:
    registry = cfg_model.PlaceholderRegistry()
    registry.register("drive", "100")
    registry.register("sensor", "20")
    env = FormEnv(placeholders=registry.current())
    template = cfg_model.Template("synthetic", bindings={"drive": "300"})
    catalog = {"synthetic": pulse_template()}
    form: Form[SyntheticCore] | Form[cfg_model.PulseSpec]
    if nested:
        form = Form(SyntheticCore, {"probe": template})
        form.bind_templates(catalog, adapter="synthetic")
        initial = form.default_input(env)
    else:
        form = Form(cfg_model.PulseSpec, {})
        initial = form.template_input(template, catalog, env)

    registry.register("sensor", "50")
    old_value = FormState(initial.model.default_input(env)).result(env)
    assert not isinstance(old_value, FormErrors), old_value
    old_cfg = form.to_cfg(old_value, env)
    assert frequency(old_cfg) == 320

    new_env = replace(env, placeholders=registry.current())
    new_value = FormState(initial.model.default_input(new_env)).result(new_env)
    assert not isinstance(new_value, FormErrors), new_value
    new_cfg = form.to_cfg(new_value, new_env)
    assert frequency(new_cfg) == 350


def test_template_binding_is_required_only_when_building_input() -> None:
    try:
        definition = pulse_template()
        form = Form(SyntheticCore, {"probe": cfg_model.Template("synthetic")})
        form.bind_templates({"synthetic": definition}, adapter="synthetic")
    except ValueError as exc:
        pytest.fail(
            f"Construction and catalog binding must not require bindings: {exc}"
        )
    with pytest.raises(ValueError, match="drive"):
        form.default_input(FormEnv())
