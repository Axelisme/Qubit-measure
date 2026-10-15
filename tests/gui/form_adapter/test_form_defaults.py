"""Templates and slots resolve captured sources into editable persistent inputs."""

import pytest
from zcu_tools.cfg_model import (
    AliasName,
    BathResetSpec,
    ConfigBase,
    Expr,
    PulseReadoutSpec,
    PulseSpec,
    Slot,
    Template,
    TemplateDef,
)
from zcu_tools.device import BaseDeviceInfo
from zcu_tools.gui.form import FormEnv, FormErrors, FormState, ModuleInline
from zcu_tools.gui.form_adapter import Form
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_types import SavedPoint
from zcu_tools.gui.session.parameters import ParameterContext
from zcu_tools.program.v2.modules import BathResetCfg, PulseCfg, PulseReadoutCfg
from zcu_tools.resources.context.library import (
    ComponentLibrary,
    LibrarySnapshot,
    ModuleLibraryDocument,
)
from zcu_tools.resources.entry.schema import PointDocument, PointGeneral


class Core(ConfigBase):
    """A required native pulse field has no GUI-specific declaration."""

    probe: PulseCfg


def templates() -> dict[str, TemplateDef]:
    """A deliberately partial persistent template supplies a required child."""
    return {
        "pulse": TemplateDef(
            PulseSpec,
            "Pulse",
            {
                "ch": 2,
                "nqz": 1,
                "freq": Expr("100 + 20"),
                "gain": 0.1,
                "waveform.style": "const",
                "waveform.length": Expr("2 * .2"),
            },
        )
    }


def environment() -> FormEnv:
    """Capture a library plus an explicitly unbound alias, without source I/O."""
    pulse = PulseSpec.model_validate(
        {
            "ch": 1,
            "nqz": 1,
            "freq": 100,
            "gain": 0.3,
            "waveform": {"style": "const", "length": 0.1},
        }
    )
    library = LibrarySnapshot(
        ModuleLibraryDocument(
            format="zcu.module-library",
            format_version="2.0",
            components={"Q": ComponentLibrary(pulses={"probe": pulse})},
        )
    )
    parameters = ParameterContext(
        point=PointDocument(
            format="zcu.parameter-container",
            format_version="1.0",
            general=PointGeneral(created_at="2026-10-13T00:00:00Z"),
        ),
        library=library,
        target=SavedPoint(entry_id="synthetic", point="synthetic"),
        roles=None,
        storage=None,
    ).bind_aliases((AliasRow("unbound", ""),), {})
    return FormEnv(parameters=parameters, library=library)


def bound_form(node: Slot | Template) -> Form[Core]:
    """Bind the synthetic template catalog through the public Form seam."""
    form = Form(Core, {"probe": node})
    form.bind_templates(templates(), adapter="synthetic")
    return form


@pytest.mark.parametrize(
    "scenario",
    [
        "key",
        "copy",
        "dangling_empty",
        "unbound_empty",
    ],
)
def test_slot_resolution_ref_copy_or_invalid(scenario: str) -> None:
    node = (
        Slot("Q.pulses.probe", copy=scenario == "copy")
        if scenario in {"key", "copy"}
        else Slot(
            "$unbound.module.probe"
            if scenario.startswith("unbound")
            else "Q.pulses.missing",
        )
    )
    form = bound_form(node)
    env = environment()
    state = FormState(form.default_input(env))
    value = state.result(env)
    if scenario.endswith("empty"):
        assert isinstance(value, FormErrors)
        assert ("probe",) in value.messages
        return
    assert not isinstance(value, FormErrors), value
    cfg = form.to_cfg(value, env)
    assert isinstance(cfg, Core), cfg
    assert cfg.probe.ch == 1
    probe = state.input.data["probe"]
    assert isinstance(probe, dict)
    assert probe["kind"] == ("ref" if scenario == "key" else "inline")
    if scenario == "key":
        assert probe["name"] == "Q.pulses.probe"


def test_template_required_nested_defaults_omit_none() -> None:
    form = bound_form(Template("pulse"))
    env = environment()
    state = FormState(form.default_input(env))
    probe = state.input.data["probe"]
    assert isinstance(probe, dict)
    module = probe["module"]
    assert isinstance(module, dict)
    assert module["phase"] == 0
    assert "mixer_freq" not in module
    assert "mux_freqs" not in module
    assert module["freq"] == 120
    value = state.result(env)
    assert not isinstance(value, FormErrors), value
    cfg = form.to_cfg(value, env)
    assert isinstance(cfg, Core), cfg
    assert cfg.probe.freq == 120 and cfg.probe.waveform.length == 0.4


def test_copy_command_does_not_reapply_initial_set() -> None:
    form = bound_form(Slot("Q.pulses.probe", copy=True, set={"gain": 0.8}))
    env = environment()
    state = FormState(form.default_input(env))
    value = state.result(env)
    assert not isinstance(value, FormErrors), value
    initial = form.to_cfg(value, env)
    assert isinstance(initial, Core) and initial.probe.gain == 0.8
    state.write({("probe", "kind"): "ref", ("probe", "name"): "Q.pulses.probe"}, env)
    state.run_command(("probe",), "to_inline", env)
    value = state.result(env)
    assert not isinstance(value, FormErrors), value
    copied = form.to_cfg(value, env)
    assert isinstance(copied, Core) and copied.probe.gain == 0.3


def test_locked_slot_is_inline_and_branch_defaults_reapply_only_lock() -> None:
    form = bound_form(Slot("Q.pulses.probe", set={"gain": 0.8}, lock={"freq": 0.0}))
    env = environment()
    state = FormState(form.default_input(env))
    value = state.result(env)
    assert not isinstance(value, FormErrors), value
    cfg = form.to_cfg(value, env)
    assert isinstance(cfg, Core) and cfg.probe.freq == 0 and cfg.probe.gain == 0.8
    inline = form.model.model_fields["probe"].annotation
    assert isinstance(inline, type) and issubclass(inline, ModuleInline)
    defaults = inline.defaults(env)
    module = defaults["module"]
    assert isinstance(module, dict) and module["freq"] == 0
    state.write({("probe", "kind"): "ref"}, env)
    assert isinstance(state.result(env), FormErrors)


@pytest.mark.parametrize("expression", ["$unbound.value", "'not a number'", "2 * 3"])
def test_root_slot_expression_overrides_require_input(
    expression: str,
) -> None:
    form = bound_form(
        Slot("Q.pulses.probe", copy=True, set={"phase": Expr(expression)})
    )
    env = environment()
    state = FormState(form.default_input(env))
    result = state.result(env)
    if expression != "2 * 3":
        assert isinstance(result, FormErrors), result
        assert result.messages == {("probe", "module", "phase"): "Field required"}
        state.write({("probe", "module", "phase"): 4}, env)
        result = state.result(env)
    if isinstance(result, FormErrors):
        pytest.fail(str(result.messages))
    cfg = form.to_cfg(result, env)
    if isinstance(cfg, FormErrors):
        pytest.fail(str(cfg.messages))
    assert cfg.probe.phase == (6 if expression == "2 * 3" else 4)
    control = bound_form(Slot("Q.pulses.probe", copy=True))
    control_result = FormState(control.default_input(env)).result(env)
    if isinstance(control_result, FormErrors):
        pytest.fail(str(control_result.messages))
    control_cfg = control.to_cfg(control_result, env)
    if isinstance(control_cfg, FormErrors):
        pytest.fail(str(control_cfg.messages))
    assert control_cfg.probe.phase == 0


class OptionalCore(ConfigBase):
    """Core nullability permits a missing slot to remain intentionally disabled."""

    probe: PulseCfg | None = None


@pytest.mark.parametrize("optional", [False, True])
def test_unbound_slot_obeys_core_required_or_optional_default(optional: bool) -> None:
    cfg_type = OptionalCore if optional else Core
    form = Form(cfg_type, {"probe": Slot("$unbound.module.probe")})
    form.bind_templates(templates(), adapter="synthetic-optional")
    env = environment()
    state = FormState(form.default_input(env))
    assert state.input.data == {}
    value = state.result(env)
    if not optional:
        assert isinstance(value, FormErrors)
        assert ("probe",) in value.messages
        return
    assert not isinstance(value, FormErrors), value
    assert form.to_cfg(value, env) == OptionalCore(probe=None)


class ResetCore(ConfigBase):
    """Nested references are part of the persistent reset Spec, not form modes."""

    reset: BathResetCfg


@pytest.mark.parametrize("unbound", [False, True])
def test_template_nested_slot_is_direct_key_or_unfilled_invalid(unbound: bool) -> None:
    catalog = {
        "reset": TemplateDef(
            BathResetSpec,
            "Reset",
            {
                "cavity_tone_cfg": Slot("Q.pulses.probe"),
                "qubit_tone_cfg": Slot("Q.pulses.probe"),
                "pi2_cfg": Slot(
                    "$unbound.module.probe" if unbound else "Q.pulses.probe"
                ),
            },
        )
    }
    form = Form(ResetCore, {"reset": Template("reset")})
    form.bind_templates(catalog, adapter="synthetic-reset")
    env = environment()
    state = FormState(form.default_input(env))
    value = state.result(env)
    if unbound:
        assert isinstance(value, FormErrors)
        return
    assert not isinstance(value, FormErrors), value
    reset = state.input.data["reset"]
    assert isinstance(reset, dict)
    module = reset["module"]
    assert isinstance(module, dict) and module["pi2_cfg"] == {
        "kind": "ref",
        "name": "Q.pulses.probe",
    }
    cfg = form.to_cfg(value, env)
    assert isinstance(cfg, ResetCore), cfg
    assert cfg.reset.pi2_cfg.gain == 0.3


class ReadoutCore(ConfigBase):
    """A synthetic core owns one flat pulse readout."""

    readout: PulseReadoutCfg


def readout_expression_form(expression: Expr | None) -> Form[ReadoutCore]:
    """Author nested defaults on existing persistent Spec families only."""
    seeds = {
        "type": "readout/pulse",
        "gen_ch": 1,
        "nqz": 1,
        "freq": 100.0,
        "gain": 0.3,
        "waveform.style": "cosine",
        "waveform.length": expression if expression is not None else 0.2,
        "ro_ch": 0,
        "ro_length": 0.2,
    }
    if expression is not None:
        seeds["trig_offset"] = expression
        seeds["phase"] = expression
    form = Form(ReadoutCore, {"readout": Template("synthetic-readout")})
    form.bind_templates(
        {"synthetic-readout": TemplateDef(PulseReadoutSpec, "Readout", seeds)},
        adapter="synthetic",
    )
    return form


@pytest.mark.parametrize("expression", ["$unbound.value", "'not a number'"])
def test_nested_template_expression_failure_requires_input(expression: str) -> None:
    env = FormEnv()
    form = readout_expression_form(Expr(expression))
    state = FormState(form.default_input(env))
    result = state.result(env)
    assert isinstance(result, FormErrors), result
    assert result.messages == {
        ("readout", "module", "trig_offset"): "Field required",
        ("readout", "module", "phase"): "Field required",
        (
            "readout",
            "module",
            "waveform",
            "module",
            "length",
        ): "Field required",
    }
    state.write(
        {
            ("readout", "module", "trig_offset"): 0.4,
            ("readout", "module", "phase"): 0.4,
            (
                "readout",
                "module",
                "waveform",
                "module",
                "length",
            ): 0.4,
        },
        env,
    )
    result = state.result(env)
    if isinstance(result, FormErrors):
        pytest.fail(str(result.messages))
    cfg = form.to_cfg(result, env)
    if isinstance(cfg, FormErrors):
        pytest.fail(str(cfg.messages))
    assert cfg.readout.trig_offset == 0.4
    assert cfg.readout.phase == 0.4
    assert cfg.readout.waveform.length == 0.4


@pytest.mark.parametrize("expression", [None, Expr("2 * .2")])
def test_nested_template_expression_and_static_defaults_are_isolated(
    expression: Expr | None,
) -> None:
    env = FormEnv()
    failed = readout_expression_form(Expr("$unbound.value"))
    form = readout_expression_form(expression)
    state = FormState(form.default_input(env))
    readout = state.input.data["readout"]
    assert isinstance(readout, dict)
    module = readout["module"]
    assert isinstance(module, dict)
    waveform_wrapper = module["waveform"]
    assert isinstance(waveform_wrapper, dict)
    waveform = waveform_wrapper["module"]
    assert isinstance(waveform, dict)
    expected = 0.0 if expression is None else 0.4
    assert module["trig_offset"] == module["phase"] == expected
    assert waveform["length"] == (0.2 if expression is None else 0.4)
    result = state.result(env)
    if isinstance(result, FormErrors):
        pytest.fail(str(result.messages))
    cfg = form.to_cfg(result, env)
    if isinstance(cfg, FormErrors):
        pytest.fail(str(cfg.messages))
    assert cfg.readout.trig_offset == cfg.readout.phase == expected
    assert cfg.readout.waveform.length == waveform["length"]
    # Another form's failed seeds must not affect this form or persistent Specs.
    failed.default_input(env)
    native = PulseReadoutCfg.model_validate(
        {
            "gen_ch": 1,
            "nqz": 1,
            "freq": 100,
            "gain": 0.3,
            "waveform": {"style": "cosine", "length": 0.2},
            "ro_ch": 0,
            "ro_length": 0.2,
        }
    )
    assert native.trig_offset == native.phase == 0
    assert state.result(env) == result


@pytest.mark.parametrize("nested", [False, True])
def test_missing_discriminator_error_names_template_and_nested_field(
    nested: bool,
) -> None:
    template_id = "partial-readout" if nested else "partial-pulse"
    spec = PulseReadoutSpec if nested else PulseSpec
    path = "waveform"
    form = (
        Form(ReadoutCore, {"readout": Template(template_id)})
        if nested
        else Form(Core, {"probe": Template(template_id)})
    )
    catalog = {template_id: TemplateDef(spec, "Partial", {f"{path}.length": 5.1})}
    with pytest.raises(ValueError, match="synthetic") as caught:
        form.bind_templates(catalog, adapter="synthetic")
    message = str(caught.value)
    assert template_id in message, message
    assert path in message, message
    assert "discriminator" in message.lower(), message


def test_explicit_discriminator_and_partial_seeds_keep_spec_defaults() -> None:
    catalog = {
        "synthetic-pulse": TemplateDef(
            PulseSpec,
            "Partial pulse",
            {
                "ch": 2,
                "nqz": 1,
                "freq": Expr("100 + 20"),
                "gain": 0.1,
                "waveform.style": "cosine",
                "waveform.length": 5.1,
            },
        )
    }
    form = Form(Core, {"probe": Template("synthetic-pulse", set={"gain": 0.7})})
    form.bind_templates(catalog, adapter="synthetic")
    env = FormEnv()
    state = FormState(form.default_input(env))
    result = state.result(env)
    if isinstance(result, FormErrors):
        pytest.fail(str(result.messages))
    cfg = form.to_cfg(result, env)
    if isinstance(cfg, FormErrors):
        pytest.fail(str(cfg.messages))
    expected = Core.model_validate(
        {
            "probe": {
                "ch": 2,
                "nqz": 1,
                "freq": 120,
                "gain": 0.7,
                "waveform": {"style": "cosine", "length": 5.1},
            }
        }
    )
    assert cfg.model_dump(mode="json") == expected.model_dump(mode="json")
    assert cfg.probe.waveform.style == "cosine"


@pytest.mark.parametrize("fault", ["unknown", "family", "tag"])
def test_template_binding_error_names_adapter_and_field_path(fault: str) -> None:
    node = Template("missing") if fault == "unknown" else Template("pulse")
    catalog = templates()
    if fault == "family":
        from zcu_tools.cfg_model import NoneResetCfg

        catalog["pulse"] = TemplateDef(NoneResetCfg, "Wrong family", {})
    elif fault == "tag":
        catalog["pulse"] = TemplateDef(
            PulseSpec, "Bad tag", {"waveform.style": "unknown"}
        )
    form = Form(Core, {"probe": node})
    with pytest.raises(ValueError, match="synthetic.*probe"):
        form.bind_templates(catalog, adapter="synthetic")


@pytest.mark.parametrize("nested", [False, True], ids=["root", "nested"])
def test_unsupported_template_seed_names_template_and_seed_path(nested: bool) -> None:
    pulse = {
        "ch": 1,
        "nqz": 1,
        "freq": 100.0,
        "gain": 0.3,
        "waveform.style": "const",
        "waveform.length": 0.2,
    }
    if nested:
        seeds = {**pulse, "desc": "note"}
        definition = TemplateDef(
            BathResetSpec,
            "Noisy reset",
            {f"cavity_tone_cfg.{path}": value for path, value in seeds.items()},
        )
        form = Form(ResetCore, {"reset": Template("noisy")})
        path = "cavity_tone_cfg.desc"
    else:
        definition = TemplateDef(PulseSpec, "Noisy pulse", {**pulse, "desc": "note"})
        form = Form(Core, {"probe": Template("noisy")})
        path = "desc"

    with pytest.raises(ValueError, match="noisy") as caught:
        form.bind_templates({"noisy": definition}, adapter="synthetic")

    assert path in str(caught.value), caught.value


class DefaultedDeviceNames(ConfigBase):
    """Device aliases retain their own static defaults, including nonempty ones."""

    device: str = ""
    nullable_device: str | None = "standby"


@pytest.mark.parametrize("binding", [None, "", "rf_source"])
def test_alias_static_default_is_valid_and_selectable(binding: str | None) -> None:
    captured = environment()
    parameters = captured.parameters
    assert parameters is not None
    env = FormEnv(
        parameters=(
            None
            if binding is None
            else parameters.bind_aliases((AliasRow("rf_dev", binding),), {})
        ),
        devices={
            name: BaseDeviceInfo(address=name, type="synthetic")
            for name in (
                () if binding is None else ("rf_source", "other_source", "standby")
            )
        },
    )
    form = Form(
        DefaultedDeviceNames,
        {"device": AliasName("rf_dev"), "nullable_device": AliasName("rf_dev")},
    )
    state = FormState(form.default_input(env))
    assert state.input.data == (
        {"device": binding, "nullable_device": binding} if binding else {}
    )
    result = state.result(env)
    assert not isinstance(result, FormErrors), result
    expected = (
        DefaultedDeviceNames(device=binding, nullable_device=binding)
        if binding
        else DefaultedDeviceNames()
    )
    assert form.to_cfg(result, env) == expected

    properties = form.model.input_schema(env)["properties"]
    assert isinstance(properties, dict)
    for name, default in (("device", ""), ("nullable_device", "standby")):
        field = properties[name]
        assert isinstance(field, dict)
        options = field["x-form-options"]
        assert isinstance(options, list)
        values = [option["value"] for option in options if isinstance(option, dict)]
        assert default in values, "The core static default must be selectable"
        assert values.count(default) == 1
        assert all(device in values for device in env.devices)
        assert (None in values) == (name == "nullable_device")
        assert tuple(values) == form.model.choices(env)[(name,)]
        for value in values:
            state.write({(name,): value}, env)
            result = state.result(env)
            if isinstance(result, FormErrors):
                pytest.fail(str(result.messages))
            cfg = form.to_cfg(result, env)
            if isinstance(cfg, FormErrors):
                pytest.fail(str(cfg.messages))
            assert getattr(cfg, name) == value
        state.write({(name,): "unknown_device"}, env)
        rejected = state.result(env)
        assert isinstance(rejected, FormErrors)
        assert rejected.messages == {(name,): "Value is not among current choices"}
        state.write({(name,): default}, env)
