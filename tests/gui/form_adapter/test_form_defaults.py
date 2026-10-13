"""Templates and slots resolve captured sources into editable persistent inputs."""

import pytest
from zcu_tools.cfg_model import (
    BathResetSpec,
    ConfigBase,
    Expr,
    PulseReadoutSpec,
    PulseSpec,
    Slot,
    Template,
    TemplateDef,
)
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
            format_version="1.0",
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
        "dangling_fallback",
        "unbound_fallback",
        "dangling_empty",
        "unbound_empty",
    ],
)
def test_slot_resolution_ref_copy_fallback_or_invalid(scenario: str) -> None:
    node = (
        Slot("Q.pulses.probe", copy=scenario == "copy")
        if scenario in {"key", "copy"}
        else Slot(
            "$unbound.module.probe"
            if scenario.startswith("unbound")
            else "Q.pulses.missing",
            fallback=Template("pulse") if scenario.endswith("fallback") else None,
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
    assert cfg.probe.ch == (2 if scenario.endswith("fallback") else 1)
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
    defaults = inline.defaults(env, {"kind": "ref", "name": "Q.pulses.probe"})
    module = defaults["module"]
    assert isinstance(module, dict) and module["freq"] == 0
    state.write({("probe", "kind"): "ref"}, env)
    assert isinstance(state.result(env), FormErrors)


@pytest.mark.parametrize("mode", ["copy", "fallback"])
@pytest.mark.parametrize("expression", ["$unbound.value", "'not a number'", "2 * 3"])
def test_root_slot_expression_overrides_require_input(
    mode: str,
    expression: str,
) -> None:
    form = bound_form(
        Slot("Q.pulses.probe", copy=True, set={"phase": Expr(expression)})
        if mode == "copy"
        else Slot(
            "missing", fallback=Template("pulse"), set={"phase": Expr(expression)}
        )
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
    control = bound_form(
        Slot("Q.pulses.probe", copy=True)
        if mode == "copy"
        else Slot("missing", fallback=Template("pulse"))
    )
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
    """A pulse readout derives ADC frequency/channel from omitted defaults."""

    readout: PulseReadoutCfg


def test_template_nested_lock_and_none_omission_preserve_spec_derivation() -> None:
    catalog = {
        "readout": TemplateDef(
            PulseReadoutSpec,
            "Readout",
            {
                "pulse_cfg.type": "pulse",
                "pulse_cfg.ch": 1,
                "pulse_cfg.nqz": 1,
                "pulse_cfg.freq": 100,
                "pulse_cfg.gain": 0.3,
                "pulse_cfg.waveform.style": "const",
                "pulse_cfg.waveform.length": 0.2,
                "ro_cfg.ro_ch": 0,
                "ro_cfg.ro_length": 0.2,
                "ro_cfg.trig_offset": 0,
            },
        )
    }
    form = Form(
        ReadoutCore, {"readout": Template("readout", lock={"pulse_cfg.freq": 0.0})}
    )
    form.bind_templates(catalog, adapter="synthetic-readout")
    env = environment()
    state = FormState(form.default_input(env))
    value = state.result(env)
    assert not isinstance(value, FormErrors), value
    cfg = form.to_cfg(value, env)
    assert isinstance(cfg, ReadoutCore), cfg
    assert cfg.readout.pulse_cfg.freq == 0
    assert cfg.readout.ro_cfg.gen_ch == 1
    assert cfg.readout.ro_cfg.ro_freq == 0


def readout_expression_form(expression: Expr | None) -> Form[ReadoutCore]:
    """Author nested defaults on existing persistent Spec families only."""
    seeds = {
        "pulse_cfg.type": "pulse",
        "pulse_cfg.ch": 1,
        "pulse_cfg.nqz": 1,
        "pulse_cfg.freq": 100.0,
        "pulse_cfg.gain": 0.3,
        "pulse_cfg.waveform.style": "cosine",
        "pulse_cfg.waveform.length": expression if expression is not None else 0.2,
        "ro_cfg.ro_ch": 0,
        "ro_cfg.ro_length": 0.2,
    }
    if expression is not None:
        seeds["ro_cfg.trig_offset"] = expression
        seeds["pulse_cfg.phase"] = expression
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
        ("readout", "module", "ro_cfg", "trig_offset"): "Field required",
        ("readout", "module", "pulse_cfg", "module", "phase"): "Field required",
        (
            "readout",
            "module",
            "pulse_cfg",
            "module",
            "waveform",
            "module",
            "length",
        ): "Field required",
    }
    state.write(
        {
            ("readout", "module", "ro_cfg", "trig_offset"): 0.4,
            ("readout", "module", "pulse_cfg", "module", "phase"): 0.4,
            (
                "readout",
                "module",
                "pulse_cfg",
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
    assert cfg.readout.ro_cfg.trig_offset == 0.4
    assert cfg.readout.pulse_cfg.phase == 0.4
    assert cfg.readout.pulse_cfg.waveform.length == 0.4


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
    ro = module["ro_cfg"]
    pulse_wrapper = module["pulse_cfg"]
    assert isinstance(pulse_wrapper, dict)
    pulse = pulse_wrapper["module"]
    assert isinstance(ro, dict) and isinstance(pulse, dict)
    waveform_wrapper = pulse["waveform"]
    assert isinstance(waveform_wrapper, dict)
    waveform = waveform_wrapper["module"]
    assert isinstance(waveform, dict)
    expected = 0.0 if expression is None else 0.4
    assert ro["trig_offset"] == pulse["phase"] == expected
    assert waveform["length"] == (0.2 if expression is None else 0.4)
    result = state.result(env)
    if isinstance(result, FormErrors):
        pytest.fail(str(result.messages))
    cfg = form.to_cfg(result, env)
    if isinstance(cfg, FormErrors):
        pytest.fail(str(cfg.messages))
    assert cfg.readout.ro_cfg.trig_offset == cfg.readout.pulse_cfg.phase == expected
    assert cfg.readout.pulse_cfg.waveform.length == waveform["length"]
    # Another form's failed seeds must not affect this form or persistent Specs.
    failed.default_input(env)
    native = PulseReadoutCfg.model_validate(
        {
            "pulse_cfg": {
                "ch": 1,
                "nqz": 1,
                "freq": 100,
                "gain": 0.3,
                "waveform": {"style": "cosine", "length": 0.2},
            },
            "ro_cfg": {"ro_ch": 0, "ro_length": 0.2},
        }
    )
    assert native.ro_cfg.trig_offset == native.pulse_cfg.phase == 0
    assert state.result(env) == result


@pytest.mark.parametrize("phase", [Expr("$unbound.value"), Expr("2 * 3"), None])
def test_nested_slot_fallback_expression_defaults_require_input(
    phase: Expr | None,
) -> None:
    catalog = templates()
    pulse = Template("pulse", set={"phase": phase} if phase is not None else {})
    catalog["readout"] = TemplateDef(
        PulseReadoutSpec,
        "Readout",
        {
            "pulse_cfg": Slot("missing", fallback=pulse),
            "ro_cfg.ro_ch": 0,
            "ro_cfg.ro_length": 0.2,
            "ro_cfg.ro_freq": 100,
        },
    )
    form = Form(ReadoutCore, {"readout": Template("readout")})
    form.bind_templates(catalog, adapter="synthetic")
    env = FormEnv()
    state = FormState(form.default_input(env))
    result = state.result(env)
    if phase == Expr("$unbound.value"):
        assert isinstance(result, FormErrors), result
        assert result.messages == {
            ("readout", "module", "pulse_cfg", "module", "phase"): "Field required"
        }
    else:
        if isinstance(result, FormErrors):
            pytest.fail(str(result.messages))
        cfg = form.to_cfg(result, env)
        if isinstance(cfg, FormErrors):
            pytest.fail(str(cfg.messages))
        assert cfg.readout.pulse_cfg.phase == (0 if phase is None else 6)
    # Removing a fallback still leaves its unresolved required slot absent.
    catalog["readout"] = TemplateDef(
        PulseReadoutSpec,
        "Readout",
        {
            "pulse_cfg": Slot("missing"),
            "ro_cfg.ro_ch": 0,
            "ro_cfg.ro_length": 0.2,
            "ro_cfg.ro_freq": 100,
        },
    )
    control = Form(ReadoutCore, {"readout": Template("readout")})
    control.bind_templates(catalog, adapter="synthetic-control")
    control_state = FormState(control.default_input(env))
    readout = control_state.input.data["readout"]
    assert isinstance(readout, dict)
    module = readout["module"]
    assert isinstance(module, dict) and "pulse_cfg" not in module
    control_result = control_state.result(env)
    assert isinstance(control_result, FormErrors)
    assert control_result.messages == {
        ("readout", "module", "pulse_cfg"): "Field required"
    }


@pytest.mark.parametrize("nested", [False, True])
def test_missing_discriminator_error_names_template_and_nested_field(
    nested: bool,
) -> None:
    template_id = "partial-readout" if nested else "partial-pulse"
    spec = PulseReadoutSpec if nested else PulseSpec
    path = "pulse_cfg.waveform" if nested else "waveform"
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
        from zcu_tools.cfg_model import DirectReadoutCfg

        catalog["pulse"] = TemplateDef(DirectReadoutCfg, "Wrong family", {})
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
        seeds = {f"pulse_cfg.{path}": value for path, value in pulse.items()}
        seeds |= {
            "ro_cfg.ro_ch": 0,
            "ro_cfg.ro_length": 0.2,
            "pulse_cfg.ro_ch": 2,
        }
        definition = TemplateDef(PulseReadoutSpec, "Noisy readout", seeds)
        form = Form(ReadoutCore, {"readout": Template("noisy")})
        path = "pulse_cfg.ro_ch"
    else:
        definition = TemplateDef(PulseSpec, "Noisy pulse", {**pulse, "desc": "note"})
        form = Form(Core, {"probe": Template("noisy")})
        path = "desc"

    with pytest.raises(ValueError, match="noisy") as caught:
        form.bind_templates({"noisy": definition}, adapter="synthetic")

    assert path in str(caught.value), caught.value
