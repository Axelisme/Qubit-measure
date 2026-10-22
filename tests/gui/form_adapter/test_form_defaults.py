"""Templates and slots resolve captured sources into editable persistent inputs."""

import pytest
from pydantic import JsonValue
from zcu_tools.cfg_model import (
    AliasName,
    BathResetSpec,
    ConfigBase,
    ConstWaveformCfg,
    Expr,
    FieldMeta,
    FlatTopWaveformSpec,
    Hidden,
    PulseReadoutSpec,
    PulseSpec,
    Slot,
    Template,
    TemplateDef,
    WaveformSpec,
)
from zcu_tools.device import BaseDeviceInfo
from zcu_tools.gui.form import (
    FormEnv,
    FormErrors,
    FormModel,
    FormState,
)
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
    state.select(("probe",), {"kind": "ref", "name": "Q.pulses.probe"}, env)
    state.run_command(("probe",), "to_inline", env)
    value = state.result(env)
    assert not isinstance(value, FormErrors), value
    copied = form.to_cfg(value, env)
    assert isinstance(copied, Core) and copied.probe.gain == 0.3


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


class EnvelopeNamedFields(ConfigBase):
    """Ordinary fields can share the expression envelope's key."""

    expr: str
    gain: float


class EnvelopeSeeds(ConfigBase):
    """Real scalar envelopes remain single diagnostic leaves."""

    amount: float
    phase: complex


def test_seed_diagnostics_distinguish_model_fields_from_scalar_envelopes() -> None:
    env = FormEnv()
    form = Form(EnvelopeNamedFields, {"expr": "label", "gain": 0.1})
    input = form.default_input(env)
    rows = {row.path: row for row in form.diagnostics(input, env, None)}
    assert set(rows) == {("expr",), ("gain",)}
    assert rows[("expr",)].seed_source is None
    assert rows[("expr",)].seed_status == "literal"

    scalar = Form(EnvelopeSeeds, {"amount": Expr("2 * 3"), "phase": 1 + 2j})
    scalar_input = scalar.default_input(env)
    scalar_rows = {row.path: row for row in scalar.diagnostics(scalar_input, env, None)}
    assert set(scalar_rows) == {("amount",), ("phase",)}
    assert scalar_rows[("amount",)].seed_source == "2 * 3"
    assert scalar_rows[("amount",)].seed_status == "available"
    assert scalar_rows[("phase",)].seed_source is None
    assert scalar_rows[("phase",)].seed_status == "literal"


def test_seed_metadata_returns_new_form_and_later_call_replaces_metadata() -> None:
    original = bound_form(Template("pulse"))
    assert callable(getattr(original, "with_seed_metadata", None)), (
        "Forms need a metadata copy operation"
    )
    env = environment()
    original_input = original.default_input(env)
    first = original.with_seed_metadata(
        {
            ("probe", "module", "gain"): FieldMeta(calibration="uncalibrated"),
            ("probe", "module", "freq"): FieldMeta(calibration="uncalibrated"),
        }
    )
    second = first.with_seed_metadata(
        {
            ("probe", "module", "gain"): FieldMeta(calibration="measured"),
        }
    )
    assert first is not original and second is not first
    original_rows = {
        row.path: row for row in original.diagnostics(original_input, env, None)
    }
    first_rows = {row.path: row for row in first.diagnostics(original_input, env, None)}
    second_rows = {
        row.path: row for row in second.diagnostics(original_input, env, None)
    }
    assert original_rows[("probe", "module", "gain")].uncalibrated is None
    assert first_rows[("probe", "module", "gain")].uncalibrated is True
    assert first_rows[("probe", "module", "freq")].uncalibrated is True
    assert second_rows[("probe", "module", "gain")].uncalibrated is False
    assert second_rows[("probe", "module", "freq")].uncalibrated is None
    assert original.default_input(env).data == original_input.data


@pytest.mark.parametrize("copy", [False, True])
def test_seed_diagnostics_preserve_slot_origin_without_inventing_calibration(
    copy: bool,
) -> None:
    form = bound_form(Slot("Q.pulses.probe", copy=copy))
    assert callable(getattr(form, "diagnostics", None)), "Forms need source diagnostics"
    env = environment()
    input = form.default_input(env)
    rows = {row.path: row for row in form.diagnostics(input, env, None)}
    row = rows[("probe", "module", "gain") if copy else ("probe",)]
    assert row.seed_source == "Q.pulses.probe"
    assert row.seed_status == "available"
    assert row.uncalibrated is None
    assert input.data == form.default_input(env).data


class OptionalCore(ConfigBase):
    """Core nullability permits a missing slot to remain intentionally disabled."""

    probe: PulseCfg | None = None


class NullableRequiredCore(ConfigBase):
    """Allowing None does not declare a disabled initial selection."""

    probe: PulseCfg | None


@pytest.mark.parametrize("cfg_type", [Core, NullableRequiredCore, OptionalCore])
def test_unbound_slot_obeys_core_required_or_optional_default(
    cfg_type: type[ConfigBase],
) -> None:
    optional = cfg_type is OptionalCore
    form = Form(cfg_type, {"probe": Slot("$unbound.module.probe")})
    form.bind_templates(templates(), adapter="synthetic-optional")
    env = environment()
    state = FormState(form.default_input(env))
    assert state.input.data == ({"probe": None} if optional else {})
    value = state.result(env)
    if not optional:
        assert isinstance(value, FormErrors)
        assert ("probe",) in value.messages
        return
    assert not isinstance(value, FormErrors), value
    assert form.to_cfg(value, env) == OptionalCore(probe=None)


@pytest.mark.parametrize("missing_slot", [False, True], ids=["absent", "missing-slot"])
@pytest.mark.parametrize("nested", [False, True], ids=["root", "nested"])
def test_default_none_module_is_explicit_and_selectable(
    missing_slot: bool, nested: bool
) -> None:
    class NestedCore(ConfigBase):
        modules: OptionalCore

    tree = {"probe": Slot("Q.pulses.missing")} if missing_slot else {}
    form = Form(NestedCore, {"modules": tree}) if nested else Form(OptionalCore, tree)
    env = environment()
    state = FormState(form.default_input(env))
    path = ("modules", "probe") if nested else ("probe",)
    assert state.input.data == (
        {"modules": {"probe": None}} if nested else {"probe": None}
    )
    target = state.describe(path, env).target
    assert target is not None and target.options is not None
    assert target.selected is not None
    assert target.options[target.selected].label == "None"
    assert not target.can_expand
    value = state.result(env)
    if isinstance(value, FormErrors):
        pytest.fail(str(value.messages))
    cfg = form.to_cfg(value, env)
    assert cfg == (NestedCore(modules=OptionalCore()) if nested else OptionalCore())

    # Disabling is only an initial choice; a captured module remains selectable.
    state.write({path: {"kind": "ref", "name": "Q.pulses.probe"}}, env)
    value = state.result(env)
    if isinstance(value, FormErrors):
        pytest.fail(str(value.messages))
    cfg = form.to_cfg(value, env)
    if isinstance(cfg, FormErrors):
        pytest.fail(str(cfg.messages))
    probe = cfg.modules.probe if isinstance(cfg, NestedCore) else cfg.probe
    assert probe is not None and probe.gain == 0.3


@pytest.mark.parametrize(
    "node",
    [
        Slot("Q.pulses.probe"),
        Slot("Q.pulses.probe", copy=True),
        Template("pulse"),
    ],
    ids=["key", "copy", "template"],
)
def test_optional_module_keeps_available_initial_selection(
    node: Slot | Template,
) -> None:
    form = Form(OptionalCore, {"probe": node})
    form.bind_templates(templates(), adapter="synthetic")
    env = environment()
    state = FormState(form.default_input(env))
    probe_input = state.input.data["probe"]
    assert isinstance(probe_input, dict)
    assert probe_input["kind"] == (
        "ref" if isinstance(node, Slot) and not node.copy else "inline"
    )
    value = state.result(env)
    if isinstance(value, FormErrors):
        pytest.fail(str(value.messages))
    cfg = form.to_cfg(value, env)
    assert isinstance(cfg, OptionalCore) and cfg.probe is not None
    assert cfg.probe.freq == (120 if isinstance(node, Template) else 100)


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

    for name, default in (("device", ""), ("nullable_device", "standby")):
        field = state.describe((name,), env).target
        assert field is not None and field.options is not None
        values = [option.value for option in field.options]
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


def nested_copy_env(*, inline: bool = False) -> FormEnv:
    """Capture a reset with reusable pulse and waveform children, without I/O."""
    pulse = PulseSpec.model_validate(
        {
            "ch": 1,
            "nqz": 1,
            "freq": 100,
            "gain": 0.5,
            "waveform": (
                {"style": "const", "length": 0.04} if inline else "Q.waveforms.envelope"
            ),
        }
    )
    return FormEnv(
        library=LibrarySnapshot(
            ModuleLibraryDocument(
                format="zcu.module-library",
                format_version="2.0",
                components={
                    "Q": ComponentLibrary(
                        pulses={"probe": pulse},
                        waveforms={"envelope": ConstWaveformCfg(length=0.04)},
                        reset={
                            "bath": BathResetSpec(
                                cavity_tone_cfg="Q.pulses.probe",
                                qubit_tone_cfg="Q.pulses.probe",
                                pi2_cfg="Q.pulses.probe",
                            )
                        },
                    )
                },
            )
        )
    )


@pytest.mark.parametrize("placement", ["template", "slot-copy"])
def test_hidden_module_seeds_supply_literals_at_dotted_spec_paths(
    placement: str,
) -> None:
    env = nested_copy_env()
    overrides = {"freq": Hidden(0.0), "waveform.length": Hidden(0.5)}
    node = (
        Template("pulse", set=overrides)
        if placement == "template"
        else Slot("Q.pulses.probe", copy=True, set=overrides)
    )
    form = bound_form(node)
    state = FormState(form.default_input(env))
    for path in (
        ("probe", "module", "freq"),
        ("probe", "module", "waveform", "module", "length"),
    ):
        description = state.describe(path, env).target
        assert description is not None and description.hidden
    gain = state.describe(("probe", "module", "gain"), env).target
    assert gain is not None and not gain.hidden and not gain.readonly
    cfg = configured(form, state, env)
    assert cfg.probe.freq == 0.0
    assert cfg.probe.waveform.length == 0.5
    state.write({("probe", "module", "gain"): 0.2}, env)
    assert configured(form, state, env).probe.gain == 0.2


@pytest.mark.parametrize("placement", ["template", "slot-copy"])
def test_hidden_module_fill_does_not_bypass_spec_validation(placement: str) -> None:
    env = environment()
    node = (
        Template("pulse", set={"freq": Hidden("not numeric")})
        if placement == "template"
        else Slot("Q.pulses.probe", copy=True, set={"freq": Hidden("not numeric")})
    )
    form = bound_form(node)
    result = FormState(form.default_input(env)).result(env)
    assert isinstance(result, FormErrors)
    assert ("probe", "module", "freq") in result.messages


def input_object(value: JsonValue) -> dict[str, JsonValue]:
    """Require an object at the public Form input boundary."""
    assert isinstance(value, dict), value
    return value


def configured[C: ConfigBase](
    form: Form[C], state: FormState[FormModel], env: FormEnv
) -> C:
    """Require a valid result and convert it through the public cfg boundary."""
    result = state.result(env)
    if isinstance(result, FormErrors):
        pytest.fail(str(result.messages))
    cfg = form.to_cfg(result, env)
    if isinstance(cfg, FormErrors):
        pytest.fail(str(cfg.messages))
    return cfg


@pytest.mark.parametrize("inline", [False, True], ids=["captured-key", "inline-child"])
def test_slot_copy_overrides_child_without_changing_captured_source(
    inline: bool,
) -> None:
    env = nested_copy_env(inline=inline)
    assert env.library is not None
    original = env.library.document()
    form = bound_form(
        Slot("Q.pulses.probe", copy=True, set={"waveform.length": 0.03, "gain": 0.7})
    )
    state = FormState(form.default_input(env))
    cfg = configured(form, state, env)
    assert cfg.probe.waveform.length == 0.03
    assert cfg.probe.gain == 0.7
    assert env.library.document() == original
    assert env.library.waveform_spec("Q.waveforms.envelope").length == 0.04

    # An installed copy remains detached even when a host supplies a later capture.
    later = FormEnv(
        library=env.library.replaced_waveform(
            "Q.waveforms.envelope", ConstWaveformCfg(length=0.9)
        )
    )
    assert configured(form, state, later).probe.waveform.length == 0.03
    state.write({("probe", "module", "waveform", "module", "length"): 0.06}, later)
    assert configured(form, state, later).probe.waveform.length == 0.06
    assert env.library.document() == original


@pytest.mark.parametrize("placement", ["root-slot", "template-slot"])
def test_slot_copy_expands_only_the_overridden_reference_path(placement: str) -> None:
    env = nested_copy_env()
    assert env.library is not None
    original = env.library.document()
    if placement == "root-slot":
        form = Form(
            ResetCore,
            {
                "reset": Slot(
                    "Q.reset.bath", copy=True, set={"pi2_cfg.waveform.length": 0.03}
                )
            },
        )
        catalog: dict[str, TemplateDef] = {}
    else:
        form = Form(ResetCore, {"reset": Template("reset")})
        catalog = {
            "reset": TemplateDef(
                BathResetSpec,
                "Reset",
                {
                    "cavity_tone_cfg": Slot("Q.pulses.probe"),
                    "qubit_tone_cfg": Slot("Q.pulses.probe"),
                    "pi2_cfg": Slot(
                        "Q.pulses.probe", copy=True, set={"waveform.length": 0.03}
                    ),
                },
            )
        }
    form.bind_templates(catalog, adapter="synthetic-copy")
    state = FormState(form.default_input(env))
    reset = input_object(input_object(state.input.data["reset"])["module"])
    assert reset["cavity_tone_cfg"] == {"kind": "ref", "name": "Q.pulses.probe"}
    assert reset["qubit_tone_cfg"] == {"kind": "ref", "name": "Q.pulses.probe"}
    pi2 = input_object(input_object(reset["pi2_cfg"])["module"])
    assert pi2["waveform"] == {
        "kind": "inline",
        "module": {"style": "const", "length": 0.03},
    }
    cfg = configured(form, state, env)
    assert cfg.reset.pi2_cfg.waveform.length == 0.03
    assert cfg.reset.cavity_tone_cfg.waveform.length == 0.04
    assert cfg.reset.qubit_tone_cfg.waveform.length == 0.04
    assert env.library.document() == original


@pytest.mark.parametrize("inline", [False, True], ids=["captured-key", "inline-child"])
def test_slot_copy_none_child_override_requires_manual_value(inline: bool) -> None:
    env = nested_copy_env(inline=inline)
    form = bound_form(Slot("Q.pulses.probe", copy=True, set={"waveform.length": None}))
    state = FormState(form.default_input(env))
    path = ("probe", "module", "waveform", "module", "length")
    result = state.result(env)
    assert isinstance(result, FormErrors)
    assert path in result.messages
    assert state.describe(path, env).target is not None
    state.write({path: 0.08}, env)
    assert configured(form, state, env).probe.waveform.length == 0.08


class FailedChildLibrary(LibrarySnapshot):
    """A captured source whose public waveform query fails at the boundary."""

    def waveform_spec(self, path: str) -> WaveformSpec:
        """Report the source failure instead of providing the requested Spec."""
        raise OSError(f"Captured waveform unavailable: {path}")


@pytest.mark.parametrize(
    ("fault", "error", "message"),
    [
        ("missing", KeyError, "absent"),
        (
            "wrong-namespace",
            ValueError,
            "Library kind 'waveform' does not match partition",
        ),
        ("malformed-self-reference", ValueError, "[Ii]ncompatible.*family"),
        ("source-failure", OSError, "Captured waveform unavailable"),
    ],
)
def test_slot_copy_child_source_errors_propagate(
    fault: str, error: type[Exception], message: str
) -> None:
    env = nested_copy_env()
    assert env.library is not None
    document = env.library.document()
    pulse = document.components["Q"].pulses["probe"]
    path = "waveform.length"
    if fault == "missing":
        pulse.waveform = "Q.waveforms.absent"
    elif fault == "wrong-namespace":
        pulse.waveform = "Q.pulses.probe"
    elif fault == "malformed-self-reference":
        # Valid persistent syntax, but flat_top is not an admissible ramp family.
        document.components["Q"].waveforms["envelope"] = FlatTopWaveformSpec(
            length=0.4, raise_waveform="Q.waveforms.envelope"
        )
        path = "waveform.raise_waveform.length"
    library = (
        FailedChildLibrary(document)
        if fault == "source-failure"
        else LibrarySnapshot(document)
    )
    original = library.document()
    form = bound_form(Slot("Q.pulses.probe", copy=True, set={path: 0.03}))
    with pytest.raises(error, match=message):
        form.default_input(FormEnv(library=library))
    assert library.document() == original
