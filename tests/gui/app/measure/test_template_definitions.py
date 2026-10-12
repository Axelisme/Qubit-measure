"""Injected definitions drive the existing library creation catalog."""

import pytest
from zcu_tools import cfg_model
from zcu_tools.experiment.cfg_editing import PROGRAM_SHAPES
from zcu_tools.gui.app.measure.adapter.lowering import schema_to_raw_dict
from zcu_tools.gui.app.measure.specs import MAIN_PROGRAM_SPEC_POLICY
from zcu_tools.gui.app.measure.template_catalog import TemplateCatalog
from zcu_tools.gui.cfg import (
    CfgSchema,
    CfgSectionValue,
    DirectValue,
    EvalValue,
    ReferenceValue,
    make_custom_reference_key,
    make_default_value,
    read_value_path,
    schema_to_raw,
)
from zcu_tools.gui.session.aliases import AliasRow
from zcu_tools.gui.session.context_types import SavedPoint
from zcu_tools.gui.session.types import ParameterContext, SessionEnv
from zcu_tools.resources.context.library import (
    ComponentLibrary,
    LibrarySnapshot,
    ModuleLibraryDocument,
)
from zcu_tools.resources.entry import ComponentSchema
from zcu_tools.resources.entry.schema import PointDocument, PointGeneral


class TemplateComponent(ComponentSchema):
    module: dict[str, str | None]


@pytest.fixture
def environment() -> SessionEnv:
    point = PointDocument.model_construct(
        format="zcu.parameter-container",
        format_version="1.0",
        general=PointGeneral(created_at="2026-10-13T00:00:00Z"),
        components={
            "Q1": TemplateComponent(
                kind="template/test",
                module={
                    "x90": "Q1.pulses.pi2",
                    "unset": None,
                    "dangling": "Q1.pulses.missing",
                },
            )
        },
    )
    library = LibrarySnapshot(
        ModuleLibraryDocument(
            format="zcu.module-library",
            format_version="1.0",
            components={
                "Q1": ComponentLibrary(
                    pulses={
                        "pi2": cfg_model.PulseSpec(
                            ch=2,
                            nqz=1,
                            freq=4200.0,
                            gain=0.25,
                            waveform=cfg_model.ConstWaveformCfg(length=0.04),
                        )
                    }
                )
            },
        )
    )
    parameters = ParameterContext(
        point=point,
        library=library,
        target=SavedPoint(entry_id="synthetic", point="working"),
        roles=None,
    ).bind_aliases((AliasRow("qubit", "Q1"),), {})
    return SessionEnv(parameters=parameters, soc=None, soccfg=None)


def test_nested_slot_keeps_linked_inputs_and_lowers_captured_library_values(
    environment: SessionEnv,
) -> None:
    catalog = TemplateCatalog()
    catalog.register_definitions(
        {
            "reset": cfg_model.TemplateDef(
                cfg_model.BathResetSpec,
                "Synthetic reset",
                {"pi2_cfg": cfg_model.Slot("{qubit}.module.x90")},
            )
        }
    )
    entry = catalog.get("reset")
    first = entry.make_value(environment)
    reference = read_value_path(first.value, "pi2_cfg")
    assert isinstance(reference, ReferenceValue)
    assert reference.chosen_key == "$qubit.module.x90"
    assert read_value_path(reference.value, "gain") == DirectValue(0.25)
    assert read_value_path(reference.value, "freq") == DirectValue(4200.0)
    raw = schema_to_raw_dict(
        CfgSchema(entry.shape(), first.value), environment.parameters
    )
    pulse = raw["pi2_cfg"]
    assert isinstance(pulse, dict)
    assert pulse["waveform"] == {"style": "const", "length": 0.04}
    assert pulse["ch"] == 2
    reference.with_field("gain", 0.9)
    assert read_value_path(
        entry.make_value(environment).value, "pi2_cfg.gain"
    ) == DirectValue(0.25)


@pytest.mark.parametrize("captured", [True, False], ids=["unbound-alias", "no-context"])
def test_nested_slot_without_alias_binding_remains_linked_empty(
    environment: SessionEnv, captured: bool
) -> None:
    assert environment.parameters is not None
    parameters = (
        environment.parameters.bind_aliases((AliasRow("qubit", ""),), {})
        if captured
        else None
    )
    catalog = TemplateCatalog()
    catalog.register_definitions(
        {
            "reset": cfg_model.TemplateDef(
                cfg_model.BathResetSpec,
                "Synthetic reset",
                {"pi2_cfg": cfg_model.Slot("$qubit.module.x90")},
            )
        }
    )
    value = catalog.get("reset").make_value(
        SessionEnv(parameters=parameters, soc=None, soccfg=None)
    )
    assert read_value_path(value.value, "pi2_cfg") == ReferenceValue(
        "$qubit.module.x90", CfgSectionValue()
    )


@pytest.mark.parametrize("slot", ["unset", "dangling"])
def test_bound_missing_slot_uses_nested_structural_defaults(
    environment: SessionEnv, slot: str
) -> None:
    catalog = TemplateCatalog()
    catalog.register_definitions(
        {
            "reset": cfg_model.TemplateDef(
                cfg_model.BathResetSpec,
                "Synthetic reset",
                {"pi2_cfg": cfg_model.Slot(f"$qubit.module.{slot}")},
            )
        }
    )
    entry = catalog.get("reset")
    assert read_value_path(
        entry.make_value(environment).value, "pi2_cfg"
    ) == read_value_path(make_default_value(entry.shape()), "pi2_cfg")


def test_waveform_slot_seed_fails_at_catalog_registration() -> None:
    catalog = TemplateCatalog()
    with pytest.raises(TypeError, match="module"):
        catalog.register_definitions(
            {
                "drive": cfg_model.TemplateDef(
                    cfg_model.PulseSpec,
                    "Drive",
                    {"waveform": cfg_model.Slot("$qubit.waveforms.base")},
                )
            }
        )


def test_slot_and_inline_children_cannot_seed_the_same_reference() -> None:
    catalog = TemplateCatalog()
    with pytest.raises(ValueError, match="children"):
        catalog.register_definitions(
            {
                "reset": cfg_model.TemplateDef(
                    cfg_model.BathResetSpec,
                    "Reset",
                    {
                        "pi2_cfg": cfg_model.Slot("$qubit.module.x90"),
                        "pi2_cfg.gain": 0.2,
                    },
                )
            }
        )


def test_public_reference_resolver_rejects_unsupported_kind_before_source_lookup(
    environment: SessionEnv,
) -> None:
    from zcu_tools.gui.app.measure.reference_defaults import resolve_library_reference

    with pytest.raises(ValueError, match="kind"):
        resolve_library_reference(
            "$unbound.waveforms.base", environment, kind="waveform", fallback=None
        )


def test_public_reference_resolver_rejects_empty_path(environment: SessionEnv) -> None:
    from zcu_tools.gui.app.measure.reference_defaults import resolve_library_reference

    with pytest.raises(ValueError, match="path"):
        resolve_library_reference(" ", environment, kind="module", fallback=None)


def test_public_reference_resolver_returns_fresh_fallback_for_a_missing_root(
    environment: SessionEnv,
) -> None:
    from zcu_tools.gui.app.measure.reference_defaults import resolve_library_reference

    spec = PROGRAM_SHAPES.module("pulse").make_spec(MAIN_PROGRAM_SPEC_POLICY)
    fallback = ReferenceValue(
        make_custom_reference_key("Pulse"), make_default_value(spec)
    )
    result = resolve_library_reference(
        "Q1.pulses.missing", environment, kind="module", fallback=lambda _env: fallback
    )
    assert result == fallback
    assert result is not fallback
    assert (
        resolve_library_reference(
            "Q1.pulses.missing", environment, kind="module", fallback=None
        )
        is None
    )


def test_injected_definitions_provide_module_and_waveform_creation_values() -> None:
    catalog = TemplateCatalog()
    catalog.register_definitions(
        {
            "drive": cfg_model.TemplateDef(
                cfg_model.PulseSpec,
                "Synthetic drive",
                {
                    "freq": cfg_model.Expr("{qubit}.freq"),
                    "gain": 0.2,
                    "waveform.style": "gauss",
                    "waveform.length": 0.1,
                    "waveform.sigma": 0.02,
                },
                default_name="drive_pi",
            ),
            "ramp": cfg_model.TemplateDef(
                cfg_model.CosineWaveformCfg,
                "Synthetic ramp",
                {"length": cfg_model.Expr("{resonator}.length")},
            ),
        }
    )
    assert catalog.list_meta() == [
        {
            "role_id": "drive",
            "label": "Synthetic drive",
            "item_kind": "module",
            "default_name": "drive_pi",
        },
        {
            "role_id": "ramp",
            "label": "Synthetic ramp",
            "item_kind": "waveform",
            "default_name": "",
        },
    ]
    assert [e.template_id for e in catalog.entries_for("waveform")] == ["ramp"]
    env = SessionEnv(parameters=None, soc=None, soccfg=None)
    drive = catalog.get("drive")
    first = drive.make_value(env)
    assert read_value_path(first.value, "freq") == EvalValue("$qubit.freq")
    assert read_value_path(first.value, "gain") == DirectValue(0.2)
    assert read_value_path(first.value, "waveform.style") == DirectValue("gauss")
    assert read_value_path(first.value, "waveform.length") == DirectValue(0.1)
    assert read_value_path(first.value, "waveform.sigma") == DirectValue(0.02)
    first.value.with_field("gain", 0.9)
    second = drive.make_value(env)
    assert read_value_path(second.value, "gain") == DirectValue(0.2)
    ramp = catalog.get("ramp").make_value(env)
    assert read_value_path(ramp.value, "length") == EvalValue("$resonator.length")


@pytest.mark.parametrize("option", ["copy", "set", "lock"])
def test_form_only_slot_options_fail_at_catalog_registration(option: str) -> None:
    slots = {
        "copy": cfg_model.Slot("$qubit.module.x90", copy=True),
        "set": cfg_model.Slot("$qubit.module.x90", set={"gain": 0.2}),
        "lock": cfg_model.Slot("$qubit.module.x90", lock={"ch": 0}),
    }
    catalog = TemplateCatalog()
    with pytest.raises(TypeError, match="Slot\\(path\\)"):
        catalog.register_definitions(
            {
                "reset": cfg_model.TemplateDef(
                    cfg_model.BathResetSpec, "Reset", {"pi2_cfg": slots[option]}
                )
            }
        )
    assert not catalog.has("reset")


def test_slot_seed_requires_a_reference_field() -> None:
    catalog = TemplateCatalog()
    with pytest.raises(TypeError, match="reference"):
        catalog.register_definitions(
            {
                "drive": cfg_model.TemplateDef(
                    cfg_model.PulseSpec,
                    "Drive",
                    {"freq": cfg_model.Slot("$qubit.module.x90")},
                )
            }
        )


def test_empty_seeds_keep_structural_blank_defaults_and_no_name_suggestion() -> None:
    catalog = TemplateCatalog()
    catalog.register_definitions(
        {"pulse:blank": cfg_model.TemplateDef(cfg_model.PulseSpec, "Blank pulse", {})}
    )
    entry = catalog.get("pulse:blank")
    env = SessionEnv(parameters=None, soc=None, soccfg=None)
    expected_spec = PROGRAM_SHAPES.module("pulse").make_spec(MAIN_PROGRAM_SPEC_POLICY)
    assert schema_to_raw(CfgSchema(entry.shape(), entry.make_value(env).value)) == (
        schema_to_raw(CfgSchema(expected_spec, make_default_value(expected_spec)))
    )
    assert entry.default_name == ""


def seeded_fallback_definitions(slot: str) -> dict[str, cfg_model.TemplateDef]:
    """Synthetic forward reference with nonstructural scalar/expression seeds."""
    return {
        "reset": cfg_model.TemplateDef(
            cfg_model.BathResetSpec,
            "Reset",
            {
                "pi2_cfg": cfg_model.Slot(
                    f"{{qubit}}.module.{slot}",
                    fallback=cfg_model.Template("drive"),
                )
            },
        ),
        "drive": cfg_model.TemplateDef(
            cfg_model.PulseSpec,
            "Drive",
            {
                "ch": 3,
                "nqz": 1,
                "freq": cfg_model.Expr("4200 + 23"),
                "gain": 0.37,
                "waveform.length": 0.08,
            },
        ),
    }


def fallback_catalog(slot: str) -> TemplateCatalog:
    catalog = TemplateCatalog()
    try:
        catalog.register_definitions(seeded_fallback_definitions(slot))
    except TypeError as exc:
        pytest.fail(f"Library catalog must accept Slot Template fallback: {exc}")
    return catalog


@pytest.mark.parametrize("slot", ["unset", "dangling"])
def test_bound_missing_slot_expands_seeded_template_inline(
    environment: SessionEnv, slot: str
) -> None:
    entry = fallback_catalog(slot).get("reset")
    first = entry.make_value(environment)
    assert read_value_path(first.value, "pi2_cfg.gain") == DirectValue(0.37)
    assert read_value_path(first.value, "pi2_cfg.freq") == EvalValue("4200 + 23")
    schema = CfgSchema(entry.shape(), first.value)
    wire = schema_to_raw(schema)
    pulse_wire = wire["pi2_cfg"]
    assert isinstance(pulse_wire, dict)
    assert pulse_wire["chosen_key"] == make_custom_reference_key("Pulse")
    children = pulse_wire["value"]
    assert isinstance(children, dict)
    assert children["freq"] == {"__kind": "eval", "expr": "4200 + 23"}
    raw = schema_to_raw_dict(schema, environment.parameters)
    pulse = raw["pi2_cfg"]
    assert isinstance(pulse, dict)
    assert pulse["ch"] == 3
    assert pulse["freq"] == 4223.0
    assert pulse["gain"] == 0.37
    assert pulse["waveform"] == {"style": "const", "length": 0.08}
    first.value.with_field("pi2_cfg.gain", 0.9)
    assert read_value_path(
        entry.make_value(environment).value, "pi2_cfg.gain"
    ) == DirectValue(0.37)


def test_available_slot_keeps_smart_link_despite_template_fallback(
    environment: SessionEnv,
) -> None:
    entry = fallback_catalog("x90").get("reset")
    reference = read_value_path(entry.make_value(environment).value, "pi2_cfg")
    assert isinstance(reference, ReferenceValue)
    assert reference.chosen_key == "$qubit.module.x90"
    assert read_value_path(reference.value, "gain") == DirectValue(0.25)
    assert read_value_path(reference.value, "freq") == DirectValue(4200.0)


@pytest.mark.parametrize("captured", [True, False], ids=["unbound-alias", "no-context"])
def test_unbound_slot_keeps_linked_empty_despite_template_fallback(
    environment: SessionEnv, captured: bool
) -> None:
    assert environment.parameters is not None
    parameters = (
        environment.parameters.bind_aliases((AliasRow("qubit", ""),), {})
        if captured
        else None
    )
    entry = fallback_catalog("unset").get("reset")
    value = entry.make_value(SessionEnv(parameters=parameters, soc=None, soccfg=None))
    assert read_value_path(value.value, "pi2_cfg") == ReferenceValue(
        "$qubit.module.unset", CfgSectionValue()
    )


@pytest.mark.parametrize(
    ("fallback_id", "module", "message"),
    [
        ("unknown", cfg_model.PulseSpec, "Unknown template.*unknown"),
        ("drive", cfg_model.ReadoutSpec, "incompatible.*ReadoutSpec"),
        ("drive", cfg_model.ConstWaveformCfg, "incompatible.*ConstWaveformCfg"),
    ],
    ids=["unknown-id", "wrong-module", "waveform"],
)
def test_invalid_slot_fallback_fails_registration_with_template_field_path(
    fallback_id: str, module: type[cfg_model.ConfigBase], message: str
) -> None:
    definitions = seeded_fallback_definitions("unset")
    definitions["reset"] = cfg_model.TemplateDef(
        cfg_model.BathResetSpec,
        "Reset",
        {
            "pi2_cfg": cfg_model.Slot(
                "$qubit.module.unset", fallback=cfg_model.Template(fallback_id)
            )
        },
    )
    definitions["drive"] = cfg_model.TemplateDef(module, "Drive", {})
    catalog = TemplateCatalog()
    with pytest.raises((ValueError, TypeError), match=f"reset.*pi2_cfg.*{message}"):
        catalog.register_definitions(definitions)
    assert not catalog.has("reset")


@pytest.mark.parametrize("option", ["copy", "set", "lock"])
def test_unsupported_slot_option_reports_template_field_at_registration(
    option: str,
) -> None:
    slots = {
        "copy": cfg_model.Slot("$qubit.module.unset", copy=True),
        "set": cfg_model.Slot("$qubit.module.unset", set={"gain": 0.2}),
        "lock": cfg_model.Slot("$qubit.module.unset", lock={"ch": 0}),
    }
    catalog = TemplateCatalog()
    with pytest.raises(TypeError, match="reset.*pi2_cfg.*Slot"):
        catalog.register_definitions(
            {
                "reset": cfg_model.TemplateDef(
                    cfg_model.BathResetSpec, "Reset", {"pi2_cfg": slots[option]}
                )
            }
        )


def test_public_reference_defaults_expand_template_using_injected_definitions(
    environment: SessionEnv,
) -> None:
    from zcu_tools.gui.app.measure.reference_defaults import resolve_library_reference

    try:
        first = resolve_library_reference(
            "$qubit.module.unset",
            environment,
            kind="module",
            fallback=cfg_model.Template("drive"),
            templates=seeded_fallback_definitions("unset"),
        )
    except TypeError as exc:
        pytest.fail(
            f"Public resolver must accept Template and injected definitions: {exc}"
        )
    assert isinstance(first, ReferenceValue)
    assert first.chosen_key == make_custom_reference_key("Pulse")
    assert read_value_path(first.value, "freq") == EvalValue("4200 + 23")
    assert read_value_path(first.value, "gain") == DirectValue(0.37)
