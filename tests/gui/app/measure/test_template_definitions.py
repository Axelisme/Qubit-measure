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


@pytest.mark.parametrize("option", ["copy", "set", "lock", "fallback"])
def test_form_only_slot_options_fail_at_catalog_registration(option: str) -> None:
    slots = {
        "copy": cfg_model.Slot("$qubit.module.x90", copy=True),
        "set": cfg_model.Slot("$qubit.module.x90", set={"gain": 0.2}),
        "lock": cfg_model.Slot("$qubit.module.x90", lock={"ch": 0}),
        "fallback": cfg_model.Slot(
            "$qubit.module.x90", fallback=cfg_model.Template("drive")
        ),
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
