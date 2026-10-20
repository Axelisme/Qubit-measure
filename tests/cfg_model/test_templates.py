"""Template definitions expand bindings without GUI or laboratory knowledge."""

import pytest
from zcu_tools import cfg_model


def test_seeds_expand_default_and_explicit_bindings_without_losing_literals() -> None:
    registry = cfg_model.PlaceholderRegistry()
    assert registry.current() == {}
    definition = cfg_model.TemplateDef(
        cfg_model.PulseSpec,
        "Synthetic pulse",
        {
            "freq": cfg_model.Expr("{drive}.freq + {sensor}.offset"),
            "desc": "Literal {drive}",
            "gain": 0.2,
            "ch": 2,
        },
    )
    with pytest.raises(ValueError, match="drive"):
        definition.expanded_seeds(defaults=registry.current())
    registry.register("drive", "$source")
    registry.register("sensor", "$detector")
    assert definition.expanded_seeds(defaults=registry.current()) == {
        "freq": cfg_model.Expr("$source.freq + $detector.offset"),
        "desc": "Literal $source",
        "gain": 0.2,
        "ch": 2,
    }
    assert definition.expanded_seeds(
        defaults=registry.current(), bindings={"drive": "$other"}
    )["freq"] == cfg_model.Expr("$other.freq + $detector.offset")
    assert (
        cfg_model.expand_path(
            "{unregistered}.module.readout", bindings={"unregistered": "S2"}
        )
        == "S2.module.readout"
    )
    assert registry.current() == {"drive": "$source", "sensor": "$detector"}
    assert cfg_model.PlaceholderRegistry().current() == {}


@pytest.mark.parametrize("text", ["{drive.missing}", "{drive", "{}", "{drive[0]}"])
def test_unknown_or_malformed_placeholders_fail_at_definition(text: str) -> None:
    with pytest.raises(ValueError, match="placeholder"):
        cfg_model.TemplateDef(cfg_model.PulseSpec, "Broken", {"freq": text})


def test_slot_path_expands_without_resolving_or_rewriting_form_options() -> None:
    slot = cfg_model.Slot(
        "{qubit}.module.x90",
        copy=True,
        set={"freq": cfg_model.Expr("{resonator}.freq"), "ch": 2},
    )
    definition = cfg_model.TemplateDef(
        cfg_model.BathResetSpec,
        "Synthetic reset",
        {"pi2_cfg": slot, "desc": "$literal", "detune": cfg_model.Expr("{qubit}.freq")},
    )
    expanded = definition.expanded_seeds(
        bindings={"qubit": "$unbound", "resonator": "$r2"}
    )
    assert expanded["pi2_cfg"] == cfg_model.Slot(
        "$unbound.module.x90",
        copy=True,
        set={"freq": cfg_model.Expr("{resonator}.freq"), "ch": 2},
    )
    assert expanded["desc"] == "$literal"
    assert expanded["detune"] == cfg_model.Expr("$unbound.freq")
    assert definition.expanded_seeds(bindings={"qubit": "$qubit"})[
        "pi2_cfg"
    ] == cfg_model.Slot("$qubit.module.x90", copy=True, set=slot.set)


@pytest.mark.parametrize("path", ["{drive.module.x90", "{drive.attr}"])
def test_slot_placeholder_errors_fail_at_definition(path: str) -> None:
    with pytest.raises(ValueError, match="placeholder"):
        cfg_model.TemplateDef(
            cfg_model.BathResetSpec, "Broken slot", {"pi2_cfg": cfg_model.Slot(path)}
        )


def test_path_expansion_requires_binding_even_for_former_primary_names() -> None:
    for name in ("missing", "qubit", "resonator"):
        with pytest.raises(ValueError, match=name):
            cfg_model.expand_path("{" + name + "}.module.readout")


def test_registry_snapshots_and_template_bindings_are_detached() -> None:
    registry = cfg_model.PlaceholderRegistry()
    registry.register("drive", "$source")
    snapshot = registry.current()
    registry.register("drive", "$later")
    assert snapshot == {"drive": "$source"}
    bindings = {"drive": "$explicit"}
    template = cfg_model.Template("synthetic", bindings=bindings)
    bindings["drive"] = "$mutated"
    assert template.bindings == {"drive": "$explicit"}
    assert (
        cfg_model.expand_path("{{literal}} {drive}", defaults=snapshot)
        == "{literal} $source"
    )


def test_runtime_cfg_is_not_a_persistent_template_type() -> None:
    with pytest.raises(TypeError, match="persistent"):
        cfg_model.TemplateDef(cfg_model.PulseCfg, "Runtime cfg", {})


def test_template_rejects_subclasses_of_persistent_specs() -> None:
    class CustomPulse(cfg_model.PulseSpec):
        pass

    class CustomWaveform(cfg_model.ConstWaveformCfg):
        pass

    for spec_type in (CustomPulse, CustomWaveform):
        with pytest.raises(TypeError, match="persistent"):
            cfg_model.TemplateDef(spec_type, "Custom", {})


def test_definition_detaches_seed_mapping_from_its_caller() -> None:
    seeds: dict[str, cfg_model.TemplateSeed] = {"gain": 0.2}
    definition = cfg_model.TemplateDef(cfg_model.PulseSpec, "Pulse", seeds)
    seeds["gain"] = 0.9
    expanded = definition.expanded_seeds()
    expanded["gain"] = 1.0
    assert definition.expanded_seeds() == {"gain": 0.2}


def test_readout_template_accepts_concrete_persistent_spec() -> None:
    definition = cfg_model.TemplateDef(
        cfg_model.PulseReadoutSpec,
        "Synthetic readout",
        {"freq": cfg_model.Expr("{sensor}.freq"), "gen_ch": 2, "ro_ch": 3},
    )

    assert definition.module is cfg_model.PulseReadoutSpec
    assert definition.expanded_seeds(bindings={"sensor": "$sensor"}) == {
        "freq": cfg_model.Expr("$sensor.freq"),
        "gen_ch": 2,
        "ro_ch": 3,
    }
