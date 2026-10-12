"""Template definitions expand bindings without GUI or laboratory knowledge."""

import pytest
from zcu_tools import cfg_model


def test_seeds_expand_default_and_explicit_bindings_without_losing_literals() -> None:
    definition = cfg_model.TemplateDef(
        cfg_model.PulseSpec,
        "Synthetic pulse",
        {
            "freq": cfg_model.Expr("{qubit}.freq + {resonator}.offset"),
            "desc": "Literal {qubit}",
            "gain": 0.2,
            "ch": 2,
        },
    )
    assert definition.expanded_seeds() == {
        "freq": cfg_model.Expr("$qubit.freq + $resonator.offset"),
        "desc": "Literal $qubit",
        "gain": 0.2,
        "ch": 2,
    }
    assert definition.expanded_seeds(qubit="$q2", resonator="$r2")["freq"] == (
        cfg_model.Expr("$q2.freq + $r2.offset")
    )
    assert cfg_model.expand_path("{qubit}.module.readout", qubit="$q2") == (
        "$q2.module.readout"
    )
    assert definition.expanded_seeds()["freq"] == cfg_model.Expr(
        "$qubit.freq + $resonator.offset"
    )


@pytest.mark.parametrize("text", ["{unknown}.freq", "{qubit.missing}", "{qubit"])
def test_unknown_or_malformed_placeholders_fail_at_definition(text: str) -> None:
    with pytest.raises(ValueError, match="placeholder"):
        cfg_model.TemplateDef(cfg_model.PulseSpec, "Broken", {"freq": text})


def test_slot_path_expands_without_resolving_or_rewriting_form_options() -> None:
    fallback = cfg_model.Template("drive", qubit="$other", set={"gain": 0.2})
    slot = cfg_model.Slot(
        "{qubit}.module.x90",
        copy=True,
        set={"freq": cfg_model.Expr("{resonator}.freq")},
        lock={"ch": 2},
        fallback=fallback,
    )
    definition = cfg_model.TemplateDef(
        cfg_model.BathResetSpec,
        "Synthetic reset",
        {"pi2_cfg": slot, "desc": "$literal", "detune": cfg_model.Expr("{qubit}.freq")},
    )
    expanded = definition.expanded_seeds(qubit="$unbound", resonator="$r2")
    assert expanded["pi2_cfg"] == cfg_model.Slot(
        "$unbound.module.x90",
        copy=True,
        set={"freq": cfg_model.Expr("{resonator}.freq")},
        lock={"ch": 2},
        fallback=fallback,
    )
    assert expanded["desc"] == "$literal"
    assert expanded["detune"] == cfg_model.Expr("$unbound.freq")
    assert definition.expanded_seeds()["pi2_cfg"] == cfg_model.Slot(
        "$qubit.module.x90", copy=True, set=slot.set, lock=slot.lock, fallback=fallback
    )


@pytest.mark.parametrize(
    "path", ["{missing}.module.x90", "{qubit.module.x90", "{qubit.attr}"]
)
def test_slot_placeholder_errors_fail_at_definition(path: str) -> None:
    with pytest.raises(ValueError, match="placeholder"):
        cfg_model.TemplateDef(
            cfg_model.BathResetSpec, "Broken slot", {"pi2_cfg": cfg_model.Slot(path)}
        )


def test_path_expansion_rejects_unknown_placeholder() -> None:
    with pytest.raises(ValueError, match="placeholder"):
        cfg_model.expand_path("{missing}.module.readout")


def test_runtime_cfg_is_not_a_persistent_template_type() -> None:
    with pytest.raises(TypeError, match="persistent"):
        cfg_model.TemplateDef(cfg_model.PulseCfg, "Runtime cfg", {})


def test_definition_detaches_seed_mapping_from_its_caller() -> None:
    seeds: dict[str, cfg_model.TemplateSeed] = {"gain": 0.2}
    definition = cfg_model.TemplateDef(cfg_model.PulseSpec, "Pulse", seeds)
    seeds["gain"] = 0.9
    expanded = definition.expanded_seeds()
    expanded["gain"] = 1.0
    assert definition.expanded_seeds() == {"gain": 0.2}
