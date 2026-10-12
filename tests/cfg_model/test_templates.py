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
