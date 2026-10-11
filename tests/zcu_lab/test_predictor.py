"""Lab component prefill is a read, not a predictor installation or write."""

from collections.abc import Generator
from pathlib import Path

import pytest
from zcu_tools.resources.entry import ResultEntry, component_registry, snapshot
from zcu_tools.simulate.fluxonium.predict import FluxoniumParams

import zcu_lab.components as components


@pytest.fixture
def entry(tmp_path: Path) -> Generator[ResultEntry]:
    before = component_registry.kinds()
    added = []
    try:
        for kind, model in (
            ("qubit/fluxonium", components.QubitSchema),
            ("resonator", components.ResonatorSchema),
        ):
            if kind not in before:
                component_registry.register(kind, model)
                added.append(kind)
        installed = component_registry.kinds()
        result = ResultEntry.create(
            "prefill",
            result_root=tmp_path / "result",
            database_root=tmp_path / "Database",
        )
        result.setup.add_component(
            "Q2",
            kind="qubit/fluxonium",
            EJ=4.1,
            EC=1.2,
            flux_half=1.75,
            flux_period=3.5,
            ext={"flux_bias": 0.012},
        )
        result.setup.add_component("R1", kind="resonator")
        yield result
        assert component_registry.kinds() == installed
    finally:
        for kind in added:
            component_registry.unregister(kind)
        assert component_registry.kinds() == before


def test_params_for_reads_partial_current_point_without_writing(
    entry: ResultEntry,
) -> None:
    point = entry.new_point("first")
    point.Q2.EJ = 4.3
    before = snapshot(point)
    params = components.params_for(point.Q2)
    assert params == FluxoniumParams(4.3, 1.2, None, 1.75, 3.5, 0.012)
    assert params.missing() == ("EL",)
    assert snapshot(point) == before
    assert components.params_for(entry.setup.Q2).EJ == 4.1


def test_params_for_keeps_unknown_bias_unknown(entry: ResultEntry) -> None:
    point = entry.new_point("first")
    point.Q2.ext = {}
    assert components.params_for(point.Q2).missing() == ("EL", "flux_bias")


def test_params_for_rejects_unsupported_kind(entry: ResultEntry) -> None:
    with pytest.raises(ValueError, match="resonator"):
        components.params_for(entry.setup.R1)


def test_params_for_rejects_invalid_extension_bias(entry: ResultEntry) -> None:
    point = entry.new_point("first")
    point.Q2.ext = {"flux_bias": "not numeric"}
    with pytest.raises(ValueError, match="flux_bias"):
        components.params_for(point.Q2)


def test_transmon_prefill_uses_own_fields_and_preserves_missing_bias() -> None:
    from zcu_tools.resources.entry.registry import ComponentRegistry

    registry = ComponentRegistry()
    components.register_all(registry)
    model = registry.get("qubit/transmon").model_validate(
        {
            "kind": "qubit/transmon",
            "EJmax": 20.0,
            "EC": 0.3,
            "d": 0.0,
            "flux_half": 0.5,
            "flux_period": 1.0,
        }
    )
    before = model.model_dump()
    params = components.params_for(model)
    assert params.to_dict() == {
        "EJmax": 20.0,
        "EC": 0.3,
        "d": 0.0,
        "flux_half": 0.5,
        "flux_period": 1.0,
        "flux_bias": None,
    }
    assert params.missing() == ("flux_bias",)
    assert model.model_dump() == before
