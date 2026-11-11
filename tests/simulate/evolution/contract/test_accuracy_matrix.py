"""Engine family accuracy against a converged independent full-Lindblad oracle."""

from contextlib import closing
from dataclasses import replace

import numpy as np
import pytest
from zcu_tools.simulate.evolution import Engine, RoutingHistory, segment_spans

from .accuracy import AccuracyMeasurement, measurement_property, require_families
from .accuracy_cases import accuracy_case, history_case
from .reference import converged_reference, density_error

CASE_NAMES = (
    "idle_coherence",
    "weak_plateau",
    "strong_plateau",
    "shaped_complex",
    "two_carriers",
    "continuous_flux",
    "joint_gate_inventory",
)
PINNED = tuple(
    pytest.param(name, family, id=f"{name}-{family}")
    for name in CASE_NAMES
    for family in accuracy_case(name).families
)


@pytest.mark.parametrize("name,family", PINNED)
def test_pinned_family_density(
    name: str, family: str, request: pytest.FixtureRequest
) -> None:
    case = accuracy_case(name)
    reference = converged_reference(case.source, case.batch)
    with closing(Engine(1, families=(family,))) as engine:
        result = engine.evolve(case.source, (case.batch,))
    require_families(result.paths, (family,))
    error = density_error(result.densities, reference.density)
    request.node.user_properties.append(
        measurement_property(
            AccuracyMeasurement(
                name + "/pinned", family, error, reference.convergence_error
            )
        )
    )
    assert error <= 2e-3, f"{name}/{family}: density error {error:.9g} > 2e-3"


@pytest.mark.parametrize("name", CASE_NAMES)
def test_default_empty_and_warm_history_density(
    name: str, request: pytest.FixtureRequest
) -> None:
    case = accuracy_case(name)
    reference = converged_reference(case.source, case.batch)
    with closing(Engine(1)) as engine:
        for state in ("empty", "warm"):
            result = engine.evolve(case.source, (case.batch,))
            assert result.paths.segments_by_kernel
            error = density_error(result.densities, reference.density)
            for family, count in result.paths.segments_by_kernel.items():
                if count:
                    request.node.user_properties.append(
                        measurement_property(
                            AccuracyMeasurement(
                                name + "/" + state,
                                family,
                                error,
                                reference.convergence_error,
                            )
                        )
                    )
            assert error <= 2e-3, f"{name}/{state}: density error {error:.9g} > 2e-3"


def test_multisegment_final_density(request: pytest.FixtureRequest) -> None:
    case = accuracy_case("weak_plateau")
    pulse = replace(
        case.batch.tracks[0],
        start_ns=np.full(2, 0.3),
        duration_ns=np.full(2, 0.9),
        amplitude=np.full(2, 0.05),
    )
    batch = replace(case.batch, tracks=(pulse,), segment_boundaries=((0.7,), (0.7,)))
    assert all(len(spans) >= 4 for spans in segment_spans(batch))
    reference = converged_reference(case.source, batch)
    with closing(Engine(1, families=("t4", "t6"))) as engine:
        result = engine.evolve(case.source, (batch,))
    require_families(result.paths, ("t4", "t6"))
    error = density_error(result.densities, reference.density)
    for family in result.paths.segments_by_kernel:
        request.node.user_properties.append(
            measurement_property(
                AccuracyMeasurement(
                    "multisegment/final", family, error, reference.convergence_error
                )
            )
        )
    assert error <= 2e-3


def test_history_changes_route_with_both_densities_validated(
    request: pytest.FixtureRequest,
) -> None:
    history = RoutingHistory()
    small = history_case(2, history)
    inventory = history_case(8, history)
    with closing(Engine(1)) as engine:
        results = [
            engine.evolve(small.source, (small.batch,)),
            engine.evolve(inventory.source, (inventory.batch,)),
            engine.evolve(small.source, (small.batch,)),
        ]
    require_families(results[0].paths, ("shaped_sambe",))
    require_families(results[2].paths, ("single_tone_gate",))
    for state, case, result in zip(
        ("empty", "inventory", "warm"), (small, inventory, small), results, strict=True
    ):
        reference = converged_reference(case.source, case.batch)
        error = density_error(result.densities, reference.density)
        for family in result.paths.segments_by_kernel:
            request.node.user_properties.append(
                measurement_property(
                    AccuracyMeasurement(
                        "history/" + state, family, error, reference.convergence_error
                    )
                )
            )
        assert error <= 2e-3, f"history/{state}: density error {error:.9g} > 2e-3"
