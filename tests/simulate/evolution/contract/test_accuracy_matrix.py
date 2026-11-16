"""Engine family accuracy against a converged independent full-Lindblad oracle."""

from collections.abc import Callable
from contextlib import closing
from dataclasses import replace

import numpy as np
import pytest
from zcu_tools.simulate.evolution import (
    Engine,
    NoValidKernel,
    RoutingHistory,
    segment_spans,
)

from .accuracy import AccuracyMeasurement, measurement_property, require_families
from .accuracy_cases import (
    AccuracyCase,
    accuracy_case,
    coherent_source,
    history_case,
    multisegment_case,
    multitone_phase_case,
)
from .guard_edge_cases import EDGE_FAMILIES, guard_edge_case
from .reference import ReferenceResult, converged_reference, density_error

CASE_NAMES = (
    "idle_coherence",
    "weak_plateau",
    "strong_plateau",
    "shaped_complex",
    "two_carriers",
    "continuous_flux",
    "joint_gate_inventory",
    "joint_constant",
    "low_carrier",
    "linear_shaped",
    "linear_plateau",
    "nearby_carriers",
    "cheap_sampled_nearby",
    "cheap_sampled_separated",
    "adiabatic_ramp",
)
KNOWN_EDGE_ERRORS = {
    "gate_cache": 0.00929654785179012,
    "single_tone_rwa": 0.021802081728112876,
    "common_frame": 0.008066669226955314,
    "shaped_sambe": 0.0030915535285839482,
}
PINNED = tuple(
    pytest.param(name, family, id=f"{name}-{family}")
    for name in CASE_NAMES
    for family in accuracy_case(name).families
)


@pytest.fixture(scope="session")
def matrix_reference() -> Callable[[AccuracyCase], ReferenceResult]:
    """Solve each named matrix input once per session in this pytest worker.

    Sources/history remain fresh per test; cached arrays contain only oracle
    results for these deterministic physical inputs, never routing products.
    """
    results: dict[tuple[str, int], ReferenceResult] = {}

    def reference_for(case: AccuracyCase) -> ReferenceResult:
        key = case.name, len(case.batch.points)
        if key not in results:
            results[key] = converged_reference(case.source, case.batch)
        return results[key]

    return reference_for


@pytest.mark.parametrize("name,family", PINNED)
def test_pinned_family_density(
    name: str,
    family: str,
    request: pytest.FixtureRequest,
    matrix_reference: Callable[[AccuracyCase], ReferenceResult],
) -> None:
    case = accuracy_case(name)
    reference = matrix_reference(case)
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


@pytest.mark.parametrize("shape", ["gaussian", "flat_top"])
@pytest.mark.parametrize("repeats,varying_phases", [(1, True), (7, False), (7, True)])
def test_multitone_gate_dual_phase_density(
    shape: str,
    repeats: int,
    varying_phases: bool,
    request: pytest.FixtureRequest,
    matrix_reference: Callable[[AccuracyCase], ReferenceResult],
) -> None:
    """A checked full-drive key plays once or repeatedly at independent phases."""
    case = multitone_phase_case(shape, repeats, varying_phases=varying_phases)
    reference = matrix_reference(case)
    with closing(Engine(1, families=("multitone_gate",))) as engine:
        result = engine.evolve(case.source, (case.batch,))
    require_families(result.paths, ("multitone_gate",))
    error = density_error(result.densities, reference.density)
    request.node.user_properties.append(
        measurement_property(
            AccuracyMeasurement(
                case.name, "multitone_gate", error, reference.convergence_error
            )
        )
    )
    assert error <= 2e-3


def test_multitone_gate_rejects_unresolved_dual_phase_key() -> None:
    """Strong phase dependence fails validation even without a secular split error."""
    case = multitone_phase_case("gaussian", 1, gain=20)
    model = case.source.model((0,))
    # Remove dissipation so the high-gain refusal is phase resolution, not D/U splitting.
    source = coherent_source(
        replace(model, rates_per_ns=np.zeros((3, 3)), dephasing_per_ns=np.zeros((3, 3)))
    )
    with (
        closing(Engine(1, families=("multitone_gate",))) as engine,
        pytest.raises(NoValidKernel) as rejected,
    ):
        engine.evolve(source, (case.batch,))
    assert rejected.value.domain == "multiple"
    assert rejected.value.attempted == ("multitone_gate",)
    assert rejected.value.points == (0,)


@pytest.mark.parametrize("family", EDGE_FAMILIES)
def test_guard_edge_density(
    family: str,
    request: pytest.FixtureRequest,
    matrix_reference: Callable[[AccuracyCase], ReferenceResult],
) -> None:
    case = guard_edge_case(family)
    reference = matrix_reference(case)
    with closing(Engine(1, families=(family,))) as engine:
        result = engine.evolve(case.source, (case.batch,))
    require_families(result.paths, (family,))
    error = density_error(result.densities, reference.density)
    request.node.user_properties.append(
        measurement_property(
            AccuracyMeasurement(case.name, family, error, reference.convergence_error)
        )
    )
    if family in KNOWN_EDGE_ERRORS:
        # Route and oracle failures must fail independently, not become xfails.
        request.node.add_marker(
            pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason=(
                    f"{family} guard-edge density error "
                    f"{KNOWN_EDGE_ERRORS[family]:.9g} > 2e-3"
                ),
            )
        )
    assert error <= 2e-3, f"{case.name}: density error {error:.9g} > 2e-3"


@pytest.mark.parametrize("name", CASE_NAMES)
def test_default_empty_and_warm_history_density(
    name: str,
    request: pytest.FixtureRequest,
    matrix_reference: Callable[[AccuracyCase], ReferenceResult],
) -> None:
    case = accuracy_case(name)
    with closing(Engine(1)) as engine:
        for state in ("empty", "warm"):
            result = engine.evolve(case.source, (case.batch,))
            assert result.paths.segments_by_kernel
            if name in ("cheap_sampled_nearby", "cheap_sampled_separated"):
                assert result.paths.segments_by_kernel == {
                    case.families[0]: len(case.batch.points)
                }
            reference = matrix_reference(case)
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


def test_multisegment_final_density(
    request: pytest.FixtureRequest,
    matrix_reference: Callable[[AccuracyCase], ReferenceResult],
) -> None:
    case = replace(multisegment_case(), name="multisegment")
    batch = case.batch
    assert all(len(spans) >= 4 for spans in segment_spans(batch))
    reference = matrix_reference(case)
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
    matrix_reference: Callable[[AccuracyCase], ReferenceResult],
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
        reference = matrix_reference(case)
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
