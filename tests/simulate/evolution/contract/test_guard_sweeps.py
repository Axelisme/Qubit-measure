"""Finite guard-error curves at Engine.evolve, not a new accuracy exception.

Gain curves retain the coarse grid and adjacent local 2e-3 brackets. The gate
curve keeps the same [.8,1,.8] normalized envelope for both baths and every V
and split, including the observed nonmonotone low-Gamma crossings. Exact input
builders are importable without pytest. Existing unity edge cases stay intact.
Known exceedances are marked only after route and two-level oracle validation.
"""

from collections.abc import Callable
from contextlib import closing
from dataclasses import replace

import pytest
from zcu_tools.simulate.evolution import Engine

from .accuracy import AccuracyMeasurement, measurement_property, require_families
from .accuracy_cases import AccuracyCase
from .guard_edge_cases import guard_edge_case
from .guard_sweep_cases import (
    GATE_SAMPLES,
    SCALED_SAMPLES,
    gate_shape_bridge_case,
    gate_split_case,
    scaled_edge_case,
)
from .reference import ReferenceResult, converged_reference, density_error


@pytest.fixture(scope="session")
def sweep_reference() -> Callable[[AccuracyCase], ReferenceResult]:
    """Cache the two-level oracle once per full input name in this worker."""
    results: dict[str, ReferenceResult] = {}

    def reference_for(case: AccuracyCase) -> ReferenceResult:
        if case.name not in results:
            results[case.name] = converged_reference(case.source, case.batch)
        return results[case.name]

    return reference_for


def check_sweep_density(
    case: AccuracyCase,
    known_error: float | None,
    request: pytest.FixtureRequest,
    reference_for: Callable[[AccuracyCase], ReferenceResult],
) -> None:
    """Compare all complex entries; setup/path/oracle failures remain failures."""
    family = case.families[0]
    with closing(case.source), closing(Engine(1, families=(family,))) as engine:
        reference = reference_for(case)
        result = engine.evolve(case.source, (case.batch,), budget=1e-3)
    require_families(result.paths, (family,))
    assert dict(result.paths.segments_by_kernel) == {family: len(case.batch.points)}
    error = density_error(result.densities, reference.density)
    request.node.user_properties.append(
        measurement_property(
            AccuracyMeasurement(case.name, family, error, reference.convergence_error)
        )
    )
    if known_error is not None:
        # Only the final accuracy assertion may be an expected failure.
        request.node.add_marker(
            pytest.mark.xfail(
                strict=True,
                raises=AssertionError,
                reason=f"{family} measured density error {known_error:.12g} > 2e-3",
            )
        )
    assert error <= 2e-3, f"{case.name}: density error {error:.12g} > 2e-3"


@pytest.mark.sim
@pytest.mark.parametrize("family,scale,known_error", SCALED_SAMPLES)
def test_gain_sweep_density(
    family: str,
    scale: float,
    known_error: float | None,
    request: pytest.FixtureRequest,
    sweep_reference: Callable[[AccuracyCase], ReferenceResult],
) -> None:
    check_sweep_density(
        scaled_edge_case(family, scale), known_error, request, sweep_reference
    )


@pytest.mark.sim
@pytest.mark.parametrize("gamma,v,split,known_error", GATE_SAMPLES)
def test_gate_split_sweep_density(
    gamma: float,
    v: float,
    split: float,
    known_error: float | None,
    request: pytest.FixtureRequest,
    sweep_reference: Callable[[AccuracyCase], ReferenceResult],
) -> None:
    check_sweep_density(
        gate_split_case(gamma=gamma, v=v, split=split),
        known_error,
        request,
        sweep_reference,
    )


@pytest.mark.sim
@pytest.mark.parametrize(
    "shaped,known_error",
    [(False, 0.009296552643110703), (True, 0.006949863690347675)],
)
def test_gate_shape_bridge_density(
    shaped: bool,
    known_error: float,
    request: pytest.FixtureRequest,
    sweep_reference: Callable[[AccuracyCase], ReferenceResult],
) -> None:
    check_sweep_density(
        gate_shape_bridge_case(shaped=shaped), known_error, request, sweep_reference
    )


@pytest.mark.sim
@pytest.mark.parametrize(
    "name,family,known_error",
    [
        ("unity_gate_edge", "single_tone_gate", 0.012847428903254271),
        ("long_gate_edge", "shaped_sambe", 0.0030387059893683213),
    ],
)
def test_cost_fallback_density(
    name: str,
    family: str,
    known_error: float | None,
    request: pytest.FixtureRequest,
    sweep_reference: Callable[[AccuracyCase], ReferenceResult],
) -> None:
    """Exclusion can select another inaccurate family; retain those findings."""
    case = (
        guard_edge_case("gate_cache")
        if name == "unity_gate_edge"
        else gate_split_case(gamma=0.01, v=0.018, split=0.00999999)
    )
    # Keep the physical-input name so both pinned families share one oracle.
    case = replace(case, families=(family,))
    check_sweep_density(case, known_error, request, sweep_reference)
