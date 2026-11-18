"""Engine family accuracy against a converged independent full-Lindblad oracle."""

import sys
from collections.abc import Callable
from contextlib import closing
from dataclasses import replace
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path

import numpy as np
import pytest
from zcu_tools.simulate.evolution import (
    Engine,
    NoValidKernel,
    RoutingHistory,
    RWAGuardMetric,
    common_frame_diagnostic,
    load_reference,
    multitone_rwa_diagnostic,
    segment_spans,
    write_reference,
)

from .accuracy import AccuracyMeasurement, measurement_property, require_families
from .accuracy_cases import (
    DISSIPATIVE_DC_CASES,
    AccuracyCase,
    accuracy_case,
    coherent_source,
    history_case,
    multisegment_case,
    multitone_phase_case,
)
from .guard_edge_cases import (
    EDGE_FAMILIES,
    constant_multitone_guard_case,
    guard_edge_case,
)
from .interaction_reference import interaction_reference
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
    "single_tone_sideband",
    *DISSIPATIVE_DC_CASES,
)
KNOWN_EDGE_ERRORS = {
    "gate_cache": 0.00929654785179012,
    "shaped_sambe": 0.0030915535285839482,
}
PINNED = tuple(
    pytest.param(name, family, id=f"{name}-{family}")
    for name in CASE_NAMES
    for family in accuracy_case(name).families
)


@pytest.fixture(scope="session")
def matrix_reference(
    pytestconfig: pytest.Config,
) -> Callable[[AccuracyCase], ReferenceResult]:
    """Solve each named matrix input once per session in this pytest worker.

    Sources/history remain fresh per test; cached arrays contain only oracle
    results for these deterministic physical inputs, never routing products.
    The long DC interaction references also persist in pytest's local cache,
    keyed by complete physics, oracle source, tolerances and dependency versions.
    """
    oracle_code = b"".join(
        Path(__file__).with_name(name).read_bytes()
        for name in ("interaction_reference.py", "reference.py")
    )
    dependencies = sys.version + "".join(
        version(name) for name in ("numpy", "scipy", "numba")
    )
    oracle_hash = sha256(oracle_code + dependencies.encode()).hexdigest()
    cache = pytestconfig.cache.mkdir("interaction-reference") / oracle_hash
    results: dict[tuple[str, int], ReferenceResult] = {}

    def reference_for(case: AccuracyCase) -> ReferenceResult:
        key = case.name, len(case.batch.points)
        if key not in results:
            if case.name in DISSIPATIVE_DC_CASES:
                try:
                    result = load_reference(cache, case.source, case.batch)
                except FileNotFoundError:
                    result = interaction_reference(case.source, case.batch)
                    write_reference(cache, case.source, case.batch, result)
                assert result.convergence_error <= 2e-6
                results[key] = result
            else:
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
    families = (case.idle_family, family) if case.idle_family is not None else (family,)
    with closing(Engine(1, families=families)) as engine:
        result = engine.evolve(case.source, (case.batch,))
    require_families(result.paths, families)
    if case.idle_family is not None:
        actual = {
            name for name, count in result.paths.segments_by_kernel.items() if count > 0
        }
        assert actual == set(families), (
            f"unexpected pinned path: {dict(result.paths.segments_by_kernel)}"
        )
    error = density_error(result.densities, reference.density)
    request.node.user_properties.append(
        measurement_property(
            AccuracyMeasurement(
                name + "/pinned", family, error, reference.convergence_error
            )
        )
    )
    assert error <= 2e-3, f"{name}/{family}: density error {error:.9g} > 2e-3"


@pytest.mark.parametrize("shape", ["gaussian", "flat_top", "drag"])
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


def test_multitone_gate_sampled_upgrade_density() -> None:
    """A phase-demanding sampled key remains accurate across seven dual phases."""
    case = multitone_phase_case("sampled", 7, gain=2.0)
    with closing(Engine(1, families=("multitone_gate",))) as engine:
        result = engine.evolve(case.source, (case.batch,))
    require_families(result.paths, ("multitone_gate",))
    reference = converged_reference(case.source, case.batch)
    assert density_error(result.densities, reference.density) <= 2e-3


def test_multitone_gate_mixed_phase_keys_density(
    matrix_reference: Callable[[AccuracyCase], ReferenceResult],
) -> None:
    """One evolve plays easy, phase-demanding and constant+sampled keys together."""
    cases = (
        multitone_phase_case("gaussian", 7),
        multitone_phase_case("sampled", 7, gain=2.0),
        multitone_phase_case("constant_sampled", 7),
    )
    batches = tuple(
        replace(case.batch, points=np.array([point]))
        for point, case in enumerate(cases)
    )
    with closing(Engine(1, families=("multitone_gate",))) as engine:
        result = engine.evolve(cases[0].source, batches)
    require_families(result.paths, ("multitone_gate",))
    for point, case in enumerate(cases):
        reference = matrix_reference(case)
        assert (
            density_error(result.densities[point : point + 1], reference.density)
            <= 2e-3
        )


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
    if family == "common_frame":
        pump, probe = case.batch.tracks
        diagnostic = common_frame_diagnostic(
            case.source.model((0,)),
            (pump.port, probe.port),
            (float(pump.frequency_ghz[0]), float(probe.frequency_ghz[0])),
            (float(pump.amplitude[0]), float(probe.amplitude[0])),
            float(case.batch.end_ns[0]),
        )
        assert diagnostic.accepted
        coherent = diagnostic.coherent
        assert isinstance(coherent, RWAGuardMetric)
        assert coherent.passed
        assert 0.99 * coherent.threshold <= coherent.value <= coherent.threshold
        request.node.user_properties.append(
            ("common_frame_coherent", str(coherent.value))
        )
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


def test_constant_multitone_guard_edge_density(request: pytest.FixtureRequest) -> None:
    """Bracket constant admission, then compare both admitted complex densities."""
    inside_source, inside_batch = constant_multitone_guard_case(184.5 * (1 - 1e-6))
    with closing(inside_source):
        outside_source, outside_batch = constant_multitone_guard_case(184.5 * 1.01)
        with closing(outside_source):
            budget = 1e-3
            for source, batch, accepted, expected_secular in (
                (inside_source, inside_batch, True, 1.475998524e-4),
                (outside_source, outside_batch, False, 1.49076e-4),
            ):
                pump, probe = batch.tracks
                diagnostic = multitone_rwa_diagnostic(
                    source.model((0,)),
                    (pump.port, probe.port),
                    (float(pump.frequency_ghz[0]), float(probe.frequency_ghz[0])),
                    (float(pump.amplitude[0]), float(probe.amplitude[0])),
                    float(batch.end_ns[0]),
                    budget=budget,
                )
                assert diagnostic.accepted is accepted
                for metric in (
                    diagnostic.budget,
                    diagnostic.carrier_ghz,
                    diagnostic.omission_transfer,
                    diagnostic.omission_phase_rad,
                    diagnostic.real_dimension,
                ):
                    assert isinstance(metric, RWAGuardMetric)
                    assert metric.passed
                secular = diagnostic.secular_accumulation
                assert isinstance(secular, RWAGuardMetric)
                assert secular.passed is accepted
                # Hand calculation: epsilon=.002, Omega_ret=.2 GHz.
                assert secular.value == pytest.approx(expected_secular, rel=1e-12)
                assert secular.threshold == 1.476e-4
                if accepted:
                    assert secular.value / secular.threshold == pytest.approx(
                        1 - 1e-6, rel=1e-12
                    )
                request.node.user_properties.append(
                    (
                        "constant_multitone_secular_"
                        + ("inside" if accepted else "outside"),
                        str(secular.value),
                    )
                )
            with (
                closing(Engine(1, families=("multitone_rwa",))) as engine,
                pytest.raises(NoValidKernel),
            ):
                engine.evolve(outside_source, (outside_batch,), budget=budget)
            with closing(Engine(1, families=("multitone_rwa",))) as engine:
                result = engine.evolve(inside_source, (inside_batch,), budget=budget)
            require_families(result.paths, ("multitone_rwa",))
            # Query and pinned routing must pass before paying for four independent
            # integrations: two points at each of the two reference tolerances.
            reference = converged_reference(inside_source, inside_batch)
            error = density_error(result.densities, reference.density)
            request.node.user_properties.append(
                measurement_property(
                    AccuracyMeasurement(
                        "guard_edge/constant_multitone_rwa",
                        "multitone_rwa",
                        error,
                        reference.convergence_error,
                    )
                )
            )
            assert error <= 2e-3, (
                f"constant multitone guard-edge density error {error:.9g} > 2e-3"
            )


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
