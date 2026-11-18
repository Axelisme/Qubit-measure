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
    ModelSource,
    NoValidKernel,
    RoutingHistory,
    RWAGuardMetric,
    TimelineBatch,
    common_frame_diagnostic,
    load_reference,
    multitone_rwa_diagnostic,
    segment_spans,
    write_reference,
)

from .accuracy import AccuracyMeasurement, measurement_property, require_families
from .accuracy_cases import (
    CPU_DP54_CASES,
    DISSIPATIVE_DC_CASES,
    OPENCL_CASES,
    AccuracyCase,
    accuracy_case,
    cpu_dp54_case,
    history_case,
    multisegment_case,
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
    "sampled_multitone_phase_sequence",
    "sampled_multitone_short_sweep_32",
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
    Long DC and cheap sampled references also persist in pytest's local cache,
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
    interaction_cache = pytestconfig.cache.mkdir("interaction-reference") / oracle_hash
    sampled_cache = pytestconfig.cache.mkdir("sampled-reference") / oracle_hash
    results: dict[tuple[str, int], ReferenceResult] = {}

    def reference_for(case: AccuracyCase) -> ReferenceResult:
        key = case.name, len(case.batch.points)
        if key not in results:
            if case.name in DISSIPATIVE_DC_CASES:
                results[key] = _persistent_reference(
                    case, interaction_cache, interaction_reference
                )
            elif case.name in ("cheap_sampled_nearby", "cheap_sampled_separated"):
                results[key] = _persistent_reference(
                    case, sampled_cache, converged_reference
                )
            elif case.name == "sampled_multitone_short_sweep_32":
                results[key] = interaction_reference(case.source, case.batch)
                assert results[key].convergence_error <= 2e-6
            else:
                results[key] = converged_reference(case.source, case.batch)
        return results[key]

    return reference_for


def _persistent_reference(
    case: AccuracyCase,
    cache: Path,
    solve: Callable[[ModelSource, TimelineBatch], ReferenceResult],
) -> ReferenceResult:
    try:
        result = load_reference(cache, case.source, case.batch)
    except FileNotFoundError:
        result = solve(case.source, case.batch)
        assert result.convergence_error <= 2e-6
        write_reference(cache, case.source, case.batch, result)
    assert result.convergence_error <= 2e-6
    return result


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
    with closing(Engine(1, families=families, backend="cpu")) as engine:
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


@pytest.mark.parametrize("name", CPU_DP54_CASES)
def test_cpu_dp54_density(name: str, request: pytest.FixtureRequest) -> None:
    case = cpu_dp54_case(name)
    families = (case.idle_family, *case.families) if case.idle_family else case.families
    with (
        closing(case.source),
        closing(Engine(1, backend="cpu", families=families)) as engine,
    ):
        result = engine.evolve(case.source, (case.batch,), telemetry=True)
        require_families(result.paths, families)
        report = result.paths.routing["dop853"]
        assert report.dp54_rows == len(case.batch.points)
        assert report.dop853_rows == 0
        reference = interaction_reference(
            case.source, case.batch, max_step_ns=case.interaction_max_step_ns
        )
    error = density_error(result.densities, reference.density)
    request.node.user_properties.append(
        measurement_property(
            AccuracyMeasurement(name, "dop853", error, reference.convergence_error)
        )
    )
    assert reference.convergence_error <= 2e-6
    assert error <= 2e-3, f"{name}: density error {error:.9g} > 2e-3"


@pytest.mark.requires_opencl
@pytest.mark.parametrize(
    ("name", "budget", "dp54"),
    [
        *(pytest.param(name, 1e-3, True, id=name) for name in OPENCL_CASES),
        pytest.param("cpu_dp54_two_tone_960", 1e-3, True, id="synthetic-two-tone-960"),
        pytest.param("cpu_dp54_duration_edge", 1e-3, True, id="duration_edge"),
        pytest.param("sampled_gaussian_9", 0.999e-3, False, id="tight-budget"),
    ],
)
def test_opencl_dop853_density(
    name: str,
    budget: float,
    dp54: bool,
    request: pytest.FixtureRequest,
) -> None:
    case = cpu_dp54_case(name) if name in CPU_DP54_CASES else accuracy_case(name)
    families = (case.idle_family, *case.families) if case.idle_family else case.families
    with closing(case.source):
        with closing(
            Engine(1, backend="opencl", families=families, chunks_per_worker=1)
        ) as engine:
            result = engine.evolve(
                case.source, (case.batch,), budget=budget, telemetry=True
            )
        require_families(result.paths, families)
        report = result.paths.routing["dop853"]
        assert report.opencl_rows == len(case.batch.points)
        assert report.dp54_rows == (report.opencl_rows if dp54 else 0)
        assert report.dop853_rows == (0 if dp54 else report.opencl_rows)
        assert result.paths.max_trace_error <= 1e-8
        densities = result.densities
        assert np.max(np.abs(densities - densities.conj().swapaxes(-1, -2))) <= 1e-10

        reference = interaction_reference(
            case.source, case.batch, max_step_ns=case.interaction_max_step_ns
        )
        assert reference.convergence_error <= 2e-6
        with closing(Engine(1, backend="cpu", families=families)) as engine:
            cpu = engine.evolve(case.source, (case.batch,), budget=budget)
    error = density_error(densities, reference.density)
    method = "dp54" if dp54 else "dop853"
    request.node.user_properties.append(
        measurement_property(
            AccuracyMeasurement(
                f"{name}/{method}", "dop853/opencl", error, reference.convergence_error
            )
        )
    )
    assert error <= 2e-3
    assert density_error(densities, cpu.densities) < 1e-3


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
    with closing(Engine(1, backend="cpu")) as engine:
        for state in ("empty", "warm"):
            result = engine.evolve(case.source, (case.batch,))
            assert result.paths.segments_by_kernel
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


@pytest.mark.parametrize(
    "name, expected_spots, copies",
    [("cheap_sampled_separated", 2, 2), ("sampled_multitone_phase_sequence", 6, 1)],
)
def test_sampled_multitone_rejects_each_failed_group(
    name: str,
    expected_spots: int,
    copies: int,
    request: pytest.FixtureRequest,
    matrix_reference: Callable[[AccuracyCase], ReferenceResult],
) -> None:
    case = accuracy_case(name)
    # Keep the piecewise-linear drive but add knot restarts to direct playback.
    # These cases must pay for and fail the probe, not fall back on price alone.
    tracks = []
    for track in case.batch.tracks:
        (envelope,) = track.envelopes
        time = envelope.sample_ns
        knots = np.interp(
            np.linspace(0, len(time) - 1, (len(time) - 1) * 64 + 1),
            np.arange(len(time)),
            time,
        )
        dense = replace(envelope, sample_ns=knots, samples=envelope.values(knots))
        tracks.append(replace(track, envelopes=(dense,)))
    batch = replace(case.batch, tracks=tuple(tracks))
    # One consumer cannot repay its own DOP853 probe. Duplicate each isolated
    # phase where needed without changing its drive or independent reference.
    stride = int(batch.points.max()) + 1
    batches = tuple(
        replace(batch, points=batch.points + index * stride) for index in range(copies)
    )
    expected_rows = copies * sum(len(spans) for spans in segment_spans(batch))
    with closing(
        Engine(1, families=("multitone_rwa", "dop853"), backend="cpu")
    ) as engine:
        inspection = engine.inspect_dispatch(case.source, batches)
        result = engine.evolve(case.source, batches, telemetry=True)
    assert len(inspection.rows) == expected_rows
    for row in inspection.rows:
        assert row.family == "dop853"
        assert any(
            candidate.family == "multitone_rwa" and candidate.guard == "rejected"
            for candidate in row.candidates
        )
    assert result.paths.segments_by_kernel == {"dop853": expected_rows}
    routing = result.paths.routing["multitone_rwa"]
    assert routing.admission_rows == expected_spots
    assert routing.rejected_rows == expected_rows
    reference = matrix_reference(case)
    error = density_error(result.densities, np.tile(reference.density, (copies, 1, 1)))
    request.node.user_properties.append(
        measurement_property(
            AccuracyMeasurement(
                name + "/spot_fallback", "dop853", error, reference.convergence_error
            )
        )
    )
    assert error <= 2e-3


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
