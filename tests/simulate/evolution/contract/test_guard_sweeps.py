"""Finite guard-error curves at Engine.evolve, not a new accuracy exception.

Gain curves retain the coarse grid and adjacent local 2e-3 brackets. The gate
curve keeps the same [.8,1,.8] normalized envelope for both baths and every V
and split, including the observed nonmonotone low-Gamma crossings. Exact input
builders are importable without pytest. Existing unity edge cases stay intact.
Known exceedances are marked only after route and two-level oracle validation.
"""

from collections.abc import Callable
from contextlib import closing

import pytest
from zcu_tools.simulate.evolution import Engine

from .accuracy import AccuracyMeasurement, measurement_property, require_families
from .accuracy_cases import AccuracyCase
from .guard_sweep_cases import (
    gate_shape_bridge_case,
    gate_split_case,
    scaled_edge_case,
)
from .reference import ReferenceResult, converged_reference, density_error

SCALED_SAMPLES = (
    ("single_tone_rwa", 0.05, 0.002056922640059744),
    ("single_tone_rwa", 0.1, 0.004026441344577358),
    ("single_tone_rwa", 0.2, 0.007368771189371792),
    ("single_tone_rwa", 0.3, 0.009421756111037571),
    ("single_tone_rwa", 0.4, 0.012146521056116133),
    ("single_tone_rwa", 0.5, 0.016884327685639938),
    ("single_tone_rwa", 0.6, 0.020885507361452307),
    ("single_tone_rwa", 0.7, 0.02338565582638955),
    ("single_tone_rwa", 0.8, 0.02372345902526813),
    ("single_tone_rwa", 0.9, 0.02142050511880427),
    ("single_tone_rwa", 1, 0.021802081728112876),
    ("common_frame", 0.05, None),
    ("common_frame", 0.1, None),
    ("common_frame", 0.2, None),
    ("common_frame", 0.3, None),
    ("common_frame", 0.4, None),
    ("common_frame", 0.5, None),
    ("common_frame", 0.6, 0.0022338288935597624),
    ("common_frame", 0.7, 0.002780801440299356),
    ("common_frame", 0.8, 0.004327340785507475),
    ("common_frame", 0.9, 0.005249151472340078),
    ("common_frame", 1, 0.008066669226955314),
    ("shaped_sambe", 0.05, 0.006976190290283084),
    ("shaped_sambe", 0.1, 0.013939786732875038),
    ("shaped_sambe", 0.2, 0.009994325894870235),
    ("shaped_sambe", 0.3, 0.003890045479847463),
    ("shaped_sambe", 0.4, 0.005218876551386592),
    ("shaped_sambe", 0.5, 0.00522067773534268),
    ("shaped_sambe", 0.6, 0.0043120889572121545),
    ("shaped_sambe", 0.7, 0.005127072522687026),
    ("shaped_sambe", 0.8, 0.0030436859867629283),
    ("shaped_sambe", 0.9, 0.0027439229863831147),
    ("shaped_sambe", 1, 0.0030915535285839482),
    ("single_tone_rwa", 0.048593750000000005, None),
    ("single_tone_rwa", 0.04859765625, 0.0020000189837015014),
    ("common_frame", 0.5242578124999999, None),
    ("common_frame", 0.524296875, 0.002000082144536808),
    ("shaped_sambe", 0.01431640625, None),
    ("shaped_sambe", 0.0143359375, 0.002000606697780786),
)
GATE_SAMPLES = (
    (1, 0.018, 0.0001, None),
    (1, 0.018, 0.00025, 0.002172180191409219),
    (1, 0.018, 0.0005, 0.005515447677048631),
    (1, 0.018, 0.001, 0.005880947950789922),
    (1, 0.018, 0.002, 0.0020873329106250948),
    (1, 0.018, 0.003, 0.002917304387132671),
    (1, 0.018, 0.004, 0.007307560557721029),
    (1, 0.018, 0.005, 0.0080745542318555),
    (1, 0.018, 0.006, 0.009301157648956159),
    (1, 0.018, 0.007, 0.008645463048205817),
    (1, 0.018, 0.008, 0.009450772257367157),
    (1, 0.018, 0.009, 0.008587737685302781),
    (1, 0.018, 0.00999999, 0.006692074814160416),
    (0.01, 0.018, 0.0001, None),
    (0.01, 0.018, 0.00025, None),
    (0.01, 0.018, 0.0005, None),
    (0.01, 0.018, 0.001, None),
    (0.01, 0.018, 0.002, None),
    (0.01, 0.018, 0.003, 0.0022881885442749953),
    (0.01, 0.018, 0.004, 0.003015382710179194),
    (0.01, 0.018, 0.005, 0.0039805202839394436),
    (0.01, 0.018, 0.006, 0.004371803365788659),
    (0.01, 0.018, 0.007, 0.005033371893005003),
    (0.01, 0.018, 0.008, 0.005335892195141911),
    (0.01, 0.018, 0.009, 0.004983074897931769),
    (0.01, 0.018, 0.00999999, 0.005381528914108578),
    (0.01, 0.05, 0.00999999, None),
    (0.01, 0.1, 0.00999999, None),
    (1, 0.018, 0.00018984375000000002, None),
    (1, 0.018, 0.00019023437500000003, 0.002002034486524994),
    (0.01, 0.018, 0.002271875, None),
    (0.01, 0.018, 0.0022720703125, 0.002000054029053949),
    (0.01, 0.018, 0.002356640625, 0.0020010038115975412),
    (0.01, 0.018, 0.0023568359374999998, None),
    (0.01, 0.018, 0.0024378906250000004, None),
    (0.01, 0.018, 0.0024380859375000004, 0.002000723587033869),
    (0.01, 0.018, 0.0027375, 0.002000106363492022),
    (0.01, 0.018, 0.0027376953125, None),
    (0.01, 0.018, 0.0027603515624999996, None),
    (0.01, 0.018, 0.0027605468749999995, 0.0020001496452692),
)


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
