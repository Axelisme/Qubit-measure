"""Importable route validation and terminal reporting for Engine accuracy cases."""

import json
from dataclasses import asdict, dataclass

from zcu_tools.simulate.evolution import PathDiagnostics

ALL_FAMILIES = (
    "t4",
    "adiabatic_map",
    "flux_dop853",
    "slow_envelope",
    "dyson",
    "single_tone_rwa",
    "floquet_period",
    "t6",
    "single_tone_gate",
    "shaped_sambe",
    "gate_cache",
    "period_table",
    "common_frame",
    "multitone_gate",
    "multitone_rwa",
    "floquet",
    "dop853",
    "dop853/opencl",
)


@dataclass(frozen=True)
class AccuracyMeasurement:
    """One case/family observation, independent of pytest outcome.

    case is a printable input/route label; family is the executed kernel name
    with an optional /backend suffix.
    density_error is max absolute error over every point and density entry.
    reference_error is the coarse/fine convergence error for the same input.
    Both errors come from finite densities via reference.density_error.
    """

    case: str
    family: str
    density_error: float
    reference_error: float


def measurement_property(measurement: AccuracyMeasurement) -> tuple[str, str]:
    """Encode one observation for pytest user_properties, including xdist transport."""
    return "evolution_accuracy", json.dumps(asdict(measurement))


def reported_measurements(
    properties: list[tuple[str, object]],
) -> list[AccuracyMeasurement]:
    """Decode accuracy properties; unrelated pytest properties are ignored.

    Malformed accuracy properties fail rather than silently hiding measurements.
    """
    measurements = []
    for name, value in properties:
        if name == "evolution_accuracy":
            if not isinstance(value, str):
                raise TypeError("accuracy property must be a JSON string")
            measurements.append(AccuracyMeasurement(**json.loads(value)))
    return measurements


def require_families(paths: PathDiagnostics, expected: tuple[str, ...]) -> None:
    """Raise AssertionError if any expected family executed no segment."""
    missing = [name for name in expected if paths.segments_by_kernel.get(name, 0) == 0]
    assert not missing, (
        f"expected families not exercised: {missing}; got {dict(paths.segments_by_kernel)}"
    )


def accuracy_table(
    measurements: list[AccuracyMeasurement], families: tuple[str, ...]
) -> list[str]:
    """Format per-family counts, maxima, worst case and 2e-3 pass/fail status.

    Families without measurements show NOT EXERCISED and dashes, not zero error.
    Counts are case/family observations, not point or segment counts.
    """
    lines = ["family cases max_density_error reference_error worst_case status"]
    for family in families:
        samples = [item for item in measurements if item.family == family]
        if not samples:
            lines.append(f"{family} 0 - - - NOT EXERCISED")
            continue
        worst = max(samples, key=lambda item: (item.density_error, item.case))
        convergence = max(item.reference_error for item in samples)
        status = "PASS" if worst.density_error <= 2e-3 else "FAIL"
        lines.append(
            f"{family} {len(samples)} {worst.density_error:.6e} "
            f"{convergence:.6e} {worst.case} {status}"
        )
    edges = sorted(
        (item for item in measurements if item.case.startswith("guard_edge/")),
        key=lambda item: (item.family, item.case),
    )
    if edges:
        lines.append("Guard edges: family case density_error reference_error status")
        for item in edges:
            status = "PASS" if item.density_error <= 2e-3 else "FAIL"
            lines.append(
                f"{item.family} {item.case} {item.density_error:.6e} "
                f"{item.reference_error:.6e} {status}"
            )
    return lines
