"""Cache-only reference preflight for a complete measurement selection."""

from contextlib import closing
from dataclasses import dataclass
from math import isfinite
from pathlib import Path
from time import perf_counter, process_time

from zcu_tools.simulate.evolution import ReferenceResult, load_reference, reference_key

from tools.family_pareto_inputs import Input, digest, physical_input


@dataclass(frozen=True)
class ReferenceCache:
    """Select an existing reference directory; None is a missing configuration.

    Loading is always cache-only. No caller can enable solving or cache writes.
    """

    directory: Path | None


@dataclass(frozen=True)
class PreparedReference:
    """One input's ready reference and untimed loading facts.

    name identifies the input. physical_sha256 includes model identity;
    oracle_sha256 omits it. key is the simulator-owned reference cache key.
    reference is shared by object identity for equal keys in the selection.
    wall_seconds/cpu_seconds cover cache loading, not scene reconstruction.
    Reused references report zero additional load time.
    """

    name: str
    physical_sha256: str
    oracle_sha256: str
    key: str
    reference: ReferenceResult
    wall_seconds: float
    cpu_seconds: float


def load_references(
    inputs: list[Input], cache: ReferenceCache
) -> list[PreparedReference]:
    """Load the whole selection before measurement, with no solver or Engine.

    Inputs are decoded in caller order and every owned source is closed.
    Equal simulator cache keys share one ReferenceResult object. All missing
    names/keys are collected into one FileNotFoundError, even after cache hits.
    No cache directory is created. A missing directory configuration raises
    ValueError. Loaded convergence errors must be finite and at most 2e-6;
    invalid convergence and other codec/I/O errors propagate immediately.
    Return ready references only when the entire selection succeeds. Both
    check-references and run_table use this preflight.
    """
    if cache.directory is None:
        raise ValueError("cache-only reference loading requires --reference-cache DIR")
    references: dict[str, ReferenceResult] = {}
    missing: list[str] = []
    prepared = []
    for item in inputs:
        case = item.build()
        with closing(case.source):
            physical_hash = digest(physical_input(case))
            oracle_hash = digest(physical_input(case, include_identity=False))
            key = reference_key(case.source, case.batch)
            wall_seconds = cpu_seconds = 0.0
            if key not in references:
                wall, cpu = perf_counter(), process_time()
                try:
                    reference = load_reference(cache.directory, case.source, case.batch)
                except FileNotFoundError:
                    missing.append(f"{item.name}: {key}")
                    continue
                wall_seconds, cpu_seconds = perf_counter() - wall, process_time() - cpu
                if (
                    not isfinite(reference.convergence_error)
                    or reference.convergence_error > 2e-6
                ):
                    raise ValueError(
                        f"reference convergence failed: {item.name}: {key}: {reference.convergence_error}"
                    )
                references[key] = reference
            prepared.append(
                PreparedReference(
                    item.name,
                    physical_hash,
                    oracle_hash,
                    key,
                    references[key],
                    wall_seconds,
                    cpu_seconds,
                )
            )
    if missing:
        raise FileNotFoundError(
            "reference cache miss:\n"
            + "\n".join(missing)
            + "\nObtain cost approval, then run the tests-side reference producer before measurement."
        )
    return prepared
