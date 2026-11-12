"""Optional disk caching for converged, complete physical-input references."""

from contextlib import closing
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter, process_time
from typing import Protocol

import numpy as np
from zcu_tools.simulate.evolution import ModelSource, TimelineBatch

from tests.simulate.evolution.contract.accuracy_cases import AccuracyCase
from tests.simulate.evolution.contract.interaction_reference import (
    interaction_reference,
)
from tests.simulate.evolution.contract.reference import (
    ReferenceResult,
    SolverTolerances,
    converged_reference,
)
from tools.family_pareto_inputs import Input, digest, physical_input

_COARSE = SolverTolerances(2e-10, 2e-12)
_FINE = SolverTolerances(2e-12, 2e-14)


@dataclass(frozen=True)
class ReferenceCache:
    """Cache location and solve permission travel together through table callers."""

    directory: Path | None
    only: bool = False


@dataclass(frozen=True)
class PreparedReference:
    """Complete reference facts; times include either solving or cache loading."""

    name: str
    physical_sha256: str
    density_shape: tuple[int, ...]
    convergence_error: float
    wall_seconds: float
    cpu_seconds: float


def prepare_references(
    inputs: list[Input], cache: ReferenceCache
) -> list[PreparedReference]:
    """Populate or verify the oracle cache without starting any Engine session."""
    if cache.directory is None:
        raise ValueError("reference preparation requires a cache directory")
    prepared = []
    for item in inputs:
        case = item.build()
        with closing(case.source):
            physical_hash = digest(physical_input(case, include_identity=False))
            wall, cpu = perf_counter(), process_time()
            result = solve_reference(
                case,
                solver=interaction_reference
                if item.category == "real"
                else converged_reference,
                cache_dir=cache.directory,
                cache_only=cache.only,
            )
            elapsed_wall, elapsed_cpu = perf_counter() - wall, process_time() - cpu
            prepared.append(
                PreparedReference(
                    item.name,
                    physical_hash,
                    result.density.shape,
                    result.convergence_error,
                    elapsed_wall,
                    elapsed_cpu,
                )
            )
    return prepared


class ReferenceSolver(Protocol):
    """Converged oracle consuming full physics and two local error controls."""

    def __call__(
        self,
        source: ModelSource,
        batch: TimelineBatch,
        *,
        coarse: SolverTolerances,
        fine: SolverTolerances,
    ) -> ReferenceResult:
        """Return point-aligned density and convergence error; failures propagate."""
        ...


def solve_reference(
    case: AccuracyCase,
    *,
    solver: ReferenceSolver = converged_reference,
    coarse: SolverTolerances = _COARSE,
    fine: SolverTolerances = _FINE,
    cache_dir: Path | None = None,
    cache_only: bool = False,
) -> ReferenceResult:
    """Solve the complete input, optionally reusing a converged on-disk reference.

    The cache is off by default. Its key includes all physical inputs, both
    tolerance levels and the convergence contract version. Solvers must compute
    the same full-Lindblad equation. Nonphysical model identity is omitted.
    The caller owns case.source. Solver and cache I/O failures propagate.
    cache_only never solves: a missing directory/key raises FileNotFoundError,
    and omitting cache_dir raises ValueError. Use it for exclusive timing runs.
    """
    if cache_dir is None:
        if cache_only:
            raise ValueError("cache-only reference requires a cache directory")
        return solver(case.source, case.batch, coarse=coarse, fine=fine)
    key = digest(
        {
            "version": 1,
            "physical": physical_input(case, include_identity=False),
            "coarse": asdict(coarse),
            "fine": asdict(fine),
            "convergence_limit": 2e-6,
        }
    )
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / f"{key}.npz"
    if destination.exists():
        with np.load(destination, allow_pickle=False) as saved:
            return ReferenceResult(saved["density"], float(saved["convergence_error"]))
    if cache_only:
        raise FileNotFoundError(f"reference cache miss: {key}")
    result = solver(case.source, case.batch, coarse=coarse, fine=fine)
    # Publish a complete file so concurrent readers never load a partial archive.
    with NamedTemporaryFile(dir=cache_dir, suffix=".npz", delete=False) as temporary:
        temporary_path = Path(temporary.name)
        try:
            np.savez(
                temporary,
                density=result.density,
                convergence_error=result.convergence_error,
            )
            temporary.flush()
            temporary_path.replace(destination)
        finally:
            temporary_path.unlink(missing_ok=True)
    return result
