"""Optional disk caching for converged, complete physical-input references."""

from contextlib import closing
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from time import perf_counter, process_time
from typing import Protocol

import numpy as np
from zcu_tools.simulate.evolution import ModelSource, TimelineBatch, reference_key

from tests.simulate.evolution.contract.interaction_reference import (
    interaction_reference,
)
from tests.simulate.evolution.contract.reference import (
    ReferenceResult,
    SolverTolerances,
    converged_reference,
)
from tools.family_pareto_inputs import Input, Scene, digest, physical_input

_COARSE = SolverTolerances(2e-10, 2e-12)
_FINE = SolverTolerances(2e-12, 2e-14)


@dataclass(frozen=True)
class ReferenceCache:
    """Configure persistent reference caching and permission to solve cache misses.

    directory is the local cache directory, or None to disable persistent
    caching. only=False allows solving on a miss; only=True requires a cache
    hit and never solves. Construction does not validate these combinations.
    solve_reference requires a directory when cache_only=True, while
    prepare_references always requires a non-None directory.
    """

    directory: Path | None
    only: bool = False


@dataclass(frozen=True)
class PreparedReference:
    """Describe one input's solved or cache-loaded reference.

    name is the stable Input name. physical_sha256 is the SHA-256 digest of
    the complete physical input excluding nonphysical model identity. It is
    not the cache key, which also includes tolerances and convergence version.
    density_shape gives the point-aligned density array dimensions as
    (points, M, M), where M is the model dimension.
    convergence_error is the dimensionless maximum absolute density difference
    between the coarse and fine tolerance solves, over all points and entries.
    wall_seconds is elapsed wall time and cpu_seconds is whole-process CPU
    time, both in seconds, for solving or loading this reference. These times
    exclude input construction and source cleanup.
    """

    name: str
    physical_sha256: str
    density_shape: tuple[int, ...]
    convergence_error: float
    wall_seconds: float
    cpu_seconds: float


def prepare_references(
    inputs: list[Input], cache: ReferenceCache
) -> list[PreparedReference]:
    """Prepare each input's reference in a persistent cache without Engine timing.

    inputs supplies stable names and builders that each create a fresh owned
    source. cache selects the directory and whether misses may be solved;
    cache.directory must be non-None even when cache.only=False. Each input is
    built in order, and its source is closed after solving or loading, including
    on failure. Real inputs use the interaction-picture solver; other inputs
    use the lab-frame solver, both with the default coarse and fine tolerances.
    Return one PreparedReference per input in the same order, with its physical
    hash, density shape, convergence error and solve/load times. No Engine
    session or measurement starts.

    Raise ValueError if cache.directory is None. A cache-only miss raises
    FileNotFoundError without solving. Builder, solver and cache I/O errors
    propagate; earlier cache writes remain if a later input fails.
    """
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
    case: Scene,
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
    key = reference_key(case.source, case.batch, coarse=coarse, fine=fine)
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / f"{key}.npz"
    if destination.exists():
        with np.load(destination, allow_pickle=False) as saved:
            return ReferenceResult(saved["density"], float(saved["convergence_error"]))
    if cache_only:
        raise FileNotFoundError(
            f"reference cache miss: {key}. Re-solving is required: obtain cost approval, "
            "then run --prepare-references before measurement. Keep the same native thread "
            "settings through run_heavy; native thread settings can change input fingerprints."
        )
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
