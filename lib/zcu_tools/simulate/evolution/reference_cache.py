"""Solver-free reference identity and atomic NPZ storage (ADR-0078)."""

from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile

import numpy as np
from numpy.typing import NDArray

from .engine import TimelineBatch
from .frozen import digest, physical_input
from .secular import ModelSource


@dataclass(frozen=True)
class SolverTolerances:
    """Independent solver local error controls: relative rtol and absolute atol.

    Both values are positive, dimensionless floats supplied by the solver owner.
    This record identifies the existing convergence contract; it runs no solver.
    """

    rtol: float
    atol: float


@dataclass(frozen=True)
class ReferenceResult:
    """Stored oracle output, with no solver or device dependencies.

    density is complex128 (points,M,M), in batch point order and DC energy basis.
    convergence_error is the maximum absolute entrywise coarse/fine difference.
    The producer owns numerical convergence; this record does not certify it.
    """

    density: NDArray[np.complex128]
    convergence_error: float


_COARSE = SolverTolerances(2e-10, 2e-12)
_FINE = SolverTolerances(2e-12, 2e-14)


def reference_key(
    source: ModelSource,
    batch: TimelineBatch,
    *,
    coarse: SolverTolerances = _COARSE,
    fine: SolverTolerances = _FINE,
) -> str:
    """Return the historical version-1 convergence cache key without solving.

    Include identity-free complete physics, both tolerances and convergence
    limit 2e-6. Exclude routing history and solver identity. Model/expansion
    construction errors propagate. Caller retains ownership of source.
    """
    return digest(
        {
            "version": 1,
            "physical": physical_input(source, batch, include_identity=False),
            "coarse": asdict(coarse),
            "fine": asdict(fine),
            "convergence_limit": 2e-6,
        }
    )


def load_reference(
    cache_dir: Path,
    source: ModelSource,
    batch: TimelineBatch,
    *,
    coarse: SolverTolerances = _COARSE,
    fine: SolverTolerances = _FINE,
) -> ReferenceResult:
    """Load a reference, always cache-only and without creating directories.

    Read <reference_key>.npz without pickle. Missing input raises FileNotFoundError
    explaining that reference preparation must happen before measurement. I/O
    and malformed archive errors propagate. This API never invokes a solver.
    The producer and measurement preflight own the convergence acceptance check.
    """
    key = reference_key(source, batch, coarse=coarse, fine=fine)
    destination = cache_dir / f"{key}.npz"
    if not destination.exists():
        raise FileNotFoundError(
            f"reference cache miss: {key}. Re-solving is required: obtain cost approval, "
            "then run the tests-side reference producer before measurement. Keep the same native thread "
            "settings through run_heavy; native thread settings can change input fingerprints."
        )
    with np.load(destination, allow_pickle=False) as saved:
        return ReferenceResult(saved["density"], float(saved["convergence_error"]))


def write_reference(
    cache_dir: Path,
    source: ModelSource,
    batch: TimelineBatch,
    result: ReferenceResult,
    *,
    coarse: SolverTolerances = _COARSE,
    fine: SolverTolerances = _FINE,
) -> str:
    """Atomically publish a producer's reference under the historical key.

    Create cache_dir as needed. Preserve density/convergence_error NPZ fields
    without pickle. Return the filename stem; replace any previous same-key
    entry only once the complete archive is written. Source/model and I/O errors
    propagate; temporary files are removed on write failure. No solver runs here.
    """
    key = reference_key(source, batch, coarse=coarse, fine=fine)
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / f"{key}.npz"
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
    return key
