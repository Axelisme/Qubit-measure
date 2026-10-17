"""Explicit constant-workload evidence, owned by numerical evolution.

Profiles may be pending acquire validation. Only Engine's developer evaluation
entry accepts that state; ordinary evolve must require complete evidence.
"""

from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from pathlib import Path

from .profiles import OneCoreProfile


@dataclass(frozen=True)
class ConstantWorkload:
    """Exact source/input/control certificate and tested execution resources.

    input_digest comes from constant_workload_certificate. workers is actual
    outer capacity; cache_bytes is the engine's opaque resident cache limit.
    Certificates include actual arrays, not merely model names or dimensions.
    """

    input_digest: str
    workers: int
    cache_bytes: int


@dataclass(frozen=True)
class AcquireEvidence:
    """Matched whole-acquire observation, seconds and uncorrected max errors.

    workload identifies exact numeric inputs/resources. temperature is cold or
    warm prepare cache (not JIT cache). legacy_seconds and candidate_seconds
    include checks, planning, guards, preparation, binding and apply. Errors
    compare populations and low-four density against an independent full RHS.
    evidence_digest identifies the raw measurement, not a production-safe flag.
    """

    workload: ConstantWorkload
    temperature: str
    legacy_seconds: float
    candidate_seconds: float
    population_error: float
    low_four_error: float
    evidence_digest: str


@dataclass(frozen=True)
class CalibratedConstantProfile:
    """Immutable stage calibration plus exact, finite acquire support domain.

    estimates records one-core costs, source commit, units, dependencies,
    runtime and fit/held-out diagnostics. kernel_source_digest is the fixed
    numeric closure from constant_kernel_revision, not the repository commit.
    workloads lists only exact validated source/input/control/resource points.
    Empty acquire_evidence means pending, never production admission. Both cold
    and warm observations are required for every supported workload. Only T6
    channel arrays are checked against large_allocation_allowance; other
    workspace is unknown, not zero, and this is not a process RSS guarantee.
    """

    estimates: OneCoreProfile
    kernel_source_digest: str
    workloads: tuple[ConstantWorkload, ...]
    acquire_evidence: tuple[AcquireEvidence, ...] = ()
    large_allocation_allowance: int = 64 * 1024 * 1024


@lru_cache(maxsize=1)
def constant_kernel_revision() -> str:
    """Hash the fixed installed numeric source closure once per process.

    Missing/unreadable source raises OSError, never a fabricated match. The
    closure describes loaded kernels; restarting is required after source edits.
    The shell/planner revision remains a separate replay/profile identity.
    """
    names = (
        "kernels/dyson.py", "kernels/_dyson_bound.py", "kernels/_dyson_paths.py",
        "kernels/_weak.py", "kernels/floquet_period.py", "kernels/t4.py",
        "kernels/t6.py", "kernels/dop853.py", "kernels/_dop853.py",
        "kernels/_dop853_chain.py", "kernels/_dop853_data.py",
        "secular.py", "kernel.py", "declarations.py", "_validation.py", "_envelope.py",
    )
    digest = sha256(b"constant-stage-pilot-v1")
    root = Path(__file__).parent
    for name in names:
        digest.update(name.encode())
        digest.update(sha256((root / name).read_bytes()).digest())
    return digest.hexdigest()
