"""Explicit constant-workload evidence, owned by numerical evolution.

Profiles may be pending acquire validation. Only Engine's developer evaluation
entry accepts that state; ordinary evolve must require complete evidence.
"""

from dataclasses import dataclass
from functools import lru_cache
from hashlib import sha256
from importlib.metadata import version
from math import isfinite
from pathlib import Path
from typing import Literal

from .engine import SEGMENT_NUMERIC_VERSION
from .profiles import OneCoreProfile

CONSTANT_PLANNER_REVISION = "constant-v2"


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
    planner_revision: str = CONSTANT_PLANNER_REVISION
    numeric_revision: str = SEGMENT_NUMERIC_VERSION

    def __post_init__(self) -> None:
        object.__setattr__(self, "workloads", tuple(self.workloads))
        object.__setattr__(self, "acquire_evidence", tuple(self.acquire_evidence))
        if (
            not self.kernel_source_digest
            or not self.workloads
            or len(set(self.workloads)) != len(self.workloads)
            or self.large_allocation_allowance < 0
            or any(
                not w.input_digest or w.workers < 1 or w.cache_bytes < 0
                for w in self.workloads
            )
        ):
            raise ValueError("invalid constant calibration domain/resources")

    def acquire_status(self) -> Literal["pending", "failed", "validated"]:
        """Require every cold/warm workload observation and both original gates.

        Stage fit errors are diagnostics, not an independent admission threshold.
        Missing all acquire observations is pending; partial or failed evidence
        is failed and cannot use the developer evaluation entry as a bypass.
        """
        if not self.acquire_evidence:
            return "pending"
        expected = {(w, t) for w in self.workloads for t in ("cold", "warm")}
        observed = {(e.workload, e.temperature) for e in self.acquire_evidence}
        if expected != observed or len(observed) != len(self.acquire_evidence):
            return "failed"
        for evidence in self.acquire_evidence:
            values = (
                evidence.legacy_seconds,
                evidence.candidate_seconds,
                evidence.population_error,
                evidence.low_four_error,
            )
            if (
                not all(isfinite(v) and v >= 0 for v in values)
                or evidence.legacy_seconds <= 0
                or evidence.candidate_seconds > evidence.legacy_seconds
                or max(evidence.population_error, evidence.low_four_error) > 0.01
                or not evidence.evidence_digest
            ):
                return "failed"
        return "validated"


@lru_cache(maxsize=1)
def constant_dependencies() -> tuple[str, ...]:
    """Return installed numerical dependency identities for evidence matching."""
    return tuple(
        f"{name}=={version(name)}" for name in ("numpy", "scipy", "numba", "joblib")
    )


@lru_cache(maxsize=1)
def constant_kernel_revision() -> str:
    """Hash the fixed installed numeric source closure once per process.

    Missing/unreadable source raises OSError, never a fabricated match. The
    closure describes loaded kernels; restarting is required after source edits.
    The shell/planner revision remains a separate replay/profile identity.
    """
    names = (
        "kernels/dyson.py",
        "kernels/_dyson_bound.py",
        "kernels/_dyson_paths.py",
        "kernels/_weak.py",
        "kernels/floquet_period.py",
        "kernels/t4.py",
        "kernels/t6.py",
        "kernels/dop853.py",
        "kernels/_dop853.py",
        "kernels/_dop853_chain.py",
        "kernels/_dop853_data.py",
        "secular.py",
        "kernel.py",
        "declarations.py",
        "_validation.py",
        "_envelope.py",
    )
    digest = sha256(b"constant-stage-pilot-v1")
    root = Path(__file__).parent
    for name in names:
        digest.update(name.encode())
        digest.update(sha256((root / name).read_bytes()).digest())
    return digest.hexdigest()
