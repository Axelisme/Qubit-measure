"""Immutable, explicitly versioned one-core estimates for shadow planning only."""

from dataclasses import dataclass
from math import isfinite

from .declarations import AlgorithmIdentity, MemoryDeclaration, StageCosts


@dataclass(frozen=True)
class ProfileDomain:
    """Inclusive dimension/budget domain and workload name for these estimates."""

    workload: str
    dimensions: tuple[int, int]
    budgets: tuple[float, float]

    def __post_init__(self) -> None:
        if (
            not self.workload
            or not 1 <= self.dimensions[0] <= self.dimensions[1]
            or not all(isfinite(x) for x in self.budgets)
            or not 0 < self.budgets[0] <= self.budgets[1]
        ):
            raise ValueError("profile domain must have ordered positive bounds")


@dataclass(frozen=True)
class ProfileProvenance:
    """Evidence origin, not a claim of current-head calibration.

    source_commit, dependencies, cpu_runtime and evidence_digest identify evidence.
    fit_error and held_out_error are relative errors, or None when unmeasured.
    uncertainty explains missing evidence. Synthetic/seed profiles remain
    uncalibrated; acquisition never benchmarks or reads machine-global data.
    """

    source_commit: str
    dependencies: tuple[str, ...]
    cpu_runtime: str
    evidence_digest: str
    uncertainty: str
    fit_error: float | None = None
    held_out_error: float | None = None

    def __post_init__(self) -> None:
        if not all(
            (
                self.source_commit,
                self.cpu_runtime,
                self.evidence_digest,
                self.uncertainty,
            )
        ):
            raise ValueError("profile provenance must be explicit")
        for value in (self.fit_error, self.held_out_error):
            if value is not None and (not isfinite(value) or value < 0):
                raise ValueError("profile errors must be finite and nonnegative")
        object.__setattr__(self, "dependencies", tuple(self.dependencies))


@dataclass(frozen=True)
class FamilyProfile:
    """One algorithm revision's constant per-row/per-key shadow estimates.

    stages uses one-core seconds: guard/apply/basis/lift per row, build per
    distinct build key, bind per distinct binding key, overhead per graph task.
    memory is in bytes, not RSS. Every stage/byte charge must be known.
    There is no worker scaling or inferred batch speedup in this profile schema.
    """

    identity: AlgorithmIdentity
    stages: StageCosts
    memory: MemoryDeclaration

    def __post_init__(self) -> None:
        if any(
            value is None
            for value in (
                self.stages.guard,
                self.stages.build,
                self.stages.bind,
                self.stages.apply,
                self.stages.basis_conversion,
                self.stages.lift,
                self.stages.task_overhead,
                self.memory.resident,
                self.memory.build_workspace,
                self.memory.apply_scratch,
                self.memory.input_bound,
            )
        ):
            raise ValueError("shadow estimates require known costs and memory")


@dataclass(frozen=True)
class OneCoreProfile:
    """Versioned, immutable shadow-only profile with full evidence labels.

    schema must be shadow-one-core-v1. profile_id and revision identify this
    fixed dataset, independently of planner/numeric revisions. domain limits
    use; families match complete AlgorithmIdentity. feature_units explicitly
    names seconds, bytes, dimension and budget. All such profiles are
    uncalibrated and cannot authorize production cost competition.
    """

    profile_id: str
    revision: str
    domain: ProfileDomain
    provenance: ProfileProvenance
    families: tuple[FamilyProfile, ...]
    schema: str = "shadow-one-core-v1"
    feature_units: tuple[str, ...] = (
        "one-core-seconds",
        "bytes",
        "dimension",
        "budget",
    )
    uncertainty: str = "uncalibrated"

    def __post_init__(self) -> None:
        object.__setattr__(self, "families", tuple(self.families))
        object.__setattr__(self, "feature_units", tuple(self.feature_units))
        identities = [family.identity for family in self.families]
        if (
            self.schema != "shadow-one-core-v1"
            or not self.profile_id
            or not self.revision
            or self.uncertainty != "uncalibrated"
            or self.feature_units
            != ("one-core-seconds", "bytes", "dimension", "budget")
            or len(identities) != len(set(identities))
        ):
            raise ValueError("invalid shadow profile identity, units or uncertainty")

    def family(self, identity: AlgorithmIdentity) -> FamilyProfile:
        """Return matching full identity; unsupported algorithm revisions raise ValueError."""
        for family in self.families:
            if family.identity == identity:
                return family
        raise ValueError("algorithm revision is outside the profile")
