"""Numeric candidate declarations, independent of engine dispatch and calibration."""

from dataclasses import dataclass, field
from math import isfinite
from typing import Literal, Protocol

import numpy as np
from numpy.typing import NDArray

from .kernel import Kernel, SegmentTable

ParameterValue = str | int | float | bool
Representation = Literal["full-density", "factor-density", "coherent"]
ValidationMode = Literal["no-build", "model-build", "certified-input"]


@dataclass(frozen=True)
class AlgorithmParameter:
    """One algorithm setting: a nonempty name and finite scalar value."""

    name: str
    value: ParameterValue

    def __post_init__(self) -> None:
        _require_name(self.name)
        if isinstance(self.value, float) and not isfinite(self.value):
            raise ValueError("algorithm parameters must be finite")


@dataclass(frozen=True)
class AlgorithmIdentity:
    """Stable name/revision, complete numeric settings and execution variant.

    revision changes when numerical semantics change. parameters contains every
    algorithm setting, not per-row controls. execution_variant distinguishes
    implementations of the same method, such as serial or exclusive-inner.
    """

    name: str
    revision: str
    parameters: tuple[AlgorithmParameter, ...] = ()
    execution_variant: str = "serial"

    def __post_init__(self) -> None:
        for name in (self.name, self.revision, self.execution_variant):
            _require_name(name)
        names = [parameter.name for parameter in self.parameters]
        if len(names) != len(set(names)):
            raise ValueError("algorithm parameter names must be unique")


@dataclass(frozen=True)
class Eligibility:
    """Conservative table-only exclusions; True never certifies a numeric guard.

    min_tones/max_tones bound playing tones; None leaves the upper bound open.
    envelope is any, constant (all active envelopes), or shaped (at least one
    nonconstant active envelope). No density, model building or worker state is
    consulted. Flux applicability and numerical bounds remain guard duties.
    """

    min_tones: int = 0
    max_tones: int | None = None
    envelope: Literal["any", "constant", "shaped"] = "any"

    def mask(self, table: SegmentTable, rows: NDArray[np.int64]) -> NDArray[np.bool_]:
        """Return row-aligned bool eligibility, without calling a numerical guard."""
        accepted = table.tones[rows] >= self.min_tones
        if self.max_tones is not None:
            accepted &= table.tones[rows] <= self.max_tones
        if self.envelope != "any":
            constant = table.constant_envelopes(rows)
            accepted &= constant if self.envelope == "constant" else ~constant
        return accepted

    def __post_init__(self) -> None:
        if self.min_tones < 0 or (
            self.max_tones is not None and self.max_tones < self.min_tones
        ):
            raise ValueError("eligibility requires ordered nonnegative tone counts")


@dataclass(frozen=True)
class GuardDeclaration:
    """Guard validation levels and handoff requirements.

    modes lists supported validation levels, not a demand for certified input.
    builds_model requires guard_prepared to retain its accepted model payload.
    accepts_unknown_input means a guard has an arbitrary-input bound; otherwise
    a missing certificate cannot be treated as known. Rejection is a bool mask.
    Guard/build/apply exceptions propagate and never request fallback or retry.
    """

    modes: tuple[ValidationMode, ...] = ("no-build",)
    builds_model: bool = False
    accepts_unknown_input: bool = True

    def __post_init__(self) -> None:
        if not self.modes or self.builds_model != ("model-build" in self.modes):
            raise ValueError("guard validation modes must match model build handoff")
        if not self.accepts_unknown_input and "certified-input" not in self.modes:
            raise ValueError("input-restricted guards must require certified input")


@dataclass(frozen=True)
class StageCosts:
    """Nonnegative finite one-core costs; None means unknown, never zero.

    guard/build/bind/apply charge those separate stages. basis_conversion and
    lift charge representation transitions. task_overhead is fixed per task.
    Units come from CostDeclaration; legacy Kernel.cost is not an input here.
    """

    guard: float | None = None
    build: float | None = None
    bind: float | None = None
    apply: float | None = None
    basis_conversion: float | None = None
    lift: float | None = None
    task_overhead: float | None = None

    def __post_init__(self) -> None:
        for value in (
            self.guard,
            self.build,
            self.bind,
            self.apply,
            self.basis_conversion,
            self.lift,
            self.task_overhead,
        ):
            if value is not None and (not isfinite(value) or value < 0):
                raise ValueError("stage costs must be finite and nonnegative")


@dataclass(frozen=True)
class BatchCost:
    """One execution-curve point: positive batch_size and its total stage costs."""

    batch_size: int
    costs: StageCosts

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("execution curve batch sizes must be positive")


@dataclass(frozen=True)
class CostDeclaration:
    """Cost units, evidence status and batch/execution dependence.

    stages gives one-row costs; curve gives total costs at increasing batch sizes.
    batch_dependence is unknown, linear, or shared-build. Empty curve is unknown.
    unit is relative-work or one-core-seconds. calibration_id is None until a
    fixed numeric profile exists. uncertainty is uncalibrated or calibrated.
    Uncalibrated declarations cannot enter production cost competition.
    """

    stages: StageCosts = field(default_factory=StageCosts)
    unit: Literal["relative-work", "one-core-seconds"] = "relative-work"
    uncertainty: Literal["uncalibrated", "calibrated"] = "uncalibrated"
    calibration_id: str | None = None
    batch_dependence: Literal["unknown", "linear", "shared-build"] = "unknown"
    curve: tuple[BatchCost, ...] = ()

    def __post_init__(self) -> None:
        if self.uncertainty == "calibrated":
            if self.unit != "one-core-seconds" or not self.calibration_id:
                raise ValueError("calibrated costs require seconds and a profile ID")
        elif self.calibration_id is not None:
            raise ValueError("uncalibrated costs cannot name a calibration profile")
        sizes = [point.batch_size for point in self.curve]
        if sizes != sorted(set(sizes)):
            raise ValueError("execution curve batch sizes must strictly increase")


@dataclass(frozen=True)
class MemoryDeclaration:
    """Byte charges, not RSS: None denotes an unknown nonnegative allocation.

    resident is whole opaque Prepared.nbytes, counted once per allocation.
    resident_lifetime allows cache reuse or evolve-only retention; live consumers
    may outlive eviction. build_workspace/apply_scratch last for their task.
    input_bound lasts for one evolve and never enters the model-only cache.
    """

    resident: int | None = None
    build_workspace: int | None = None
    apply_scratch: int | None = None
    input_bound: int | None = None
    resident_lifetime: Literal["cache", "evolve"] = "cache"

    def __post_init__(self) -> None:
        for value in (
            self.resident,
            self.build_workspace,
            self.apply_scratch,
            self.input_bound,
        ):
            if value is not None:
                _require_bytes(value)


@dataclass(frozen=True)
class RepresentationTransition:
    """Required input/output representations; no implicit projection is allowed.

    full-density is the default. factor-density requires a separability
    certificate; coherent retains populations and secular transfer. Neither
    representation authorizes discarding correlations from a full input.
    """

    input: Representation = "full-density"
    output: Representation = "full-density"


@dataclass(frozen=True)
class ReuseDeclaration:
    """Algorithm reuse namespace and key-completeness requirements.

    namespace partitions algorithms/revisions/variants. Build identity additionally
    includes model identity, budget, representation and kernel prepare key.
    Binding adds input certificate, port and all endpoint-affecting parameters.
    Endpoint identity adds complete span controls and phase history. Equal prepare
    keys alone never authorize input binding or endpoint reuse. Model products
    may cross evolves; bindings/endpoints never do.
    """

    namespace: str

    def __post_init__(self) -> None:
        _require_name(self.namespace)


@dataclass(frozen=True)
class KernelDeclaration:
    """One typed candidate contract; it does not override legacy dispatch.

    identity is stable numerical provenance. eligibility only excludes candidates.
    guard describes validation and accepted build handoff. costs/memory separate
    resources with unknowns explicit. accuracy is exact-secular, adaptive, or
    guarded-approximation, not an end-to-end error guarantee. transition declares
    state representation. reuse owns key namespaces. coverage is interval or
    contiguous-span. fallback is terminal-domain or legacy-order; exceptions
    never fall back. parallel is outer-independent or exclusive-inner; kernels
    must not open a pool without an exclusive CPU lease.
    """

    identity: AlgorithmIdentity
    eligibility: Eligibility
    guard: GuardDeclaration
    accuracy: Literal["exact-secular", "adaptive", "guarded-approximation"]
    reuse: ReuseDeclaration
    costs: CostDeclaration = field(default_factory=CostDeclaration)
    memory: MemoryDeclaration = field(default_factory=MemoryDeclaration)
    transition: RepresentationTransition = field(
        default_factory=RepresentationTransition
    )
    coverage: Literal["interval", "contiguous-span"] = "interval"
    fallback: Literal["terminal-domain", "legacy-order"] = "legacy-order"
    parallel: Literal["outer-independent", "exclusive-inner"] = "outer-independent"


def _require_bytes(value: object) -> None:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("memory bytes must be a nonnegative integer")


def _require_name(value: str) -> None:
    if not value.strip():
        raise ValueError("declaration names and revisions must be nonempty")


class DeclaredKernel(Kernel, Protocol):
    """Built-in kernel with a declaration; legacy injected Kernel stays unchanged."""

    declaration: KernelDeclaration
