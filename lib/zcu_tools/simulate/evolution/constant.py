"""Explicit constant profile planning and inline pinned numerical replay.

Bare one-core estimates remain experimental. Engine may select a complete
constant cover only after calibrated evidence admission or explicit developer
candidate evaluation. Only nonempty source-input, fixed-flux, constant zero/
one-tone intervals are supported. No prefix is inferred or evolved here.
"""

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field, replace
from hashlib import sha256
from threading import Event, Lock

import numpy as np
from numpy.typing import NDArray

from ._dispatch import Dispatch
from ._parallel import execution_pool, native_limit
from ._planning import Geometry
from ._schedule import Schedule
from .cache import CacheEntry, PrepareCache
from .calibration import (
    CONSTANT_PLANNER_REVISION,
    CalibratedConstantProfile,
    ConstantWorkload,
    constant_dependencies,
    constant_kernel_revision,
)
from .engine import (
    SEGMENT_NUMERIC_VERSION,
    Cancelled,
    EvolutionResult,
    PathDiagnostics,
    TimelineBatch,
)
from .profiles import OneCoreProfile, ProfileAdmission
from .provider import LegacyKernelAdapter, LegacyProvider, segment
from .secular import ModelSource
from .shadow import (
    ShadowCandidate,
    ShadowPlan,
    ShadowRequest,
    shadow_plan,
)


@dataclass(frozen=True)
class RowCertificate:
    """Explicit guard outcome for one family/row and complete source-input digest.

    row indexes segmented geometry; family names its kernel. accepted is the
    actual numerical guard result, never inferred from cache. input_digest
    identifies source density, models and all controls used for validation.
    """

    row: int
    family: str
    accepted: bool
    input_digest: str


@dataclass(frozen=True)
class PinnedConstantPlan:
    """Experimental numerical assignment and provenance, not an arbitrary graph.

    geometry retains all numerical row controls and point identities, without
    evolved states or prepared payloads. report retains selected cover, safe
    incumbent, algorithm parameters and profile ID/
    revision. certificates retains actual row acceptance with source-input
    identity. input_digest pins every model, density and segmented control.
    assumptions_digest pins request and immutable cache inventory used in scoring.
    budget pins numerical accuracy. admission explains why production must use
    legacy dispatch. planner_revision and numeric_revision must match this
    implementation on replay. Production admission is always False.
    """

    report: ShadowPlan
    geometry: Geometry = field(compare=False, repr=False)
    certificates: tuple[RowCertificate, ...]
    input_digest: str
    assumptions_digest: str
    budget: float
    admission: ProfileAdmission
    planner_revision: str = CONSTANT_PLANNER_REVISION
    numeric_revision: str = SEGMENT_NUMERIC_VERSION


class ConstantPlanner:
    """Own constant candidate validation and inline pinned-cover execution.

    entries explicitly replaces the constant registry; None uses existing built-in
    declarations. cache_bytes is the replay resident prepare limit, not RSS.
    Logical worker width belongs to ShadowRequest and launches no team.
    Replay is inline only; neither it nor plan changes Engine defaults.
    """

    def __init__(
        self,
        entries: Sequence[LegacyKernelAdapter] | None = None,
        *,
        cache_bytes: int = 0,
    ) -> None:
        """Capture declared constant kernels; reject invalid resident cache size."""
        self._provider = LegacyProvider(entries)
        self._cache = PrepareCache(cache_bytes)
        self._lock = Lock()
        self._cache_input_digest = ""

    def plan(
        self,
        source: ModelSource,
        batches: Sequence[TimelineBatch],
        profile: OneCoreProfile,
        request: ShadowRequest,
        *,
        inventory: tuple[CacheEntry, ...] = (),
    ) -> PinnedConstantPlan:
        """Validate guards and score a complete constant acquire using neutral costs.

        request dimension must match source density; its budget controls guards.
        inventory is a serialized immutable cache snapshot for observation only.
        Keys come from kernel.prepare_keys, not caller encoding. t4/t6 must
        accept their entire domains. Unsupported geometry or incumbent profile
        revisions raise ValueError; optional profile misses are diagnosed and
        excluded. Guard-building variants are unsupported. Missing terminals or
        rejected terminal guards raise RuntimeError. Numerical exceptions
        propagate without reassignment or retry.
        """
        return self.plan_geometry(
            source,
            _constant_geometry(source, batches),
            profile,
            request,
            inventory=inventory,
        )

    def plan_geometry(
        self,
        source: ModelSource,
        geometry: Geometry,
        profile: OneCoreProfile,
        request: ShadowRequest,
        *,
        inventory: tuple[CacheEntry, ...] = (),
    ) -> PinnedConstantPlan:
        """Plan complete already segmented geometry under the same constant contract.

        geometry must be the unmodified output of provider.segment for source.
        This avoids repeating segmentation in Engine. All guards, actual input
        certificates, complete incumbent checks and bounded search still run.
        Unsupported geometry raises ValueError before candidate evaluation.
        """
        with self._lock, native_limit.acquire():
            if not _supported_geometry(geometry):
                raise ValueError("unsupported constant geometry")
            return self._plan_geometry(
                source,
                geometry,
                profile,
                request,
                inventory,
                _input_digest(geometry, source),
            )

    def _plan_geometry(
        self,
        source: ModelSource,
        geometry: Geometry,
        profile: OneCoreProfile,
        request: ShadowRequest,
        inventory: tuple[CacheEntry, ...],
        digest: str,
    ) -> PinnedConstantPlan:
        if request.dimension != source.initial_density.shape[0]:
            raise ValueError("request dimension differs from source input")
        candidates: list[ShadowCandidate] = []
        certificates: list[RowCertificate] = []
        rows = np.arange(len(geometry.table.timeline), dtype=np.int64)
        # Constant algorithm eligibility belongs here, not to a plugin
        # declaration. Guards retain the algorithms' numerical authority.
        tones = geometry.table.tones
        available = self._provider.eligible(geometry.table, rows)
        eligible = {
            # Exact secular T4: model build per key, analytic apply per row.
            "t4": rows[tones == 0],
            # Dyson: certified-input binding and retained-path apply below.
            "dyson": rows[tones == 1],
            # Hilbert period: frequency-key build and row endpoint apply;
            # guard owns the carrier floor and splitting subdivision limit.
            "floquet_period": rows[tones == 1],
            # T6: full single-tone incumbent, with d**4 channel allowance.
            "t6": rows[tones == 1],
            # Direct DOP853: cheap model preparation and row integration.
            # Fixed controls/duration belong to the exact calibrated domain.
            "dop853": rows,
        }
        dc = source.model(tuple(0 for _ in range(geometry.table.level.shape[1])))
        for name in ("t4", "dyson", "floquet_period", "t6", "dop853"):
            if name not in available or not len(eligible[name]):
                continue
            entry = self._provider.entry(name)
            # Current constant families validate without building payloads.
            # Builder variants need evolve-local handoff ownership, not stored tables.
            if entry.kernel.guard_builds:
                raise ValueError(
                    "constant planner does not support guard-building variants"
                )
            selected = eligible[name]
            accepted = entry.kernel.guard(
                geometry.table, selected, source, request.budget
            )
            keys = entry.kernel.prepare_keys(geometry.table, selected)
            for row, valid, key in zip(selected, accepted, keys, strict=True):
                port = int(geometry.table.tone_port[row, 0]) if tones[row] else -1
                certificates.append(RowCertificate(int(row), name, bool(valid), digest))
                candidates.append(
                    ShadowCandidate(
                        f"{name}:{row}",
                        int(row),
                        entry.declaration,
                        (
                            name,
                            dc.identity,
                            request.budget,
                        ),
                        key.tobytes(),
                        "accepted" if valid else "rejected",
                        f"{digest}:source-input:port:{port}".encode(),
                    )
                )
        by_id = {candidate.candidate_id: candidate for candidate in candidates}
        safe = tuple(
            f"{'t4' if tones == 0 else 't6'}:{row}"
            for row, tones in enumerate(geometry.table.tones)
        )
        for candidate_id in safe:
            if candidate_id not in by_id:
                raise RuntimeError(
                    f"constant cover requires kernel {candidate_id.split(':')[0]}"
                )
            if by_id[candidate_id].guard != "accepted":
                raise RuntimeError(
                    f"no valid kernel for constant interval {candidate_id}"
                )
        report = shadow_plan(
            geometry.table, tuple(candidates), safe, profile, inventory, request
        )
        assumptions = sha256(repr((request, inventory)).encode()).hexdigest()
        return PinnedConstantPlan(
            report,
            geometry,
            tuple(certificates),
            digest,
            assumptions,
            request.budget,
            profile.production_admission(
                tuple(c.declaration.identity for c in candidates)
            ),
        )

    def _revision_admission(
        self, profile: CalibratedConstantProfile
    ) -> ProfileAdmission:
        if (
            profile.planner_revision != CONSTANT_PLANNER_REVISION
            or profile.numeric_revision != SEGMENT_NUMERIC_VERSION
        ):
            return ProfileAdmission(
                "legacy", False, "unsupported profile execution revision"
            )
        dependencies = profile.estimates.provenance.dependencies
        if dependencies and dependencies != constant_dependencies():
            return ProfileAdmission(
                "legacy", False, "numeric dependency revision mismatch"
            )
        current = tuple(
            self._provider.entry(name).declaration.identity
            for name in ("t4", "dyson", "floquet_period", "t6", "dop853")
        )
        revision = profile.estimates.production_admission(current)
        if not revision.revisions_match:
            return revision
        try:
            source_revision = constant_kernel_revision()
        except OSError:
            return ProfileAdmission("legacy", False, "numeric source unavailable")
        if profile.kernel_source_digest != source_revision:
            return ProfileAdmission("legacy", False, "numeric source revision mismatch")
        return revision

    def _admission(
        self,
        geometry: Geometry,
        source: ModelSource,
        profile: CalibratedConstantProfile,
        *,
        request: ShadowRequest,
        cache_bytes: int,
    ) -> tuple[ProfileAdmission, str | None]:
        """Check exact domain/resources/evidence before guards or assignment.

        geometry is the complete segmented acquire. A pending decision is legacy
        for production and may be evaluated only by Engine's developer entry.
        Stale source, unsupported geometry/resources, failed or incomplete
        observations never authorize production. Numeric model failures propagate.
        """
        revision = self._revision_admission(profile)
        if not revision.revisions_match:
            return revision, None
        if not _supported_geometry(geometry):
            return ProfileAdmission(
                "legacy", True, "unsupported constant geometry"
            ), None
        dimension = source.initial_density.shape[0]
        domain = profile.estimates.domain
        if not (
            domain.dimensions[0] <= dimension <= domain.dimensions[1]
            and domain.budgets[0] <= request.budget <= domain.budgets[1]
        ):
            return ProfileAdmission(
                "legacy", True, "outside calibrated budget/dimension"
            ), None
        digest = _input_digest(geometry, source)
        workload = ConstantWorkload(digest, request.workers, cache_bytes)
        if workload not in profile.workloads:
            return ProfileAdmission(
                "legacy", True, "outside exact workload/resource certificates"
            ), None
        # Preserve the complete T6 incumbent, including its dominant d**4
        # channels, before any numerical candidate guard can run.
        rows = np.flatnonzero(geometry.table.tones == 1).astype(np.int64)
        large = 0
        if len(rows):
            keys = self._provider.entry("t6").kernel.prepare_keys(geometry.table, rows)
            large = len(np.unique(keys)) * (4 * 515 + 64) * dimension**4 * 16
        if large > profile.large_allocation_allowance:
            return ProfileAdmission(
                "legacy", True, "T6 channel allocation exceeds allowance"
            ), None
        status = profile.acquire_status()
        return ProfileAdmission(
            "constant" if status == "validated" else "legacy",
            True,
            status + " matched acquire evidence",
            status == "pending",
        ), digest

    def _dispatch(
        self,
        source: ModelSource,
        geometry: Geometry,
        profile: CalibratedConstantProfile,
        *,
        request: ShadowRequest,
        inventory: tuple[CacheEntry, ...],
        digest: str,
    ) -> Dispatch:
        """Plan an already admitted/evaluation-qualified constant workload.

        Engine owns admission, call serialization and execution. This performs
        all guards and bounded whole-acquire search, but no preparation or apply.
        It never grants production admission or catches numeric failures.
        """
        # Dyson scans about 8*d**5 coupling paths once per distinct certified
        # input binding, then evaluates retained paths per row. No trial density
        # or path construction is used for this cost estimate. In this exact
        # source-input geometry a binding key includes model/port plus the full
        # input/control certificate; unknown prefixes cannot enter this planner.
        # A dense cheap bound covers first/second-order modal paths. Coefficients
        # are normalized only to the measured profile dimension, never to a
        # historical larger model or Kernel.cost()'s relative weight of one.
        dimension = request.dimension
        reference_dimension = profile.estimates.domain.dimensions[0]
        path_bound = 4 * dimension**6 + 2 * dimension**4
        reference_paths = 4 * reference_dimension**6 + 2 * reference_dimension**4
        estimates = replace(
            profile.estimates,
            families=tuple(
                replace(
                    family,
                    stages=replace(
                        family.stages,
                        bind=float(family.stages.bind or 0)
                        * (dimension / reference_dimension) ** 5,
                        apply=float(family.stages.apply or 0)
                        * path_bound
                        / reference_paths,
                    ),
                )
                if family.identity.name == "dyson"
                else family
                for family in profile.estimates.families
            ),
        )
        pinned = self._plan_geometry(
            source, geometry, estimates, request, inventory, digest
        )
        names = np.full(len(pinned.report.selected.cover), "", dtype="<U32")
        for candidate in pinned.report.selected.cover:
            names[candidate.row] = candidate.declaration.identity.name
        return Dispatch(names, PathDiagnostics(), np.ones(len(names)))

    def production_dispatch(
        self,
        source: ModelSource,
        geometry: Geometry,
        profile: CalibratedConstantProfile,
        *,
        request: ShadowRequest,
        cache: PrepareCache,
    ) -> Dispatch | None:
        """Return a qualified production cover, or None before any guards.

        Checks current source, exact numeric workload/resources and complete
        cold/warm evidence. Numerical exceptions propagate, never retry legacy.
        The input snapshot/digest is shared only within this serialized call.
        """
        return self._profile_dispatch(
            source, geometry, profile, request, cache, evaluation=False
        )

    def evaluation_dispatch(
        self,
        source: ModelSource,
        geometry: Geometry,
        profile: CalibratedConstantProfile,
        *,
        request: ShadowRequest,
        cache: PrepareCache,
    ) -> Dispatch:
        """Return a developer cover with only acquire evidence allowed pending.

        Domain/resource/revision mismatch or failed evidence raises ValueError
        before guards. No production admission is granted. All actual inputs
        are certified anew; no digest or evolved state is cached across calls.
        """
        dispatch = self._profile_dispatch(
            source, geometry, profile, request, cache, evaluation=True
        )
        if dispatch is None:
            raise ValueError("constant evaluation has no qualified cover")
        return dispatch

    def _profile_dispatch(
        self,
        source: ModelSource,
        geometry: Geometry,
        profile: CalibratedConstantProfile,
        request: ShadowRequest,
        cache: PrepareCache,
        *,
        evaluation: bool,
    ) -> Dispatch | None:
        with self._lock, native_limit.acquire():
            decision, digest = self._admission(
                geometry,
                source,
                profile,
                request=request,
                cache_bytes=cache.limit,
            )
            if decision.execution != "constant" and not (
                evaluation and decision.pending_acquire
            ):
                if evaluation:
                    raise ValueError("constant evaluation rejected: " + decision.reason)
                return None
            if digest is None:
                raise ValueError("qualified constant workload lacks input certificate")
            return self._dispatch(
                source,
                geometry,
                profile,
                request=request,
                inventory=cache.inventory(),
                digest=digest,
            )

    def replay(
        self,
        pinned: PinnedConstantPlan,
        source: ModelSource,
        batches: Sequence[TimelineBatch],
        *,
        cancel: Event,
    ) -> EvolutionResult:
        """Recompute exactly the pinned assignment inline, without cost search.

        Matching source/model/density/controls, selected kernel identities and
        certificates are mandatory. Unsupported planner/numeric/kernel revisions
        or changed inputs raise ValueError before applying anything. Cache capacity
        may differ from planning. Cancellation raises Cancelled, no partial result.
        Build/bind/apply failures propagate; exceedances warn once and keep output.
        """
        with self._lock, native_limit.acquire():
            if cancel.is_set():
                raise Cancelled()
            if (
                pinned.planner_revision != CONSTANT_PLANNER_REVISION
                or pinned.numeric_revision != SEGMENT_NUMERIC_VERSION
                or pinned.report.planner_revision != "shadow-interval-v1"
            ):
                raise ValueError("unsupported pinned replay revision")
            geometry = _constant_geometry(source, batches)
            digest = _input_digest(geometry, source)
            if (
                digest != pinned.input_digest
                or _input_digest(pinned.geometry, source) != digest
            ):
                raise ValueError("pinned replay input/model/controls differ")
            cover = pinned.report.selected.cover
            if sorted(candidate.row for candidate in cover) != list(
                range(len(geometry.table.timeline))
            ):
                raise ValueError(
                    "pinned cover must contain every interval exactly once"
                )
            proofs = {
                (certificate.row, certificate.family): certificate
                for certificate in pinned.certificates
            }
            names = np.full(len(cover), "", dtype="<U32")
            for candidate in cover:
                name = candidate.declaration.identity.name
                try:
                    entry = self._provider.entry(name)
                except KeyError as error:
                    raise ValueError(
                        "unsupported pinned kernel revision/family"
                    ) from error
                if candidate.declaration != entry.declaration:
                    raise ValueError("unsupported pinned kernel revision or parameters")
                proof = proofs.get((candidate.row, name))
                if proof is None or not proof.accepted or proof.input_digest != digest:
                    raise ValueError("missing or mismatched pinned row certificate")
                row = np.array([candidate.row], dtype=np.int64)
                key = entry.kernel.prepare_keys(geometry.table, row)[0].tobytes()
                dc = source.model(
                    tuple(0 for _ in range(geometry.table.level.shape[1]))
                )
                if (
                    candidate.guard != "accepted"
                    or candidate.prepare_key != key
                    or candidate.namespace != (name, dc.identity, pinned.budget)
                ):
                    raise ValueError("pinned candidate input/key/budget mismatch")
                names[candidate.row] = name
            kernels = {
                name: self._provider.entry(str(name)).kernel
                for name in np.unique(names)
            }
            # Certificates are valid only for the exact pinned inputs above.
            # No guard or cost assignment is repeated, even after cache eviction.
            if self._cache_input_digest != digest:
                self._cache.clear()
                self._cache_input_digest = digest
            dispatch = Dispatch(names, PathDiagnostics(), np.ones(len(names)))
            schedule = Schedule(
                pinned.geometry, dispatch, source, pinned.budget, cancel
            )
            with execution_pool(1) as pool:
                result = schedule.run(kernels, self._cache, pool, 1)
            if cancel.is_set():
                raise Cancelled()
            _warn_exceedances(result)
            return result

    def close(self) -> None:
        """Release replay cache; no worker resources are retained."""
        with self._lock:
            self._cache.clear()


def _constant_geometry(
    source: ModelSource, batches: Sequence[TimelineBatch]
) -> Geometry:
    if not batches or not any(len(batch.points) for batch in batches):
        raise ValueError("constant planner requires a nonempty acquire")
    geometry = segment(source, batches)
    if not _supported_geometry(geometry):
        raise ValueError(
            "constant planner requires one source-input fixed-flux constant interval per timeline"
        )
    geometry.points.setflags(write=False)
    geometry.flux_drive.setflags(write=False)
    return geometry


def _supported_geometry(geometry: Geometry) -> bool:
    table = geometry.table
    rows = np.arange(len(table.timeline), dtype=np.int64)
    return not (
        not len(rows)
        or len(rows) != len(geometry.points)
        or not np.array_equal(np.sort(table.timeline), np.arange(len(geometry.points)))
        or np.any(table.start_ns != 0)
        or np.any(table.tones > 1)
        or not table.constant_envelopes(rows).all()
        or table.flux_waveform is not None
        or np.any(geometry.flux_drive)
        or table.prefix_inputs
    )


def constant_workload_certificate(
    source: ModelSource, batches: Sequence[TimelineBatch]
) -> str:
    """Certify exact constant geometry, numeric source, input and full controls.

    This identifies a finite calibration workload, not guard acceptance. Invalid
    geometry raises ValueError; model errors propagate without fallback.
    """
    return _input_digest(_constant_geometry(source, batches), source)


def _array_token(array: NDArray[np.generic]) -> bytes:
    return repr((array.shape, array.dtype.str)).encode() + array.tobytes()


def _input_digest(geometry: Geometry, source: ModelSource) -> str:
    digest = sha256()
    table = geometry.table
    # The geometry is restricted above: no sampled envelopes, flux or prefixes.
    for array in (
        geometry.points,
        table.timeline,
        table.start_ns,
        table.duration_ns,
        table.level,
        table.tones,
        table.tone_port,
        table.tone_frequency_ghz,
        table.tone_amplitude,
        table.tone_theta_rad,
        table.tone_envelope,
        table.tone_envelope_offset_ns,
        source.initial_density,
    ):
        digest.update(sha256(_array_token(array)).digest())
    levels = {tuple(int(v) for v in level) for level in table.level}
    levels.add(tuple(0 for _ in range(table.level.shape[1])))
    for level in sorted(levels):
        model = source.model(level)
        digest.update(
            repr((level, model.identity, tuple(d.name for d in model.drives))).encode()
        )
        for array in (
            model.energies_ghz,
            model.eigenvectors,
            model.rates_per_ns,
            model.dephasing_per_ns,
        ):
            digest.update(sha256(_array_token(array)).digest())
        for drive in model.drives:
            digest.update(sha256(_array_token(drive.in_phase_ghz)).digest())
            digest.update(sha256(_array_token(drive.quadrature_ghz)).digest())
    return digest.hexdigest()


def _warn_exceedances(result: EvolutionResult) -> None:
    exceeded = result.paths.exceeded_by_kernel
    if exceeded:
        details = ", ".join(
            f"{name}: {count}" for name, count in sorted(exceeded.items())
        )
        warnings.warn(
            f"segment accuracy check exceeded in {sum(exceeded.values())} segments "
            f"({details}); results use the uncorrected approximation",
            RuntimeWarning,
            stacklevel=3,
        )
