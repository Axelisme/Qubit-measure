"""Experimental constant planning and inline pinned numerical replay.

This developer seam never changes Engine dispatch or admits production physics.
Only nonempty acquires with one source-input, fixed-flux, constant zero/one-tone
interval per timeline are supported. No prefix is inferred or evolved here.
"""

import warnings
from collections.abc import Sequence
from dataclasses import dataclass, field
from hashlib import sha256
from threading import Event, Lock

import numpy as np
from numpy.typing import NDArray

from ._dispatch import Dispatch
from ._parallel import execution_pool, native_limit
from ._planning import Geometry
from ._schedule import Schedule
from .cache import CacheEntry, PrepareCache
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
from .shadow import ShadowCandidate, ShadowPlan, ShadowRequest, shadow_plan

CONSTANT_PLANNER_REVISION = "experimental-constant-v1"


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
        with self._lock, native_limit.acquire():
            geometry = _constant_geometry(source, batches)
            if request.dimension != source.initial_density.shape[0]:
                raise ValueError("request dimension differs from source input")
            digest = _input_digest(geometry, source)
            candidates: list[ShadowCandidate] = []
            certificates: list[RowCertificate] = []
            rows = np.arange(len(geometry.table.timeline), dtype=np.int64)
            eligible = self._provider.eligible(geometry.table, rows)
            dc = source.model(tuple(0 for _ in range(geometry.table.level.shape[1])))
            for name in ("t4", "dyson", "floquet_period", "t6", "dop853"):
                if name not in eligible or not len(eligible[name]):
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
                    certificates.append(
                        RowCertificate(int(row), name, bool(valid), digest)
                    )
                    candidates.append(
                        ShadowCandidate(
                            f"{name}:{row}",
                            int(row),
                            entry.declaration,
                            (
                                entry.declaration.reuse.namespace,
                                dc.identity,
                                request.budget,
                            ),
                            key.tobytes(),
                            "accepted" if valid else "rejected",
                            f"{digest}:row:{row}".encode(),
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
                    or candidate.namespace
                    != (entry.declaration.reuse.namespace, dc.identity, pinned.budget)
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
    table = geometry.table
    rows = np.arange(len(table.timeline), dtype=np.int64)
    if (
        len(rows) != len(geometry.points)
        or not np.array_equal(np.sort(table.timeline), np.arange(len(geometry.points)))
        or np.any(table.start_ns != 0)
        or np.any(table.tones > 1)
        or not table.constant_envelopes(rows).all()
        or table.flux_waveform is not None
        or np.any(geometry.flux_drive)
        or table.prefix_inputs
    ):
        raise ValueError(
            "constant planner requires one source-input fixed-flux constant interval per timeline"
        )
    geometry.points.setflags(write=False)
    geometry.flux_drive.setflags(write=False)
    return geometry


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
