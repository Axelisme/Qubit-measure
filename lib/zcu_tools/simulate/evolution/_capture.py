"""Caller-owned frozen inputs and actual execution-route observations."""

from __future__ import annotations

from collections.abc import Collection, Generator, Sequence
from contextlib import contextmanager
from copy import copy
from threading import Lock
from typing import TYPE_CHECKING, Literal, TypedDict
from uuid import uuid4

import numpy as np
from numpy.typing import NDArray
from pydantic import TypeAdapter

from ._inspection import GuardDecision, RowKey

if TYPE_CHECKING:
    from ._planning import Geometry
    from .kernels._multitone_spot import SampledGroup
from .engine import SEGMENT_NUMERIC_VERSION, TimelineBatch
from .frozen import FROZEN_PHYSICAL_VERSION, FrozenRecord, physical_input
from .secular import ModelSource


class CaptureCandidate(TypedDict):
    """Offered family and its last guard decision within this classification round."""

    family: str
    guard: GuardDecision


class CaptureRoute(TypedDict):
    """One actual row classification; coordinates follow RowKey within its batch.

    start_ns/duration_ns are absolute ns and span length. round distinguishes
    caller classification from worker resolution. prefix_run indexes this
    record's prefix_runs, or is None for the main run. family is None while
    deferred. candidates retain first-offer order without costs/certificates.
    """

    point: int
    segment: int
    start_ns: float
    duration_ns: float
    round: Literal["caller", "worker"]
    prefix_run: int | None
    family: str | None
    candidates: list[CaptureCandidate]


class CapturePosition(TypedDict):
    """Point ID and chronological split-segment index, as in RowKey."""

    point: int
    segment: int


class CaptureRowKey(CapturePosition):
    """Point and segment with their zero-based batch index in the same evolve."""

    batch: int


class CaptureAdopter(CaptureRowKey):
    """A prefix-input adopting row in the same evolve."""


class CapturePrefixRun(TypedDict):
    """One executed prefix, located in its representative batch's record.

    representative is the group row immediately after the shared prefix.
    adopters includes that row and every other row using the computed input.
    """

    representative: CapturePosition
    adopters: list[CaptureAdopter]


class CaptureSampledSpot(CaptureRowKey):
    """A selected probe row and its observed error, if actually compared.

    max_density_error is the maximum elementwise absolute difference between
    the full RWA and CPU DOP853 probe densities, or None if not compared.
    """

    max_density_error: float | None


class CaptureSampledAdmission(TypedDict):
    """One considered amplitude group's sampled multitone RWA decision.

    rows lists its complete-inventory members. spots lists the distinct chosen
    probe rows. accepted/rejected mean a completed numerical guard, not the
    final route. cost_skipped means a cheaper competitor made the probe
    unnecessary. not_probed means RWA was preferred on price or allowed by a
    pin, but no comparison completed. Uncompared spots have no error value.
    """

    rows: list[CaptureRowKey]
    decision: Literal["accepted", "rejected", "cost_skipped", "not_probed"]
    spots: list[CaptureSampledSpot]


class CaptureMetadata(TypedDict):
    """JSON metadata shared by capture producers and readers.

    capture_id identifies one evolve, not the buffer. batch_index/count include
    empty batches. budget and sorted families (None means unrestricted) are the
    Engine controls. source_identity is the frozen DC model identity. Version
    fields preserve the numeric-algorithm string and physical-codec integer.
    complete becomes True only on normal return. routes contain actual rounds;
    prefix_runs describe shared execution, not fabricated adopter work.
    sampled_rwa_admissions records one entry per considered amplitude group
    in its first member's batch, including groups with members in other batches.
    """

    capture_id: str
    batch_index: int
    batch_count: int
    budget: float
    families: list[str] | None
    source_identity: str
    segment_numeric_version: str
    frozen_physical_version: int
    complete: bool
    routes: list[CaptureRoute]
    prefix_runs: list[CapturePrefixRun]
    sampled_rwa_admissions: list[CaptureSampledAdmission]


_CAPTURE_METADATA_ADAPTER = TypeAdapter(CaptureMetadata)


def read_capture_metadata(record: FrozenRecord) -> CaptureMetadata:
    """Validate record.metadata and return the shared capture metadata mapping.

    Pass a FrozenRecord, normally from read_frozen_manifest. Non-dict metadata,
    missing required keys, wrong types or Literal values at any depth, and extra
    keys at any depth raise the original pydantic.ValidationError with field
    paths. No defaults, coercion or compatibility are supplied. Only JSON shape
    is checked, not cross-field relationships, solver conditions or guard
    numerics. Does not reconstruct a model or modify the record.
    """
    return _CAPTURE_METADATA_ADAPTER.validate_python(
        record.metadata, strict=True, extra="forbid"
    )


class CaptureBuffer:
    """Accumulate optional Engine.evolve recordings in memory, without I/O.

    Construct without arguments and pass as capture=. Read records after evolve
    returns or raises. Each call appends one FrozenRecord per batch, including
    empty batches, with a fresh call ID. records is a tuple snapshot, but opaque
    metadata dicts are not deeply frozen; callers must not modify them.
    Failed freezing leaves only batches already frozen. Execution failures and
    cancellation retain those records with complete=False and observed events.
    Exceptions propagate unchanged. A concurrent evolve using this same buffer
    raises ValueError at entry, even when waiting for an Engine lock.

    Engine currently registers shifts only from batch.segment_shifts, which the
    physical codec preserves and reconstructs. Source registry contents are not
    serialized. A future registration path outside batch corrections must add
    an explicit non-reconstructibility marker at that time.
    """

    def __init__(self) -> None:
        self._records: list[FrozenRecord] = []
        self._lock = Lock()

    @property
    def records(self) -> tuple[FrozenRecord, ...]:
        """Return an append-order tuple snapshot; read only after evolve ends."""
        return tuple(self._records)

    @contextmanager
    def recording(
        self,
        source: ModelSource,
        batches: Sequence[TimelineBatch],
        budget: float,
        families: Collection[str] | None,
    ) -> Generator[list[CaptureMetadata]]:
        """Own one Engine recording scope and yield metadata for route events.

        source supplies the numeric models and initial density. batches is this
        evolve's ordered input, including empty batches. budget is the solve
        allowance recorded with those inputs; families is the Engine allowlist,
        or None for unrestricted routing. This method does not select families.

        Entry creates a fresh capture_id, freezes batches in order and appends
        incomplete records. The yielded list belongs to this scope and lets
        Engine append route events. Normal scope exit marks all its records
        complete=True. Every exit sorts recorded events and releases ownership.
        Concurrent entry raises ValueError immediately. Freezing failure leaves
        only batches already appended; those remain incomplete. Scope and
        freezing exceptions propagate unchanged with all observed events retained.

        Other callers should pass this buffer to Engine.evolve, rather than
        enter recording themselves. Python permits direct entry: it only freezes
        inputs, adds no route events, and still marks complete=True on normal exit.
        """
        if not self._lock.acquire(blocking=False):
            raise ValueError("capture buffer is already recording an evolve")
        metadata: list[CaptureMetadata] = []
        try:
            capture_id = uuid4().hex
            for index, batch in enumerate(batches):
                physical = physical_input(source, batch)
                item: CaptureMetadata = {
                    "capture_id": capture_id,
                    "batch_index": index,
                    "batch_count": len(batches),
                    "budget": budget,
                    "families": None if families is None else sorted(families),
                    "source_identity": source.model(
                        (0,) * batch.flux.level.shape[1]
                    ).content_identity,
                    "segment_numeric_version": SEGMENT_NUMERIC_VERSION,
                    "frozen_physical_version": FROZEN_PHYSICAL_VERSION,
                    "complete": False,
                    "routes": [],
                    "prefix_runs": [],
                    "sampled_rwa_admissions": [],
                }
                self._records.append(
                    FrozenRecord(f"{capture_id}-{index}", physical, item)
                )
                metadata.append(item)
            yield metadata
            for item in metadata:
                item["complete"] = True
        finally:
            try:
                for item in metadata:
                    order = {
                        int(point): index
                        for index, point in enumerate(
                            batches[item["batch_index"]].points
                        )
                    }
                    item["sampled_rwa_admissions"].sort(
                        key=lambda group: (
                            order[group["rows"][0]["point"]],
                            group["rows"][0]["segment"],
                        )
                    )
                    item["routes"].sort(
                        key=lambda event: (
                            event["prefix_run"]
                            if event["prefix_run"] is not None
                            else len(item["prefix_runs"]),
                            event["round"] == "worker",
                            order[event["point"]],
                            event["segment"],
                        )
                    )
            finally:
                self._lock.release()


class CaptureRun:
    """Map local classifier rows to original batch coordinates without pricing."""

    def __init__(
        self,
        geometry: Geometry,
        batches: Sequence[TimelineBatch],
        metadata: list[CaptureMetadata],
    ):
        self.metadata = metadata
        batch_indices = [
            index for index, batch in enumerate(batches) for _ in batch.points
        ]
        table = geometry.table
        first = np.searchsorted(table.timeline, table.timeline, side="left")
        self.keys = tuple(
            RowKey(
                batch_indices[int(timeline)],
                int(geometry.points[timeline]),
                row - int(first[row]),
            )
            for row, timeline in enumerate(table.timeline)
        )
        self.prefix_run: int | None = None
        self._sampled: dict[int, CaptureSampledAdmission] = {}
        self._sampled_lock = Lock()

    def sampled_group(
        self,
        group: SampledGroup,
        decision: Literal["accepted", "rejected", "cost_skipped", "not_probed"],
        errors: NDArray[np.float64] | None = None,
    ) -> None:
        """Record complete-table group coordinates without retaining probe states.

        errors aligns with group.spot_rows on numerical acceptance/rejection.
        Price decisions omit errors and cannot replace a completed comparison.
        Call on the complete CaptureRun, not a local prefix view.
        """
        first = group.rows[0]
        with self._sampled_lock:
            item = self._sampled.get(first)
            if (
                errors is None
                and item is not None
                and item["decision"] in {"accepted", "rejected"}
            ):
                return
            spots: list[CaptureSampledSpot] = []
            for index, row in enumerate(group.spot_rows):
                key = self.keys[row]
                spots.append(
                    {
                        "batch": key.batch,
                        "point": key.point,
                        "segment": key.segment,
                        "max_density_error": None
                        if errors is None
                        else float(errors[index]),
                    }
                )
            if item is not None:
                item["decision"] = decision
                item["spots"] = spots
            else:
                admission: CaptureSampledAdmission = {
                    "rows": [
                        {"batch": key.batch, "point": key.point, "segment": key.segment}
                        for key in (self.keys[row] for row in group.rows)
                    ],
                    "decision": decision,
                    "spots": spots,
                }
                self._sampled[first] = admission
                self.metadata[self.keys[first].batch]["sampled_rwa_admissions"].append(
                    admission
                )

    def prefix(
        self, geometry: Geometry, first: int, row: int, group: int
    ) -> CaptureRun:
        key = self.keys[row]
        runs = self.metadata[key.batch]["prefix_runs"]
        prefix = copy(self)
        prefix.keys = self.keys[first:row]
        prefix.prefix_run = len(runs)
        runs.append(
            {
                "representative": {"point": key.point, "segment": key.segment},
                "adopters": [
                    {
                        "batch": adopter.batch,
                        "point": adopter.point,
                        "segment": adopter.segment,
                    }
                    for adopter, candidate in zip(
                        self.keys, geometry.table.prefix_group or (), strict=True
                    )
                    if candidate == group
                ],
            }
        )
        return prefix

    def round(self, geometry: Geometry, *, worker: bool = False) -> CaptureRound:
        return CaptureRound(self, geometry, worker=worker)


class CaptureRound:
    """Keep only decisions actually observed during one classification round."""

    def __init__(self, run: CaptureRun, geometry: Geometry, *, worker: bool):
        self.run = run
        self.table = geometry.table
        self.worker = worker
        self.candidates: dict[int, dict[str, CaptureCandidate]] = {}

    def observe(
        self, family: str, rows: NDArray[np.int64], guard: GuardDecision
    ) -> None:
        for row in rows:
            candidates = self.candidates.setdefault(int(row), {})
            if guard == "untested_after_short_circuit" and family in candidates:
                continue
            candidates[family] = {"family": family, "guard": guard}

    def finish(self, names: NDArray[np.str_]) -> None:
        for row, candidates in self.candidates.items():
            key = self.run.keys[row]
            self.run.metadata[key.batch]["routes"].append(
                {
                    "point": key.point,
                    "segment": key.segment,
                    "start_ns": float(self.table.start_ns[row]),
                    "duration_ns": float(self.table.duration_ns[row]),
                    "round": "worker" if self.worker else "caller",
                    "prefix_run": self.run.prefix_run,
                    "family": str(names[row]) or None,
                    "candidates": list(candidates.values()),
                }
            )
