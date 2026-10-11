"""Ordered accepted writes over point and library single-file transactions."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from math import isfinite
from typing import TYPE_CHECKING, Protocol
from uuid import uuid4

from pydantic import TypeAdapter

from zcu_tools.cfg_model import (
    ConfigBase,
    ModuleSpec,
    WaveformSpec,
    module_spec_from_raw,
    waveform_spec_from_raw,
)

from .ledger import RecordsLedger
from .ledger_models import (
    AcceptedPayload,
    AcceptedWrite,
    JsonObject,
    JsonValue,
    LedgerEvent,
    Origin,
    SourceReference,
)
from .points import PointView
from .provenance import Provenance
from .storage import PointStorageBinding, bind_sample_table

if TYPE_CHECKING:
    from zcu_tools.resources.context.library import ModuleLibrary


@dataclass(frozen=True)
class ParameterWriteback:
    """One concrete dotted point path, working-unit JSON value and uncertainty.

    path has its role prefix already resolved by the caller. value may also be
    native complex, encoded by AcceptedWrite. stderr is None for unknown or a
    finite non-negative working-unit number; it is never converted to SI.
    """

    path: str
    value: JsonValue | complex
    stderr: float | None = None


@dataclass(frozen=True)
class ModuleWriteback:
    """One concrete library key and persistent module Spec (not a raw mapping).

    path follows the library owner's module key contract, with roles already
    resolved. value retains inline Specs and concrete child references. stderr
    is unknown when None, otherwise a finite nonnegative working-unit uncertainty
    attached to every accepted leaf source; no unit conversion occurs.
    """

    path: str
    value: ModuleSpec
    stderr: float | None = None


@dataclass(frozen=True)
class WaveformWriteback:
    """One concrete library waveform key and persistent Spec, never raw mapping.

    path follows the library owner's waveform key contract, with roles already
    resolved. value retains inline Specs and concrete child references. stderr
    is unknown when None, otherwise a finite nonnegative working-unit uncertainty
    attached to every accepted leaf source; no unit conversion occurs.
    """

    path: str
    value: WaveformSpec
    stderr: float | None = None


@dataclass(frozen=True)
class SampleWriteback:
    """One component's producer-defined JSON fields for an accepted sample.

    component is a concrete safe single path segment, not a role alias. value
    contains measurement fields, including optional device values and units;
    none are inferred or converted. run_id, point, at and source are supplied
    by the executor and override those keys if present in value.
    """

    component: str
    value: JsonObject


type WritebackItem = (
    ParameterWriteback | ModuleWriteback | WaveformWriteback | SampleWriteback
)


@dataclass(frozen=True)
class WritebackEvidence:
    """Caller-captured source facts shared by all writes in one acceptance.

    kind is the nonempty captured experiment name. origin identifies gui,
    notebook or mcp. run_id is the captured acquisition identity, or None when
    native acquisition evidence is unavailable. source is an optional ledger
    reference copied into accepted; ordinary single-run acceptance uses None.
    """

    kind: str
    origin: Origin
    run_id: str | None = None
    source: SourceReference | None = None


def _require_named_spec(value: object) -> None:
    if not isinstance(value, ConfigBase):
        raise TypeError("library writeback requires a named Spec, not a raw mapping")


def _accepted_write(item: WritebackItem) -> AcceptedWrite:
    if not isinstance(item, SampleWriteback) and item.stderr is not None:
        stderr = TypeAdapter(float).validate_python(item.stderr, strict=True)
        if not isfinite(stderr) or stderr < 0:
            raise ValueError("stderr must be finite and non-negative, or None")
    if isinstance(item, ParameterWriteback):
        return AcceptedWrite.model_validate({"path": item.path, "value": item.value})
    if isinstance(item, SampleWriteback):
        return AcceptedWrite(path=item.component, value=item.value)
    _require_named_spec(item.value)
    if isinstance(item, ModuleWriteback):
        value = module_spec_from_raw(item.value).to_dict()
    else:
        value = waveform_spec_from_raw(item.value).to_dict()
    return AcceptedWrite(path=item.path, value=value)


class WritebackApplyError(RuntimeError):
    """An accepted item failed; prior items and the accepted event remain.

    event_id identifies the already-appended accepted event. applied counts the
    successfully committed prefix, not attempted items. __cause__ is the original
    item failure. Ledger append failures occur before this error boundary and
    propagate unchanged. No rollback, automatic retry or recovery is provided.
    """

    def __init__(self, *, event_id: str, applied: int) -> None:
        self.event_id = event_id
        self.applied = applied
        super().__init__(f"Writeback {event_id} failed after {applied} applied items")


class WritebackTarget(Protocol):
    """Bound single-item write capabilities for a saved point.

    entry_id/ledger identify the owning entry; point is its saved label.
    Each setter commits and validates one item or raises without installing it.
    Publication belongs to the context owner, not this protocol or the executor.
    append_sample receives producer fields separately from accepted metadata.
    """

    entry_id: str
    ledger: RecordsLedger
    point: str

    def set_parameter(
        self, path: str, value: JsonValue, source: Provenance
    ) -> None: ...
    def set_module(self, path: str, value: ModuleSpec, source: Provenance) -> None: ...
    def set_waveform(
        self, path: str, value: WaveformSpec, source: Provenance
    ) -> None: ...
    def append_sample(
        self,
        component: str,
        fields: JsonObject,
        *,
        run_id: str,
        at: datetime,
        source: str,
    ) -> None: ...


class PointWritebackTarget:
    """Bind a captured PointView/storage for Notebook or other persistent callers.

    Construction does no I/O. The caller supplies matching captured capabilities.
    Library opening is lazy; owner key/family/source/conflict errors propagate.
    Each setter commits one file; sample rows append through the sample owner.
    """

    def __init__(self, point: PointView, storage: PointStorageBinding) -> None:
        self._point = point
        self._storage = storage
        self.entry_id = str(storage.entry_id)
        self.ledger = storage.ledger
        self.point = storage.point
        self._library_handle: ModuleLibrary | None = None

    def set_parameter(self, path: str, value: JsonValue, source: Provenance) -> None:
        """Commit one concrete point path with already-built provenance."""
        with self._point.edit() as draft:
            draft.set(path, value, provenance=source)

    def _library(self) -> ModuleLibrary:
        # Library imports entry provenance; defer to avoid the package cycle.
        from zcu_tools.resources.context.library import ModuleLibrary

        if self._library_handle is None:
            self._library_handle = ModuleLibrary(
                self._storage.result_path
                / "points"
                / self._storage.point
                / "module_cfg.yaml",
                ledger=self._storage.result_path / "records" / "ledger.jsonl",
                entry_id=self.entry_id,
            )
        return self._library_handle

    def set_module(self, path: str, value: ModuleSpec, source: Provenance) -> None:
        """Commit one concrete library module root; owner validates its family."""
        with self._library().edit() as draft:
            draft.set_module(path, value, provenance=source)

    def set_waveform(self, path: str, value: WaveformSpec, source: Provenance) -> None:
        """Commit one concrete waveform root through the library owner."""
        with self._library().edit() as draft:
            draft.set_waveform(path, value, provenance=source)

    def append_sample(
        self,
        component: str,
        fields: JsonObject,
        *,
        run_id: str,
        at: datetime,
        source: str,
    ) -> None:
        """Append producer fields with the executor's accepted identity/time/source."""
        bind_sample_table(self._storage.result_path, component).append(
            run_id=run_id, point=self.point, at=at, source=source, **fields
        )


def apply_writeback(
    items: tuple[WritebackItem, ...],
    *,
    target: WritebackTarget,
    evidence: WritebackEvidence,
) -> str:
    """Validate concrete values, append accepted, then commit in supplied order.

    target binds the destination and its entry ledger; callers resolve roles
    before this entry. Nonempty items, strict JSON/Spec values, finite nonnegative
    stderr, kind and sample run_id are validated before append. Target owners
    validate paths, schema, sources and transactions when each item is reached.
    Samples receive run_id, saved point, UTC at and accepted source; producer
    device fields are preserved, not inferred. Return the event ID on success.

    Ledger append failures propagate unchanged, with no writes. An item failure
    raises WritebackApplyError(event_id, applied) from its original cause, stops
    later items and retains the successful prefix and event. No per-item progress
    callback, cross-file rollback or automatic retry is provided.
    """
    if not items:
        raise ValueError("writeback items must be nonempty")
    kind = TypeAdapter(str).validate_python(evidence.kind, strict=True)
    if not kind.strip():
        raise ValueError("writeback kind must be nonempty")
    if any(isinstance(item, SampleWriteback) for item in items) and (
        evidence.run_id is None or not evidence.run_id.strip()
    ):
        raise ValueError("sample writeback requires a nonempty run_id")
    writes = tuple(_accepted_write(item) for item in items)
    event = LedgerEvent(
        id=str(uuid4()),
        kind="accepted",
        at=datetime.now(UTC).isoformat(),
        entry_id=target.entry_id,
        origin=evidence.origin,
        payload=AcceptedPayload(
            writes=writes, run_id=evidence.run_id, source=evidence.source
        ),
    )
    target.ledger.append(event)
    for applied, (item, write) in enumerate(zip(items, writes, strict=True)):
        provenance = Provenance(
            source=event.id,
            kind=kind,
            run_id=evidence.run_id,
            at=event.at,
            stderr=None if isinstance(item, SampleWriteback) else item.stderr,
        )
        try:
            if isinstance(item, SampleWriteback):
                if evidence.run_id is None:
                    raise ValueError("sample writeback requires a run_id")
                fields = {
                    key: value
                    for key, value in item.value.items()
                    if key not in {"run_id", "point", "at", "source"}
                }
                target.append_sample(
                    item.component,
                    fields,
                    run_id=evidence.run_id,
                    at=datetime.fromisoformat(event.at),
                    source=event.id,
                )
            elif isinstance(item, ParameterWriteback):
                target.set_parameter(write.path, write.value, provenance)
            elif isinstance(item, ModuleWriteback):
                target.set_module(item.path, item.value, provenance)
            else:
                target.set_waveform(item.path, item.value, provenance)
        except Exception as exc:
            raise WritebackApplyError(event_id=event.id, applied=applied) from exc
    return event.id
