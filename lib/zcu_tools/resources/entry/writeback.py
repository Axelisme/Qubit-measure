"""Ordered accepted writes over point and library single-file transactions."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
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
from zcu_tools.core.numeric_scalar import normalize_numpy_scalar
from zcu_tools.format_version import YamlMap, YamlValue

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
from .storage import PointStorageBinding, bind_record_table
from .views import document_value, snapshot

if TYPE_CHECKING:
    from zcu_tools.resources.context.library import ModuleLibrary


@dataclass(frozen=True)
class ParameterWriteback:
    """One concrete dotted point path, working-unit JSON value and caller metadata.

    path has its role prefix already resolved by the caller. value may also be
    native complex, encoded by AcceptedWrite. metadata is None to keep the
    path's existing opaque metadata, or a string-keyed YAML mapping that
    replaces it in the same point commit ({} clears it). The executor neither
    interprets its keys nor adds source, time or event fields.
    """

    path: str
    value: JsonValue | complex
    metadata: YamlMap | None = None


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
class RecordWriteback:
    """One component's producer-defined JSON fields for an accepted record.

    component is a concrete safe single path segment, not a role alias. value
    is a dict of measurement fields, including optional device values and units.
    The executor recursively converts numpy floating/integer scalars in dicts
    and lists to Python float/int before strict JSON validation. Float rounding
    follows Python float; integers retain their exact value. Non-finite results
    and other non-JSON inputs are rejected before any write. Units are not
    converted and the producer tree is not mutated. run_id, point, at and source
    are supplied by the executor and override those keys if present in value.
    """

    component: str
    value: Mapping[str, object]


type WritebackItem = (
    ParameterWriteback | ModuleWriteback | WaveformWriteback | RecordWriteback
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


def _normalize_record_value(value: object) -> object:
    """Copy JSON containers, projecting numpy numbers without coercing other values."""
    if isinstance(value, dict):
        return {key: _normalize_record_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_record_value(item) for item in value]
    return normalize_numpy_scalar(value)


def _accepted_write(item: WritebackItem) -> AcceptedWrite:
    if isinstance(item, ParameterWriteback) and item.metadata is not None:
        TypeAdapter(YamlMap).validate_python(item.metadata, strict=True)
    if (
        isinstance(item, (ModuleWriteback, WaveformWriteback))
        and item.stderr is not None
    ):
        stderr = TypeAdapter(float).validate_python(item.stderr, strict=True)
        if not isfinite(stderr) or stderr < 0:
            raise ValueError("stderr must be finite and non-negative, or None")
    if isinstance(item, ParameterWriteback):
        return AcceptedWrite.model_validate({"path": item.path, "value": item.value})
    if isinstance(item, RecordWriteback):
        fields = TypeAdapter(JsonObject).validate_python(
            _normalize_record_value(item.value)
        )
        return AcceptedWrite(path=item.component, value=fields)
    _require_named_spec(item.value)
    if isinstance(item, ModuleWriteback):
        value = module_spec_from_raw(item.value).to_dict()
    else:
        value = waveform_spec_from_raw(item.value).to_dict()
    return AcceptedWrite(path=item.path, value=value)


@dataclass(frozen=True)
class ParameterUpdate:
    """One concrete point path/value/metadata command in a single-file batch.

    path is dotted and role-resolved. value uses the point owner's YAML value
    representation. metadata=None preserves existing metadata; a mapping
    replaces it, including {} to clear it. The point owner validates each draft
    change and commits the entire batch only when every change succeeds.
    """

    path: str
    value: YamlValue
    metadata: YamlMap | None = None


@dataclass(frozen=True)
class LibraryUpdate:
    """One module or waveform command and its optional source in a library batch.

    item supplies the concrete key, named Spec and family. source=None requests
    manual provenance; otherwise the library owner validates the explicit source
    against its ledger. item.stderr is not interpreted here: accepted executors
    already include it in source. All commands share one library transaction.
    """

    item: ModuleWriteback | WaveformWriteback
    source: Provenance | None = None


@dataclass(frozen=True)
class CommittedWrite:
    """One confirmed item at its concrete path, detached at commit time.

    index is this item's zero-based position in the supplied apply_writeback
    tuple, not its commit position. path is a point path, library key or record
    component and need not be unique. value is the owner's actual working-unit
    value, including record metadata.
    It never borrows a proposal or a later destination snapshot.
    """

    index: int
    path: str
    value: YamlValue


@dataclass(frozen=True)
class WritebackResult:
    """One accepted event and detached items in document commit order.

    event_id identifies the single accepted ledger event. committed contains
    point items, then library items, then records, preserving supplied order
    within each group. On failure it includes only fully committed groups (each
    record is a group). Each item's index identifies its supplied position.
    """

    event_id: str
    committed: tuple[CommittedWrite, ...]


class WritebackApplyError(RuntimeError):
    """An accepted group failed; prior groups and the accepted event remain.

    event_id identifies the already-appended accepted event. applied counts the
    items in fully committed groups, not groups or attempted items. result retains the same
    event ID and each detached committed value. __cause__ is the original
    item failure. Ledger append failures occur before this error boundary and
    propagate unchanged. No rollback, automatic retry or recovery is provided.
    """

    def __init__(self, result: WritebackResult) -> None:
        self.result = result
        self.event_id = result.event_id
        self.applied = len(result.committed)
        super().__init__(
            f"Writeback {self.event_id} failed after {self.applied} applied items"
        )


class WritebackTarget(Protocol):
    """Bound document-batch write capabilities for a saved point.

    entry_id/ledger identify the owning entry; point is its saved label.
    set_parameters/set_modules each validate an ordered nonempty batch in one
    draft and commit one file, or raise without committing any batch item.
    They return one detached actual value per command, captured immediately
    after that command's validation so later overwrites do not alter receipts.
    Values are returned only after the entire batch commits. ParameterUpdate
    and LibraryUpdate describe value, metadata and source contracts.
    Publication belongs to the context owner, not this protocol or executor. append_record accepts a
    safe component segment, producer fields and separate accepted run_id/UTC
    at/event source; it records this target's point label through the record owner.
    """

    entry_id: str
    ledger: RecordsLedger
    point: str

    def set_parameters(
        self, updates: tuple[ParameterUpdate, ...]
    ) -> tuple[YamlValue, ...]: ...
    def set_modules(
        self, updates: tuple[LibraryUpdate, ...]
    ) -> tuple[YamlMap, ...]: ...
    def append_record(
        self,
        component: str,
        fields: JsonObject,
        *,
        run_id: str,
        at: datetime,
        source: str,
    ) -> JsonObject: ...


def commit_parameter_updates(
    point: PointView, updates: tuple[ParameterUpdate, ...]
) -> tuple[YamlValue, ...]:
    """Commit ordered updates through an already-bound point's single edit.

    point is the caller's existing PointView; this function does not reopen it
    or select a GUI target. updates must be nonempty. ParameterUpdate defines
    concrete paths, YAML values and optional metadata replacement.
    Return one detached actual value per update in supplied order, captured
    immediately after its draft validation, only after the whole edit commits.
    Later updates cannot change earlier receipts. Empty input raises ValueError.
    Path, schema, metadata, conflict and I/O errors propagate without committing
    any update in this batch. No GUI publication or accepted event is produced.
    """
    if not updates:
        raise ValueError("parameter updates must be nonempty")
    actual: list[YamlValue] = []
    with point.edit() as draft:
        for update in updates:
            draft.set(update.path, update.value)
            if update.metadata is not None:
                draft.set_meta(update.path, update.metadata)
            actual.append(document_value(snapshot(draft), update.path))
    return tuple(actual)


def commit_library_updates(
    library: ModuleLibrary, updates: tuple[LibraryUpdate, ...]
) -> tuple[YamlMap, ...]:
    """Commit ordered module/waveform updates through one existing library edit.

    library is the caller's already-bound ModuleLibrary; it is not reopened.
    updates must be nonempty. LibraryUpdate defines each concrete key, named
    Spec family and source; the library validates sources against its ledger.
    Return one detached actual Spec mapping per update in supplied order,
    captured after that draft setter, only after the whole edit commits.
    Later updates cannot change earlier receipts. Empty input raises ValueError.
    Key, family, schema, source, conflict and I/O errors propagate without
    committing batch updates. The library's edit-time disk cleanup policy still
    applies; completed cleanup is not rolled back. No GUI publication or
    accepted event is produced.
    """
    if not updates:
        raise ValueError("library updates must be nonempty")
    actual: list[YamlMap] = []
    with library.edit() as draft:
        for update in updates:
            item = update.item
            if isinstance(item, ModuleWriteback):
                draft.set_module(item.path, item.value, provenance=update.source)
                kind = "module"
            else:
                draft.set_waveform(item.path, item.value, provenance=update.source)
                kind = "waveform"
            actual.append(draft.snapshot().entry_value(item.path, kind=kind))
    return tuple(actual)


class PointWritebackTarget:
    """Bind a captured PointView/storage for Notebook or other persistent callers.

    Construction does no I/O. The caller supplies matching captured capabilities.
    Library opening is lazy; owner key/family/source/conflict errors propagate.
    Each setter commits one file; record rows append through the record owner.
    """

    def __init__(self, point: PointView, storage: PointStorageBinding) -> None:
        self._point = point
        self._storage = storage
        self.entry_id = str(storage.entry_id)
        self.ledger = storage.ledger
        self.point = storage.point
        self._library_handle: ModuleLibrary | None = None

    def set_parameter(
        self, path: str, value: JsonValue, metadata: YamlMap | None
    ) -> YamlValue:
        """Commit one value/metadata pair; return its detached actual value."""
        return self.set_parameters((ParameterUpdate(path, value, metadata),))[0]

    def set_parameters(
        self, updates: tuple[ParameterUpdate, ...]
    ) -> tuple[YamlValue, ...]:
        """Commit one nonempty point batch under WritebackTarget's batch contract."""
        return commit_parameter_updates(self._point, updates)

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

    def set_module(self, path: str, value: ModuleSpec, source: Provenance) -> YamlMap:
        """Commit one module and return its actual cached Spec mapping."""
        return self.set_modules((LibraryUpdate(ModuleWriteback(path, value), source),))[
            0
        ]

    def set_waveform(
        self, path: str, value: WaveformSpec, source: Provenance
    ) -> YamlMap:
        """Commit one waveform and return its actual cached Spec mapping."""
        return self.set_modules(
            (LibraryUpdate(WaveformWriteback(path, value), source),)
        )[0]

    def set_modules(self, updates: tuple[LibraryUpdate, ...]) -> tuple[YamlMap, ...]:
        """Commit one nonempty mixed module/waveform batch, or leave it unwritten."""
        if not updates:
            raise ValueError("library updates must be nonempty")
        return commit_library_updates(self._library(), updates)

    def append_record(
        self,
        component: str,
        fields: JsonObject,
        *,
        run_id: str,
        at: datetime,
        source: str,
    ) -> JsonObject:
        """Append and return the record owner's complete committed row."""
        return bind_record_table(self._storage.result_path, component).append(
            run_id=run_id, point=self.point, at=at, source=source, **fields
        )


def _append_accepted_record(
    target: WritebackTarget,
    write: AcceptedWrite,
    evidence: WritebackEvidence,
    event: LedgerEvent,
) -> JsonObject:
    """Pass normalized producer fields and accepted coordinates to the record owner."""
    if evidence.run_id is None:
        raise ValueError("record writeback requires a run_id")
    if not isinstance(write.value, dict):
        raise TypeError("accepted record fields must be a JSON object")
    fields = {
        key: value
        for key, value in write.value.items()
        if key not in {"run_id", "point", "at", "source"}
    }
    return target.append_record(
        write.path,
        fields,
        run_id=evidence.run_id,
        at=datetime.fromisoformat(event.at),
        source=event.id,
    )


def apply_writeback(
    items: tuple[WritebackItem, ...],
    *,
    target: WritebackTarget,
    evidence: WritebackEvidence,
) -> WritebackResult:
    """Validate values, append accepted, then commit point/library/record groups.

    target binds the destination and its entry ledger; callers resolve roles
    before this entry. Nonempty items, strict JSON/Spec values, parameter
    metadata representation, finite nonnegative library stderr, kind and record
    run_id are validated before append. Record dict/list values first normalize
    numpy floating/integer scalars to Python numbers without mutating producers;
    non-finite results and other non-JSON values remain invalid. Target owners
    validate paths, schema, sources and transactions within each group draft. Parameters commit the
    caller's metadata unchanged, or keep existing metadata when it is None; no
    event, source or time is added to it. Library items receive the accepted
    Provenance. Records receive run_id, saved point, UTC at and accepted source;
    producer device fields are preserved, not inferred. Return WritebackResult
    with the accepted event ID and each owner's confirmed committed value.

    Accepted payload writes retain supplied order. Committed results instead
    follow point group, library group, then individual records; supplied order
    is preserved within each group. Each CommittedWrite.index maps back to the
    supplied tuple even when paths repeat. Point and library each use one edit;
    a failing item leaves its entire group unwritten. Receipts capture each
    validated draft value and are released only after that group commits.

    Ledger append failures propagate unchanged, with no writes. A group failure
    raises WritebackApplyError(result) from its original cause, stops later
    groups and retains fully committed groups and the event. applied counts
    their items, not groups. No cross-file rollback or automatic retry occurs.
    """
    if not items:
        raise ValueError("writeback items must be nonempty")
    kind = TypeAdapter(str).validate_python(evidence.kind, strict=True)
    if not kind.strip():
        raise ValueError("writeback kind must be nonempty")
    if any(isinstance(item, RecordWriteback) for item in items) and (
        evidence.run_id is None or not evidence.run_id.strip()
    ):
        raise ValueError("record writeback requires a nonempty run_id")
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
    parameters: list[ParameterUpdate] = []
    parameter_indices: list[int] = []
    library: list[LibraryUpdate] = []
    library_indices: list[int] = []
    record_indices: list[int] = []
    for index, (item, write) in enumerate(zip(items, writes, strict=True)):
        if isinstance(item, ParameterWriteback):
            parameters.append(ParameterUpdate(write.path, write.value, item.metadata))
            parameter_indices.append(index)
        elif isinstance(item, RecordWriteback):
            record_indices.append(index)
        else:
            library.append(
                LibraryUpdate(
                    item,
                    Provenance(
                        source=event.id,
                        kind=kind,
                        run_id=evidence.run_id,
                        at=event.at,
                        stderr=item.stderr,
                    ),
                )
            )
            library_indices.append(index)

    committed: list[CommittedWrite] = []
    try:
        if parameters:
            values = target.set_parameters(tuple(parameters))
            committed.extend(
                CommittedWrite(index, writes[index].path, deepcopy(value))
                for index, value in zip(parameter_indices, values, strict=True)
            )
        if library:
            library_values = target.set_modules(tuple(library))
            committed.extend(
                CommittedWrite(index, writes[index].path, deepcopy(value))
                for index, value in zip(library_indices, library_values, strict=True)
            )
        for index in record_indices:
            write = writes[index]
            value = _append_accepted_record(target, write, evidence, event)
            committed.append(CommittedWrite(index, write.path, deepcopy(value)))
    except Exception as exc:
        raise WritebackApplyError(WritebackResult(event.id, tuple(committed))) from exc
    return WritebackResult(event.id, tuple(committed))
