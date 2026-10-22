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
from .storage import PointStorageBinding, bind_sample_table
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
class SampleWriteback:
    """One component's producer-defined JSON fields for an accepted sample.

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


def _normalize_sample_value(value: object) -> object:
    """Copy JSON containers, projecting numpy numbers without coercing other values."""
    if isinstance(value, dict):
        return {key: _normalize_sample_value(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_normalize_sample_value(item) for item in value]
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
    if isinstance(item, SampleWriteback):
        fields = TypeAdapter(JsonObject).validate_python(
            _normalize_sample_value(item.value)
        )
        return AcceptedWrite(path=item.component, value=fields)
    _require_named_spec(item.value)
    if isinstance(item, ModuleWriteback):
        value = module_spec_from_raw(item.value).to_dict()
    else:
        value = waveform_spec_from_raw(item.value).to_dict()
    return AcceptedWrite(path=item.path, value=value)


@dataclass(frozen=True)
class CommittedWrite:
    """One confirmed item at its concrete path, detached at commit time.

    path is a point path, library key or sample component, in submission order.
    value is the owner's actual working-unit value, including sample metadata.
    It never borrows a proposal or a later destination snapshot.
    """

    path: str
    value: YamlValue


@dataclass(frozen=True)
class WritebackResult:
    """One accepted event and its ordered, detached successful item prefix.

    event_id identifies the single accepted ledger event. committed contains
    every item on success, or only completed items on WritebackApplyError.
    """

    event_id: str
    committed: tuple[CommittedWrite, ...]


class WritebackApplyError(RuntimeError):
    """An accepted item failed; prior items and the accepted event remain.

    event_id identifies the already-appended accepted event. applied counts the
    successfully committed prefix, not attempted items. result retains the same
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
    """Bound single-item write capabilities for a saved point.

    entry_id/ledger identify the owning entry; point is its saved label.
    Each setter commits/validates one item and returns its detached actual value,
    formed by that write owner without reopening the destination, or raises.
    set_parameter accepts a concrete dotted point path, a working-unit JSON
    value and caller metadata: None keeps the path's metadata, a mapping
    replaces it in the same commit. set_module/set_waveform accept concrete
    library keys and named Specs; source is the accepted library Provenance for
    that item. Publication belongs to the
    context owner, not this protocol or the executor. append_sample accepts a
    safe component segment, producer fields and separate accepted run_id/UTC
    at/event source; it records this target's point label through the sample owner.
    """

    entry_id: str
    ledger: RecordsLedger
    point: str

    def set_parameter(
        self, path: str, value: JsonValue, metadata: YamlMap | None
    ) -> YamlValue: ...
    def set_module(
        self, path: str, value: ModuleSpec, source: Provenance
    ) -> YamlMap: ...
    def set_waveform(
        self, path: str, value: WaveformSpec, source: Provenance
    ) -> YamlMap: ...
    def append_sample(
        self,
        component: str,
        fields: JsonObject,
        *,
        run_id: str,
        at: datetime,
        source: str,
    ) -> JsonObject: ...


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

    def set_parameter(
        self, path: str, value: JsonValue, metadata: YamlMap | None
    ) -> YamlValue:
        """Commit value/metadata and return the actual cached value, without I/O."""
        with self._point.edit() as draft:
            draft.set(path, value)
            if metadata is not None:
                draft.set_meta(path, metadata)
        return document_value(snapshot(self._point), path)

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
        library = self._library()
        with library.edit() as draft:
            draft.set_module(path, value, provenance=source)
        return library.snapshot().entry_value(path, kind="module")

    def set_waveform(
        self, path: str, value: WaveformSpec, source: Provenance
    ) -> YamlMap:
        """Commit one waveform and return its actual cached Spec mapping."""
        library = self._library()
        with library.edit() as draft:
            draft.set_waveform(path, value, provenance=source)
        return library.snapshot().entry_value(path, kind="waveform")

    def append_sample(
        self,
        component: str,
        fields: JsonObject,
        *,
        run_id: str,
        at: datetime,
        source: str,
    ) -> JsonObject:
        """Append and return the sample owner's complete committed row."""
        return bind_sample_table(self._storage.result_path, component).append(
            run_id=run_id, point=self.point, at=at, source=source, **fields
        )


def apply_writeback(
    items: tuple[WritebackItem, ...],
    *,
    target: WritebackTarget,
    evidence: WritebackEvidence,
) -> WritebackResult:
    """Validate concrete values, append accepted, then commit in supplied order.

    target binds the destination and its entry ledger; callers resolve roles
    before this entry. Nonempty items, strict JSON/Spec values, parameter
    metadata representation, finite nonnegative library stderr, kind and sample
    run_id are validated before append. Sample dict/list values first normalize
    numpy floating/integer scalars to Python numbers without mutating producers;
    non-finite results and other non-JSON values remain invalid. Target owners
    validate paths, schema, sources and transactions when each item is reached. Parameters commit the
    caller's metadata unchanged, or keep existing metadata when it is None; no
    event, source or time is added to it. Library items receive the accepted
    Provenance. Samples receive run_id, saved point, UTC at and accepted source;
    producer device fields are preserved, not inferred. Return WritebackResult
    with the accepted event ID and each owner's confirmed committed value.

    Ledger append failures propagate unchanged, with no writes. An item failure
    raises WritebackApplyError(result) from its original cause, stops
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
    committed: list[CommittedWrite] = []
    for item, write in zip(items, writes, strict=True):
        try:
            if isinstance(item, SampleWriteback):
                if evidence.run_id is None:
                    raise ValueError("sample writeback requires a run_id")
                if not isinstance(write.value, dict):
                    raise TypeError("accepted sample fields must be a JSON object")
                fields = {
                    key: value
                    for key, value in write.value.items()
                    if key not in {"run_id", "point", "at", "source"}
                }
                value = target.append_sample(
                    item.component,
                    fields,
                    run_id=evidence.run_id,
                    at=datetime.fromisoformat(event.at),
                    source=event.id,
                )
            elif isinstance(item, ParameterWriteback):
                value = target.set_parameter(write.path, write.value, item.metadata)
            else:
                provenance = Provenance(
                    source=event.id,
                    kind=kind,
                    run_id=evidence.run_id,
                    at=event.at,
                    stderr=item.stderr,
                )
                if isinstance(item, ModuleWriteback):
                    value = target.set_module(item.path, item.value, provenance)
                else:
                    value = target.set_waveform(item.path, item.value, provenance)
        except Exception as exc:
            raise WritebackApplyError(
                WritebackResult(event.id, tuple(committed))
            ) from exc
        committed.append(CommittedWrite(write.path, deepcopy(value)))
    return WritebackResult(event.id, tuple(committed))
