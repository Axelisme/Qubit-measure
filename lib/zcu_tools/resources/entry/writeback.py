"""Ordered accepted writes to arb assets, point parameters, library and records."""

from __future__ import annotations

from collections.abc import Iterator, Mapping
from copy import deepcopy
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
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
from zcu_tools.resources.document_names import LIBRARY_FILENAME, PARAMETERS_FILENAME
from zcu_tools.resources.waveform_assets import ArbWaveformDatabase

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
from .parameter_view import ParameterView
from .storage import StorageBinding, bind_record_table
from .views import (
    document_value,
    parameter_origin,
    snapshot,
    source_path,
)

if TYPE_CHECKING:
    from zcu_tools.core.waveform_format import WaveformSamples
    from zcu_tools.resources.context.library import ModuleLibrary


@dataclass(frozen=True)
class ParameterWriteback:
    """One concrete dotted point path, working-unit JSON value and caller metadata.

    path has its role prefix already resolved by the caller. value may also be
    native complex, encoded by AcceptedWrite. metadata is None to keep the
    path's existing opaque metadata, or a string-keyed YAML mapping that
    replaces it in the same point commit ({} clears it). The executor neither
    interprets its keys nor adds source, time or event fields. create_kind is
    the registered kind used only if the component is missing; None does not
    create it. Only the proposed field seeds creation, with model defaults for
    other fields. Existing components retain their kind and other fields.
    """

    path: str
    value: JsonValue | complex
    metadata: YamlMap | None = None
    create_kind: str | None = None


@dataclass(frozen=True)
class ModuleWriteback:
    """One concrete library key and persistent module Spec (not a raw mapping).

    path follows the library owner's module key contract, with roles already
    resolved. value retains inline Specs and concrete child references.
    """

    path: str
    value: ModuleSpec


@dataclass(frozen=True)
class WaveformWriteback:
    """One concrete library waveform key and persistent Spec, never raw mapping.

    path follows the library owner's waveform key contract, with roles already
    resolved. value retains inline Specs and concrete child references.
    """

    path: str
    value: WaveformSpec


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


@dataclass(frozen=True)
class ArbAssetWriteback:
    """Publish WaveformSamples at a validated entry-scoped arb asset key.

    key matches ^[A-Za-z][A-Za-z0-9_]*$. samples hold the reference time in
    microseconds and normalized I/Q. Publication never replaces an asset.
    """

    key: str
    samples: WaveformSamples


type WritebackItem = (
    ParameterWriteback
    | ModuleWriteback
    | WaveformWriteback
    | RecordWriteback
    | ArbAssetWriteback
)


@dataclass(frozen=True)
class WritebackEvidence:
    """Caller-captured source facts shared by all writes in one acceptance.

    origin identifies gui, notebook or mcp. run_id is the captured acquisition
    identity, or None when native acquisition evidence is unavailable.
    source is an optional ledger
    reference copied into accepted; ordinary single-run acceptance uses None.
    """

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


def _asset_summary(item: ArbAssetWriteback) -> YamlMap:
    return {
        "sample_count": int(item.samples.time.size),
        "duration_us": float(item.samples.time[-1]),
    }


def _accepted_write(item: WritebackItem) -> AcceptedWrite:
    if isinstance(item, ArbAssetWriteback):
        return AcceptedWrite(
            path=f"arb_waveforms/{item.key}", value=_asset_summary(item)
        )
    if isinstance(item, ParameterWriteback) and item.metadata is not None:
        TypeAdapter(YamlMap).validate_python(item.metadata, strict=True)
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
    replaces it, including {} to clear it. create_kind optionally names the
    registered kind to create when the component is absent; None keeps missing
    component errors. Only this update's fields seed creation; all other fields
    use registered model defaults. Existing components keep their kind and other
    fields. Values use working units. The point owner validates each draft
    change and commits the entire batch only when every change succeeds.
    """

    path: str
    value: YamlValue
    metadata: YamlMap | None = None
    create_kind: str | None = None


@dataclass(frozen=True)
class LibraryUpdate:
    """One library command; item supplies its concrete key, named Spec and family."""

    item: ModuleWriteback | WaveformWriteback


@dataclass(frozen=True)
class CommittedWrite:
    """One confirmed item at its concrete path, detached at commit time.

    index is the item's zero-based position in the supplied apply_writeback
    tuple, not its commit position. path is an entry-relative arb asset key
    (arb_waveforms/<key>), a point path, library key or record component.
    value is the owner's actual working-unit value, including record metadata,
    or the asset's status (created/existing), sample count and duration in us.
    It never borrows a proposal or a later destination snapshot.
    """

    index: int
    path: str
    value: YamlValue


@dataclass(frozen=True)
class WritebackResult:
    """One accepted event and detached items in document commit order.

    event_id identifies the single accepted ledger event. committed contains
    arb assets, point items, library items, then records, in supplied order
    within each group. On failure it includes committed groups and any asset
    prefix confirmed before an I/O error. Each index identifies the supplied position.
    """

    event_id: str
    committed: tuple[CommittedWrite, ...]


class WritebackApplyError(RuntimeError):
    """An accepted group failed; prior groups and the accepted event remain.

    event_id identifies the already-appended accepted event. applied counts the
    confirmed items in result, including an asset prefix after an I/O error.
    result retains the same event ID and detached values. __cause__ is the original
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
    """Bound document-batch write capabilities for entry or point documents.

    ledger identifies the owning entry; point is its label or None for entry scope.
    set_parameters/set_modules each validate an ordered nonempty batch in one
    draft and commit one file, or raise without committing any batch item.
    They return one detached actual value per command, captured immediately
    after that command's validation so later overwrites do not alter receipts.
    Values are returned only after the entire batch commits. ParameterUpdate
    and LibraryUpdate describe value and metadata contracts.
    Publication belongs to the context owner, not this protocol or executor.
    append_record accepts a safe component segment, producer fields and separate accepted run_id/UTC
    at/event source; it records this target's point label through the record owner.
    """

    ledger: RecordsLedger

    @property
    def point(self) -> str | None:
        """Return the bound point label, or None for entry-level writes."""
        ...

    def publish_arb_assets(
        self, items: tuple[ArbAssetWriteback, ...]
    ) -> Iterator[YamlMap]:
        """Iterate to publish nonempty, distinct keys to the bound entry's asset root.

        Validate the group before writing. Yield each confirmed status and
        summary in order. A preflight conflict yields nothing; an I/O error
        stops iteration after any confirmed prefix. Never replace assets.
        """
        ...

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
    view: ParameterView, updates: tuple[ParameterUpdate, ...]
) -> tuple[YamlValue, ...]:
    """Commit ordered updates through a bound entry/point view's single edit.

    view is the caller's existing ParameterView; do not reopen it
    or select a GUI target. updates must be nonempty. ParameterUpdate defines
    concrete paths, working-unit YAML values, optional metadata replacement
    and opt-in creation with registered model defaults (ADR-0008).
    Return one detached actual value per update in supplied order, captured
    immediately after its draft validation, only after the whole edit commits.
    Later updates cannot change earlier receipts. Empty input raises ValueError.
    Path, schema, metadata, conflict and I/O errors propagate without committing
    any update in this batch. No GUI publication or accepted event is produced.
    """
    if not updates:
        raise ValueError("parameter updates must be nonempty")
    actual: list[YamlValue] = []
    with view.edit() as draft:
        for update in updates:
            if update.create_kind is not None:
                name, *segments = update.path.split(".")
                if not hasattr(draft, name):
                    fields: YamlMap = {segments[-1]: update.value}
                    for segment in reversed(segments[:-1]):
                        fields = {segment: fields}
                    draft.add_component(name, kind=update.create_kind, **fields)
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
    Spec and family; the library validates them without consulting a ledger.
    Return one detached actual Spec mapping per update in supplied order,
    captured after that draft setter, only after the whole edit commits.
    Later updates cannot change earlier receipts. Empty input raises ValueError.
    Key, family, schema, conflict and I/O errors propagate without
    committing batch updates. No GUI publication or accepted event is produced.
    """
    if not updates:
        raise ValueError("library updates must be nonempty")
    actual: list[YamlMap] = []
    with library.edit() as draft:
        for update in updates:
            item = update.item
            if isinstance(item, ModuleWriteback):
                draft.set_module(item.path, item.value)
                kind = "module"
            else:
                draft.set_waveform(item.path, item.value)
                kind = "waveform"
            actual.append(draft.snapshot().entry_value(item.path, kind=kind))
    return tuple(actual)


def publish_arb_asset_updates(
    database_path: Path, items: tuple[ArbAssetWriteback, ...]
) -> Iterator[YamlMap]:
    """Publish samples under the entry's database_path/arb_waveforms directory.

    Items must have distinct keys. The asset owner checks the whole batch
    before writing. Yield each item's detached status, sample count and
    duration in input order; I/O failure leaves the confirmed prefix intact.
    """
    repository = ArbWaveformDatabase(database_path / "arb_waveforms")
    statuses = repository.publish_batch({item.key: item.samples for item in items})
    for item, status in zip(items, statuses, strict=True):
        yield {"status": status, **_asset_summary(item)}


class BoundDocumentWritebackTarget:
    """Write to one bound view's parameter document and adjacent library.

    view is a global or point ParameterView; storage is its capture_storage
    binding. Construction compares the complete source and both output roots,
    rejecting other entries, scopes or roots with ValueError before any write.
    No document is opened or refreshed. ledger is storage.ledger; point is the
    captured label, or None for global. Parameter/library batches each validate
    and commit one file and return detached actual values. The adjacent library
    opens lazily, creating a missing file only on its first nonempty edit.
    Assets publish to the captured entry database, and records append through
    the captured entry owner. Schema, conflict and I/O failures propagate;
    accepted events and earlier commits are not rolled back.
    """

    def __init__(self, view: ParameterView, storage: StorageBinding) -> None:
        origin = parameter_origin(view)
        document_dir = storage.result_path
        if storage.point is not None:
            document_dir = document_dir / "points" / storage.point
        if (
            (document_dir / PARAMETERS_FILENAME).resolve() != origin.source.resolve()
            or storage.result_path.resolve() != origin.result_root / storage.entry_name
            or storage.database_path.resolve()
            != origin.database_root / storage.entry_name
        ):
            raise ValueError("View and storage must belong to the same bound document")
        self._view = view
        self._storage = storage
        self.ledger = storage.ledger
        self.point = storage.point
        self._library_handle: ModuleLibrary | None = None

    def publish_arb_assets(
        self, items: tuple[ArbAssetWriteback, ...]
    ) -> Iterator[YamlMap]:
        """Publish assets in the captured entry's database directory, not the point."""
        return publish_arb_asset_updates(self._storage.database_path, items)

    def set_parameter(
        self, path: str, value: JsonValue, metadata: YamlMap | None
    ) -> YamlValue:
        """Commit one value/metadata pair; return its detached actual value."""
        return self.set_parameters((ParameterUpdate(path, value, metadata),))[0]

    def set_parameters(
        self, updates: tuple[ParameterUpdate, ...]
    ) -> tuple[YamlValue, ...]:
        """Commit one nonempty parameter batch under WritebackTarget's contract."""
        return commit_parameter_updates(self._view, updates)

    def _library(self) -> ModuleLibrary:
        from zcu_tools.resources.context.library import ModuleLibrary

        if self._library_handle is None:
            self._library_handle = ModuleLibrary(
                source_path(self._view).parent / LIBRARY_FILENAME,
                allow_absent=True,
            )
        return self._library_handle

    def set_module(self, path: str, value: ModuleSpec) -> YamlMap:
        """Commit one module and return its actual cached Spec mapping."""
        return self.set_modules((LibraryUpdate(ModuleWriteback(path, value)),))[0]

    def set_waveform(self, path: str, value: WaveformSpec) -> YamlMap:
        """Commit one waveform and return its actual cached Spec mapping."""
        return self.set_modules((LibraryUpdate(WaveformWriteback(path, value)),))[0]

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


def _indexed_assets(items: tuple[WritebackItem, ...]) -> dict[int, ArbAssetWriteback]:
    assets: dict[int, ArbAssetWriteback] = {}
    keys: set[str] = set()
    for index, item in enumerate(items):
        if isinstance(item, ArbAssetWriteback):
            if item.key in keys:
                raise ValueError(f"Duplicate arb asset key: {item.key!r}")
            keys.add(item.key)
            assets[index] = item
    return assets


def _commit_arb_assets(
    assets: dict[int, ArbAssetWriteback],
    writes: tuple[AcceptedWrite, ...],
    target: WritebackTarget,
) -> Iterator[CommittedWrite]:
    if assets:
        receipts = target.publish_arb_assets(tuple(assets.values()))
        for index, value in zip(assets, receipts, strict=True):
            yield CommittedWrite(index, writes[index].path, value)


def apply_writeback(
    items: tuple[WritebackItem, ...],
    *,
    target: WritebackTarget,
    evidence: WritebackEvidence,
) -> WritebackResult:
    """Validate values, append accepted, then commit assets, point, library and records.

    target binds the destination and its entry ledger; callers resolve roles
    before this entry. Check for empty input, duplicate asset keys, invalid
    JSON/Spec values, invalid parameter metadata and missing record run_id before
    append. Asset ledger writes contain only the path, sample count and duration.
    Record dict/list values first normalize numpy floating/integer scalars to
    Python numbers without mutating producers; non-finite results and other
    non-JSON values remain invalid. Target owners validate paths and schemas.
    Parameters commit the caller's metadata unchanged, or keep it when None; no
    event, source or time is added to it. Library items store only Specs.
    Records receive run_id, saved point, UTC at and accepted source;
    producer device fields are preserved, not inferred. Return WritebackResult
    with the accepted event ID and each owner's confirmed committed value.

    Accepted payload writes retain supplied order. Committed results instead
    follow the asset group, point group, library group, then individual records.
    Each group preserves supplied order; CommittedWrite.index maps back to
    the supplied tuple even when paths repeat. An asset conflict leaves the
    whole asset group unwritten. Point and library each use one edit; receipts
    capture validated draft values and are released only after that group commits.

    Ledger append failures propagate unchanged, with no writes. A group failure
    raises WritebackApplyError(result) from its original cause, stops later
    groups and retains the accepted event, committed groups and any asset
    prefix confirmed before the I/O error. No cross-file rollback or retry occurs.
    """
    if not items:
        raise ValueError("writeback items must be nonempty")
    if any(isinstance(item, RecordWriteback) for item in items) and (
        evidence.run_id is None or not evidence.run_id.strip()
    ):
        raise ValueError("record writeback requires a nonempty run_id")
    assets = _indexed_assets(items)
    writes = tuple(_accepted_write(item) for item in items)
    event = LedgerEvent(
        id=str(uuid4()),
        kind="accepted",
        at=datetime.now(UTC).isoformat(),
        origin=evidence.origin,
        payload=AcceptedPayload(
            writes=writes, run_id=evidence.run_id, source=evidence.source
        ),
    )
    # ADR-0063: acceptance precedes writes; no group rolls back another.
    target.ledger.append(event)
    parameters: list[ParameterUpdate] = []
    parameter_indices: list[int] = []
    library: list[LibraryUpdate] = []
    library_indices: list[int] = []
    record_indices: list[int] = []
    for index, (item, write) in enumerate(zip(items, writes, strict=True)):
        if isinstance(item, ParameterWriteback):
            parameters.append(
                ParameterUpdate(
                    write.path, write.value, item.metadata, item.create_kind
                )
            )
            parameter_indices.append(index)
        elif isinstance(item, RecordWriteback):
            record_indices.append(index)
        elif isinstance(item, (ModuleWriteback, WaveformWriteback)):
            library.append(LibraryUpdate(item))
            library_indices.append(index)

    committed: list[CommittedWrite] = []
    try:
        # Append as each asset is confirmed so an I/O error retains the prefix.
        for receipt in _commit_arb_assets(assets, writes, target):
            committed.append(receipt)
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
