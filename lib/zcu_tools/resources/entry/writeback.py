"""Ordered accepted writes over point and library single-file transactions."""

from dataclasses import dataclass
from datetime import UTC, datetime
from math import isfinite
from uuid import uuid4

from pydantic import TypeAdapter

from zcu_tools.cfg_model import (
    ConfigBase,
    ModuleSpec,
    WaveformSpec,
    module_spec_from_raw,
    waveform_spec_from_raw,
)

from .ledger_models import (
    AcceptedPayload,
    AcceptedWrite,
    JsonValue,
    LedgerEvent,
    Origin,
    SourceReference,
)
from .points import PointView
from .provenance import Provenance
from .storage import PointStorageBinding


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


type WritebackItem = ParameterWriteback | ModuleWriteback | WaveformWriteback


@dataclass(frozen=True)
class WritebackEvidence:
    """Caller-captured source facts shared by all writes in one acceptance.

    kind is the experiment name for a run, or analysis kind for a cross-run
    analysis; it is nonempty. origin identifies gui, notebook or mcp. run_id is
    the captured run identity, or None for cross-run analysis. source optionally
    references the analyzed event; None is ordinary single-run acceptance.
    """

    kind: str
    origin: Origin
    run_id: str | None = None
    source: SourceReference | None = None


def _require_named_spec(value: object) -> None:
    if not isinstance(value, ConfigBase):
        raise TypeError("library writeback requires a named Spec, not a raw mapping")


def _accepted_write(item: WritebackItem) -> AcceptedWrite:
    if isinstance(item, ParameterWriteback):
        if item.stderr is not None:
            stderr = TypeAdapter(float).validate_python(item.stderr, strict=True)
            if not isfinite(stderr) or stderr < 0:
                raise ValueError("stderr must be finite and non-negative, or None")
        return AcceptedWrite.model_validate({"path": item.path, "value": item.value})
    _require_named_spec(item.value)
    if isinstance(item, ModuleWriteback):
        value = module_spec_from_raw(item.value).to_dict()
    else:
        value = waveform_spec_from_raw(item.value).to_dict()
    return AcceptedWrite(path=item.path, value=value)


def apply_writeback(
    items: tuple[WritebackItem, ...],
    *,
    point: PointView,
    storage: PointStorageBinding,
    evidence: WritebackEvidence,
) -> str:
    """Append one accepted event, then commit each item in supplied order.

    point and storage must be the caller's captured owning point and storage
    binding. Items use concrete paths (role resolution belongs to the caller).
    Nonempty items, strict JSON/Spec values, finite non-negative stderr and a
    nonempty evidence.kind are validated before appending. Library key/family
    and point schema validation happen at their existing owner during apply.

    Each item uses its owner's single-file edit. The first failure propagates
    unchanged and stops later items; already committed items and the accepted
    event remain. There is no retry or cross-item/cross-file rollback. Library
    opens lazily at the captured point, only if a library item is reached.
    Return the accepted event ID only after all writes succeed.
    """
    # Library imports entry provenance; defer the import to avoid that cycle.
    from zcu_tools.resources.context.library import ModuleLibrary

    if not items:
        raise ValueError("writeback items must be nonempty")
    kind = TypeAdapter(str).validate_python(evidence.kind, strict=True)
    if not kind.strip():
        raise ValueError("writeback kind must be nonempty")
    writes = tuple(_accepted_write(item) for item in items)
    event = LedgerEvent(
        id=str(uuid4()),
        kind="accepted",
        at=datetime.now(UTC).isoformat(),
        entry_id=str(storage.entry_id),
        origin=evidence.origin,
        payload=AcceptedPayload(
            writes=writes,
            analysis_summary={},
            run_id=evidence.run_id,
            source=evidence.source,
        ),
    )
    storage.ledger.append(event)
    library: ModuleLibrary | None = None
    for item, write in zip(items, writes, strict=True):
        provenance = Provenance(
            source=event.id,
            kind=kind,
            run_id=evidence.run_id,
            at=event.at,
            stderr=item.stderr if isinstance(item, ParameterWriteback) else None,
        )
        if isinstance(item, ParameterWriteback):
            with point.edit() as draft:
                draft.set(write.path, write.value, provenance=provenance)
        else:
            if library is None:
                library = ModuleLibrary(
                    storage.result_path / "points" / storage.point / "module_cfg.yaml",
                    ledger=storage.result_path / "records" / "ledger.jsonl",
                    entry_id=str(storage.entry_id),
                )
            with library.edit() as draft:
                if isinstance(item, ModuleWriteback):
                    draft.set_module(item.path, item.value, provenance=provenance)
                else:
                    draft.set_waveform(item.path, item.value, provenance=provenance)
    return event.id
