"""Owner-facing edit, preview and submission records, separate from adapter proposals."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal, TypedDict

from pydantic import JsonValue

from zcu_tools.gui.form import FormWritePath

if TYPE_CHECKING:
    from .service import WritebackDraft

# Shared edit default: leave the proposal unchanged; None replaces its value.
UNCHANGED: Any = object()


@dataclass(frozen=True)
class WritebackEdit:
    """One ordered agent edit; omitted values differ from explicit None.

    session_id is the draft-local item ID. target_name, when given, retargets
    the item. proposed_value, when supplied, replaces a parameter proposal.
    edits, when given, is one Form write batch for a module item: model-field
    leaf paths mapped to JSON input values, which may be {"expr": text}.
    """

    session_id: str
    target_name: str | None = None
    proposed_value: Any = UNCHANGED
    edits: Mapping[FormWritePath, JsonValue] | None = None


type WritebackKind = Literal["parameter", "module", "record", "arb_asset"]


@dataclass(frozen=True)
class WritebackWritten:
    """Actual destination values around a successful context write.

    id is the draft-local item ID. kind identifies the write owner. target is a
    concrete destination. before is the preview of that destination before
    application; for assets it is the sample comparison, not the samples.
    after is the committed value. Record values are complete JSONL rows.
    """

    id: str
    kind: WritebackKind
    target: str
    before: object
    after: object


@dataclass(frozen=True)
class WritebackValues:
    """Complete preview facts for one item at its captured destination.

    actual_path is the captured parameter path, library key or record component,
    after any draft-local retarget edit. current is the latest stored value at
    that destination, or None if missing. proposed is the complete validated
    value; for a module it is the Spec from the item's Form, evaluated against
    the creation capture and dumped in the same shape as a stored library
    entry. unit is the resources-provided static schema label for display only,
    or None for destinations without one (including ext and module items).
    Values never convert. warning gives any known Apply conflict; None means
    no preview warning.
    """

    actual_path: str
    current: object
    proposed: object
    unit: str | None
    warning: str | None = None


class ArbAssetSummary(TypedDict):
    """Sample count, reference duration (µs) and maximum |I + iQ|; no arrays."""

    sample_count: int
    duration_us: float
    peak_abs: float


@dataclass(frozen=True)
class ArbAssetValues(WritebackValues):
    """Arb asset preview with a typed summary instead of raw sample arrays."""

    current: Literal["absent", "identical", "different"] | None
    proposed: ArbAssetSummary


@dataclass(frozen=True)
class WritebackSelection:
    """One captured draft submission, in caller-specified group order.

    draft is an active owner handle. ids lists explicit draft-local IDs,
    independent of GUI selection. Submission uses current values and destinations.
    """

    draft: WritebackDraft
    ids: tuple[str, ...]


class WritebackDestination(TypedDict):
    """Fixed persisted destination in the current Result root.

    entry_name names the entry directory. point is a saved point label, or None
    for the entry-level global documents. This receipt always names an entry.
    """

    entry_name: str
    point: str | None


@dataclass(frozen=True)
class WritebackApplied:
    """One confirmed write captured at submission, never a later preview.

    id is draft-local; kind identifies the write owner. target is the concrete
    destination. value is the owner's committed value, including asset status.
    destination identifies the fixed entry name within the current result root
    and the point that received it.
    """

    id: str
    kind: WritebackKind
    target: str
    value: object
    destination: WritebackDestination


@dataclass(frozen=True)
class WritebackAcceptance:
    """Receipt for an ordered group submission, without retry or rollback.

    applied_ids has one tuple per submitted draft, in submission order; only
    confirmed writes appear. error is None on completion, otherwise the first
    write failure diagnostic. written has one group per submitted draft, with
    the actual values and fixed destinations for its confirmed prefix. Invalid
    IDs raise before writes.
    """

    applied_ids: tuple[tuple[str, ...], ...]
    error: str | None = None
    # One group per submitted draft, containing detached actual successful values.
    written: tuple[tuple[WritebackApplied, ...], ...] = ()
