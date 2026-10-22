"""Owner-facing edit, preview and submission records, separate from adapter proposals."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Literal

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


@dataclass(frozen=True)
class WritebackWritten:
    """Actual destination values around a successful context write.

    id is the draft-local item ID. kind identifies parameter/module/sample.
    target is a concrete point path, library key or sample component. before is
    the latest stored value before application, or None when absent. after is the
    value read after successful application. Sample values are complete JSONL
    rows, including accepted metadata, not the producer-only proposal fields.
    """

    id: str
    kind: Literal["parameter", "module", "sample"]
    target: str
    before: object
    after: object


@dataclass(frozen=True)
class WritebackValues:
    """Complete preview facts for one item at its captured destination.

    actual_path is the captured parameter path, library key or sample component,
    after any draft-local retarget edit. current is the latest stored value at
    that destination, or None if missing. proposed is the complete validated
    value; for a module it is the Spec from the item's Form, evaluated against
    the creation capture and dumped in the same shape as a stored library
    entry. unit is the resources-provided static schema label for display only,
    or None for destinations without one (including ext and module items).
    Values never convert.
    """

    actual_path: str
    current: object
    proposed: object
    unit: str | None


@dataclass(frozen=True)
class WritebackSelection:
    """One captured draft submission, in caller-specified group order.

    draft is an active owner handle. ids lists explicit draft-local IDs,
    independent of GUI selection. Submission uses current values and destinations.
    """

    draft: WritebackDraft
    ids: tuple[str, ...]


@dataclass(frozen=True)
class WritebackAcceptance:
    """Receipt for an ordered group submission, without retry or rollback.

    applied_ids has one tuple per submitted draft, in submission order; only
    confirmed writes appear. error is None on completion, otherwise the first
    write failure diagnostic. Invalid IDs raise before writes.
    """

    applied_ids: tuple[tuple[str, ...], ...]
    error: str | None = None
