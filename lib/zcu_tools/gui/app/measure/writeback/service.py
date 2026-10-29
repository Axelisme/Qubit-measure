from __future__ import annotations

import copy
import logging
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from pydantic import JsonValue

from zcu_tools.cfg_model import ModuleSpec
from zcu_tools.format_version import YamlMap
from zcu_tools.gui.expected_error import FailedPreconditionError, InvalidInputError
from zcu_tools.gui.form import FormWritePath
from zcu_tools.gui.session.context_control import ContextControlPort
from zcu_tools.gui.session.context_types import CommandOrigin, ContextTarget
from zcu_tools.gui.session.library_entry import ModuleEntryForm
from zcu_tools.gui.session.types import ParameterContext
from zcu_tools.resources.entry import document_unit
from zcu_tools.resources.entry import writeback as concrete
from zcu_tools.resources.entry.ledger_models import AcceptedWrite, JsonObject

from .contracts import UNCHANGED, WritebackApplied
from .contracts import WritebackAcceptance as WritebackAcceptance
from .contracts import WritebackEdit as WritebackEdit
from .contracts import WritebackSelection as WritebackSelection
from .contracts import WritebackValues as WritebackValues
from .contracts import WritebackWritten as WritebackWritten
from .items import ModuleWriteback, ParameterWriteback, RecordWriteback, WritebackItem
from .values import (
    applied_values,
    destination_path,
    destination_value,
    read_destination_value,
)

logger = logging.getLogger(__name__)

_GUI_ORIGIN = CommandOrigin("gui")


# Kind prefixes are deliberately draft-local. Two independent drafts can both
# expose ``param-1`` without sharing any Form or mutable item state.
_KIND_PREFIX = {
    ParameterWriteback: "param",
    ModuleWriteback: "ml",
    RecordWriteback: "record",
}


@dataclass
class _DraftEntry:
    """Writeback's private item/Form association.

    ``WritebackItem`` is an adapter proposal and intentionally does not carry
    an editing Form or presentation summaries. A module item's Form (None for
    other items and without a captured context) and S2 display-only baseline
    summaries stay here, owned by ``WritebackDraft``
    (app-local read model, not adapter contract).
    """

    item: WritebackItem
    form: ModuleEntryForm | None = None
    # S2 display-only baseline — not on WritebackItem, not persisted/wire.
    current_summary: str | None = None
    proposed_summary: str | None = None
    applied: bool = False


@dataclass(frozen=True)
class _RunSource:
    """Captured Run facts, independent of the editable destination/proposals."""

    experiment_name: str
    run_id: str | None
    device_snapshot: (
        Mapping[str, JsonObject] | Callable[[], Mapping[str, JsonObject] | None] | None
    )


def _entries_to_apply(
    entries: Sequence[_DraftEntry], item_ids: tuple[str, ...] | None
) -> list[_DraftEntry]:
    """Resolve explicit IDs independently of the GUI checkbox selection."""
    if item_ids is None:
        return [entry for entry in entries if entry.item.selected]
    requested = set(item_ids)
    if len(requested) != len(item_ids):
        raise InvalidInputError("duplicate writeback item IDs")
    unknown = requested - {entry.item.session_id for entry in entries}
    if unknown:
        raise InvalidInputError(f"unknown writeback item IDs: {sorted(unknown)}")
    return [entry for entry in entries if entry.item.session_id in requested]


@dataclass(frozen=True)
class WritebackDraftApplication:
    """One draft's application outcome, without rollback or automatic retry.

    applied_ids lists confirmed committed IDs: draft order on full success,
    committed-group order on partial failure.
    written groups all prepared destination paths by parameter/ml_modules/records;
    it describes the requested writes, including uncommitted paths after failure.
    error is the original write failure after a confirmed prefix, or None.
    values contains detached actual values and fixed destinations for committed
    IDs when receipt capture was requested, otherwise an empty tuple.
    """

    applied_ids: tuple[str, ...]
    written: dict[str, list[str]]
    error: BaseException | None = None
    values: tuple[WritebackApplied, ...] = ()


class WritebackDraft:
    """Own one pane's detached proposals, Forms, captured source and applied state.

    Create through WritebackService.create_draft. Public operations require an
    active draft; teardown is idempotent. Forms are draft-local, and app callers
    must obtain control admission immediately before each Form access or edit.
    Source evaluation stays fixed while preview and Apply use active documents.
    """

    __slots__ = (
        "_owner",
        "_write",
        "_entries",
        "_closed",
        "_parameters",
        "_run_source",
        "_unavailable_reasons",
    )

    def __init__(
        self,
        owner: WritebackService,
        write_port: ContextControlPort,
        items: Iterable[WritebackItem],
        parameters: ParameterContext | None,
        *,
        arb_names: Sequence[str],
        run_source: _RunSource,
    ) -> None:
        self._run_source = copy.deepcopy(run_source)
        self._unavailable_reasons: tuple[str, ...] = ()
        self._owner = owner
        self._write = write_port
        self._entries: list[_DraftEntry] = []
        self._parameters = parameters
        self._closed = False
        for raw_item in items:
            item = self._copy_item(raw_item)
            prefix = self._kind_prefix(item)
            item.session_id = self._next_session_id(self._entries, prefix)
            entry = _DraftEntry(item)
            self._capture_baseline(entry, parameters)
            if isinstance(item, ModuleWriteback) and parameters is not None:
                entry.form = self._open_form(
                    entry, item.proposal, parameters, arb_names
                )
            self._entries.append(entry)

    @property
    def parameters(self) -> ParameterContext | None:
        """Fixed creation capture, or None for a context-free empty draft.

        This read-only source fixes role bindings and library values, not the
        Apply destination. Later selection changes never replace it.
        """
        return self._parameters

    @property
    def device_snapshot(self) -> Mapping[str, JsonObject] | None:
        """Project detached Run device settings; malformed historical evidence raises.

        Deferred sources validate only on this explicit read. None means the
        Run did not capture device evidence, not that validation failed.
        """
        source = self._run_source.device_snapshot
        return copy.deepcopy(source() if callable(source) else source)

    @property
    def unavailable_reasons(self) -> tuple[str, ...]:
        """Draft diagnostics; opaque record content never creates a reason."""
        return self._unavailable_reasons

    @property
    def experiment_name(self) -> str:
        """Experiment name fixed at creation, used as accepted evidence kind."""
        return self._run_source.experiment_name

    @property
    def run_id(self) -> str | None:
        """Historical RunMetadata identity fixed at creation, or None without it.

        Save does not change this value. It identifies a Run, not whether that
        Run has a saved file; native Load retains the file's original identity.
        """
        return self._run_source.run_id

    @property
    def items(self) -> tuple[WritebackItem, ...]:
        """Current item projection, without any editor identity."""
        return tuple(entry.item for entry in self._entries)

    def validate_selection(
        self, service: WritebackService, item_ids: tuple[str, ...] | None
    ) -> None:
        """Validate ownership, activity and IDs without reading or writing context.

        service is used only for identity against the creating service.
        Foreign drafts raise InvalidInputError before closed drafts raise
        FailedPreconditionError. None selects checked items; () selects none.
        Explicit IDs ignore checkboxes and resolve in draft order, not ID order.
        Duplicate or unknown IDs raise InvalidInputError. No values or
        destinations are evaluated, and no snapshot or write is made.
        Cross-draft coordinators use this before accept; app frontends still
        obtain tab/pane admission through WritebackControlPort.
        """
        if self._owner is not service:
            raise InvalidInputError("unknown writeback draft")
        self._selected_entries(item_ids)

    def _require_active(self) -> None:
        if self._closed:
            raise FailedPreconditionError("writeback draft has been torn down")

    def _selected_entries(self, item_ids: tuple[str, ...] | None) -> list[_DraftEntry]:
        self._require_active()
        return _entries_to_apply(self._entries, item_ids)

    @property
    def is_active(self) -> bool:
        """Whether this draft has not been torn down."""
        return not self._closed

    def preview(self) -> list[WritebackItem]:
        """Return ordered proposal projections; a closed draft raises."""
        self._require_active()
        return list(self.items)

    def preview_values(
        self, *, session_id: str | None = None
    ) -> dict[str, WritebackValues]:
        """Read the active document set, or current=None without an open entry.

        Proposed module values evaluate against the creation capture, never
        current focus, point, or slot references. Missing destinations have
        current=None. An invalid module Form raises LibraryEntryInvalid.
        session_id=None reads every item, including unselected items. A supplied
        session_id reads only that item's facts, without lowering siblings; an
        unknown ID raises KeyError before reading the destination. Empty bulk
        reads return {}; empty item reads raise KeyError. Preview does not authorize
        Apply: without an entry it still evaluates proposals from their fixed
        source, but reports current=None. Inactive drafts are rejected.
        Target/source/I/O and requested proposal validation errors propagate.
        """
        self._require_active()
        items = self.items
        if session_id is not None:
            items = tuple(item for item in items if item.session_id == session_id)
            if not items:
                raise KeyError(session_id)
        if not items:
            return {}
        parameters = self._require_parameters()
        target = (
            self._write.read_context().target
            if self._write.get_entry_info() is not None
            else None
        )
        current = None if target is None else self._write.read_target(target)
        values: dict[str, WritebackValues] = {}
        for item in items:
            path = destination_path(item, parameters)
            if isinstance(item, RecordWriteback):
                proposed = item.record
            elif isinstance(item, ParameterWriteback):
                proposed = AcceptedWrite.model_validate(
                    {"path": path, "value": item.proposed_value}
                ).value
            else:
                entry = self._find_draft_entry(item.session_id)
                proposed = self._module_spec(entry).model_dump(mode="python")
            values[item.session_id] = WritebackValues(
                actual_path=path,
                current=(
                    read_destination_value(self._write, self, item, current, path)
                    if current is not None
                    else None
                ),
                proposed=proposed,
                unit=(
                    document_unit(current.document(), path)
                    if isinstance(item, ParameterWriteback) and current is not None
                    else None
                ),
            )
        return values

    def _require_parameters(self) -> ParameterContext:
        parameters = self.parameters
        return ParameterContext.empty() if parameters is None else parameters

    def _current_target(self) -> ContextTarget:
        target = self._write.read_context().target
        if target is None:
            raise FailedPreconditionError(
                "Open an entry first", reason_code="no_context"
            )
        return target

    def edit(
        self,
        session_id: str,
        *,
        selected: bool | None = None,
        target_name: str | None = None,
        proposed_value: Any = UNCHANGED,
        edits: Mapping[FormWritePath, JsonValue] | None = None,
    ) -> None:
        """Apply one draft-local edit, in the order selection, target, value, edits.

        selected only changes the GUI default apply selection. target_name
        retargets any item. proposed_value replaces a parameter proposal; other
        items raise InvalidInputError. edits writes one batch into a module
        item's Form with FormState.write semantics: the batch installs whole,
        ordinary invalid values stay in the Form and are reported by preview
        and apply, and structurally unknown paths raise InvalidInputError
        naming the item without changing the Form. Edits on a non-module item
        raise InvalidInputError; on a module item without a captured context,
        FailedPreconditionError. Earlier changes in this call are kept when a
        later one fails. Content changes mark the item unapplied.
        """
        self._require_active()
        entry = self._find_draft_entry(session_id)
        item = entry.item
        if selected is not None and selected != item.selected:
            item.selected = selected
        if target_name is not None and target_name != item.target_name:
            item.target_name = target_name
            self._mark_unapplied(entry)
            # Keep the bounded proposed summary in sync for a module retarget.
            if isinstance(item, ModuleWriteback):
                # Preserve original current_summary; recompute proposed_summary
                # from new target (still bounded, not full cfg).
                is_update = entry.current_summary == "present"
                action = "update" if is_update else "create"
                if item.role_id:
                    entry.proposed_summary = f"{action} {item.role_id}"
                else:
                    entry.proposed_summary = f"{action} → {item.target_name}"
        if proposed_value is not UNCHANGED:
            if not isinstance(item, ParameterWriteback):
                raise InvalidInputError(
                    f"{session_id!r} is not a parameter item; proposed_value invalid"
                )
            if not self._values_equal(proposed_value, item.proposed_value):
                item.proposed_value = copy.deepcopy(proposed_value)
                entry.proposed_summary = self._format_value_summary(proposed_value)
                self._mark_unapplied(entry)
        if edits is not None:
            self._write_form(entry, edits)

    def _write_form(
        self, entry: _DraftEntry, edits: Mapping[FormWritePath, JsonValue]
    ) -> None:
        """Write one batch into a module item's Form; see edit for errors."""
        session_id = entry.item.session_id
        if not isinstance(entry.item, ModuleWriteback):
            raise InvalidInputError(
                f"{session_id!r} is not a module item; edits invalid"
            )
        form = self._require_form(entry)
        try:
            form.state.write(edits, form.env)
        except ValueError as exc:
            # FormState rejects unknown/overlapping paths and missing sources
            # before installing anything; report them as the agent's input error.
            raise InvalidInputError(f"{session_id!r}: {exc}") from exc

    def get_item_form(self, session_id: str) -> ModuleEntryForm:
        """Return the module item's Form that edits, preview and apply all use.

        Viewers write it with form.state.write(batch, form.env), execute
        form.state.run_command(path, name, form.env), or use a FormEditor
        on (form.state, form.env). Both operations publish through the same
        state subscription and mark the item unapplied. App callers obtain
        admission through get_writeback_item_form_for_pane immediately before
        each operation, not just when opening the view. A non-module
        item raises InvalidInputError; a module item without a captured
        context raises FailedPreconditionError. Inactive drafts and
        unknown IDs raise as for edit.
        """
        self._require_active()
        entry = self._find_draft_entry(session_id)
        if not isinstance(entry.item, ModuleWriteback):
            raise InvalidInputError(f"{session_id!r} is not a module item; no Form")
        return self._require_form(entry)

    def apply(
        self,
        *,
        item_ids: tuple[str, ...] | None = None,
        origin: CommandOrigin = _GUI_ORIGIN,
    ) -> dict[str, Any]:
        """Evaluate against the draft capture, append accepted, then set in order.

        Role paths resolve through the creation capture, never current focus.
        All selected values, including module Specs from their Forms, are built
        before the ledger append; an invalid module Form raises
        LibraryEntryInvalid with its field paths and writes nothing. Append
        failure writes nothing. On item failure, mark the successful prefix applied and re-raise
        the original cause unchanged, retaining accepted evidence and writes.
        Empty selection is a no-op without context. Nonempty writes require an
        open entry and use its active global or point documents. GUI defaults
        to the GUI origin; MCP callers explicitly identify the MCP frontend.
        Unknown/duplicate IDs or concrete destinations fail before any write.
        Accepted payload uses self.run_id;
        application never re-reads Run or Save state to determine its identity.
        Each parameter's metadata is replaced by {"stderr": x} when the item has
        stderr, else by {}; no source, kind, run_id or time is written there.
        """
        result = self._apply_draft(item_ids=item_ids, origin=origin)
        if result.error is not None:
            raise result.error from result.error.__cause__
        return {"applied_ids": list(result.applied_ids), "written": result.written}

    def accept(
        self,
        item_ids: tuple[str, ...] | None,
        *,
        origin: CommandOrigin = _GUI_ORIGIN,
    ) -> WritebackDraftApplication:
        """Apply one draft and capture actual committed values for coordination.

        None selects checked items; () selects none. Explicit IDs ignore
        checkboxes and use draft order. Closed drafts, unknown/duplicate IDs,
        invalid module Forms, malformed proposal values, duplicate destinations
        and missing context raise before writing. Empty selection needs no
        context. Source evaluation uses the fixed
        capture; the destination is read now, not during validate_selection.
        Append/admission failures propagate. A write-group failure, including
        destination document validation, returns its original cause in error,
        preserving accepted evidence, committed IDs
        and their values. written still includes every prepared path, not just
        the confirmed prefix. No rollback or retry occurs. GUI is the default
        origin; MCP callers supply CommandOrigin("mcp").
        This is the cross-draft coordinator seam, not frontend admission.
        """
        return self._apply_draft(item_ids=item_ids, origin=origin, capture_values=True)

    def _apply_draft(
        self,
        *,
        item_ids: tuple[str, ...] | None,
        origin: CommandOrigin,
        capture_values: bool = False,
    ) -> WritebackDraftApplication:
        self._require_active()
        selected_entries = self._selected_entries(item_ids)
        written: dict[str, list[str]] = {
            "parameter": [],
            "ml_modules": [],
            "records": [],
        }
        if not selected_entries:
            return WritebackDraftApplication((), written)
        target = self._current_target()
        parameters = self._require_parameters()
        destinations: set[str] = set()
        items: list[concrete.WritebackItem] = []
        for entry in selected_entries:
            item = entry.item
            if isinstance(item, RecordWriteback):
                path = item.target_name
                items.append(concrete.RecordWriteback(path, item.record))
                written["records"].append(path)
            elif isinstance(item, ParameterWriteback):
                path = destination_path(item, parameters)
                write = AcceptedWrite.model_validate(
                    {"path": path, "value": item.proposed_value}
                )
                metadata: YamlMap = (
                    {"stderr": item.stderr} if item.stderr is not None else {}
                )
                items.append(concrete.ParameterWriteback(path, write.value, metadata))
                written["parameter"].append(path)
            else:
                path = destination_path(item, parameters)
                items.append(concrete.ModuleWriteback(path, self._module_spec(entry)))
                written["ml_modules"].append(path)
            if path in destinations:
                raise InvalidInputError(f"duplicate writeback destination: {path!r}")
            destinations.add(path)
        try:
            outcome = self._write.apply_writeback(
                target,
                tuple(items),
                evidence=concrete.WritebackEvidence(
                    origin=origin.origin,
                    run_id=self.run_id,
                ),
            )
        except concrete.WritebackApplyError as exc:
            completed_entries = tuple(
                selected_entries[item.index] for item in exc.result.committed
            )
            for entry in completed_entries:
                entry.applied = True
            # Keep GUI/wire classification and the producer's own cause chain.
            cause = exc.__cause__
            if cause is None:
                raise RuntimeError(
                    "WritebackApplyError requires its original cause"
                ) from exc
            return WritebackDraftApplication(
                tuple(entry.item.session_id for entry in completed_entries),
                written,
                cause,
                applied_values(
                    target,
                    tuple(entry.item for entry in selected_entries),
                    exc.result,
                )
                if capture_values
                else (),
            )
        for entry in selected_entries:
            entry.applied = True
        return WritebackDraftApplication(
            tuple(entry.item.session_id for entry in selected_entries),
            written,
            values=applied_values(
                target, tuple(entry.item for entry in selected_entries), outcome
            )
            if capture_values
            else (),
        )

    def writeback_write(
        self,
        changes: tuple[WritebackEdit, ...],
        *,
        origin: CommandOrigin = _GUI_ORIGIN,
        on_applied: Callable[[tuple[WritebackApplied, ...]], None] | None = None,
    ) -> list[WritebackWritten]:
        """Edit sequentially and return active-document before/after values.

        Read the currently selected destination before application, then use each write
        owner's committed value; no postcommit destination read is needed.
        Targets in receipts are concrete paths resolved from the creation capture.
        Empty changes are a no-op. Edit failures, including a listed module
        whose Form is invalid (LibraryEntryInvalid), retain the successful
        draft prefix without writes. Read/append/setter failures propagate; accepted
        evidence and any successful writes remain. MCP supplies
        CommandOrigin("mcp"); GUI uses the default. on_applied, when supplied,
        receives the confirmed prefix before any write error propagates; it must
        only record the detached receipt, not issue further mutations.
        """
        self._require_active()
        if not changes:
            return []
        parameters = self._require_parameters()
        ids = tuple(change.session_id for change in changes)
        if len(set(ids)) != len(ids):
            raise InvalidInputError("Repeated writeback IDs")
        items = []
        for change in changes:
            self.edit(
                change.session_id,
                target_name=change.target_name,
                proposed_value=change.proposed_value,
                edits=change.edits,
            )
            entry = self._find_draft_entry(change.session_id)
            if isinstance(entry.item, ModuleWriteback):
                # Built only to reject invalid input before any context read or write.
                self._module_spec(entry)
            items.append(entry.item)
        paths = [destination_path(item, parameters) for item in items]
        current = self._write.read_target(self._current_target())
        before = [
            read_destination_value(self._write, self, item, current, path)
            for item, path in zip(items, paths, strict=True)
        ]
        applied = self.accept(ids, origin=origin)
        if on_applied is not None:
            on_applied(applied.values)
        if applied.error is not None:
            raise applied.error from applied.error.__cause__
        previous_values = {
            item.session_id: previous
            for item, previous in zip(items, before, strict=True)
        }
        return [
            WritebackWritten(
                id=committed.id,
                kind=committed.kind,
                target=committed.target,
                before=previous_values[committed.id],
                after=committed.value,
            )
            for committed in applied.values
        ]

    def teardown(self) -> None:
        """Close this draft idempotently; later active operations raise.

        Item Forms hold no session or resource, so closing releases nothing.
        """
        self._closed = True

    @staticmethod
    def _copy_item(item: WritebackItem) -> WritebackItem:
        copied = copy.deepcopy(item)
        copied.session_id = ""
        return copied

    @staticmethod
    def _kind_prefix(item: WritebackItem) -> str:
        for item_type, prefix in _KIND_PREFIX.items():
            if isinstance(item, item_type):
                return prefix
        raise InvalidInputError(
            f"unsupported writeback item type: {type(item).__name__}"
        )

    @staticmethod
    def _next_session_id(entries: list[_DraftEntry], prefix: str) -> str:
        count = sum(entry.item.session_id.startswith(f"{prefix}-") for entry in entries)
        return f"{prefix}-{count + 1}"

    def _find_draft_entry(self, session_id: str) -> _DraftEntry:
        for entry in self._entries:
            if entry.item.session_id == session_id:
                return entry
        raise InvalidInputError(f"unknown writeback session_id: {session_id!r}")

    def _open_form(
        self,
        entry: _DraftEntry,
        proposal: ModuleSpec,
        parameters: ParameterContext,
        arb_names: Sequence[str],
    ) -> ModuleEntryForm:
        form = ModuleEntryForm(proposal, parameters, arb_names=arb_names)
        # The Form lives exactly as long as its entry, so the subscription is
        # never removed and its unsubscribe handle is not kept.
        form.state.subscribe(lambda: self._mark_unapplied(entry))
        return form

    @staticmethod
    def _require_form(entry: _DraftEntry) -> ModuleEntryForm:
        if entry.form is None:
            raise FailedPreconditionError(
                f"{entry.item.session_id!r} has no Form without a captured context"
            )
        return entry.form

    def _module_spec(self, entry: _DraftEntry) -> ModuleSpec:
        """Return the module item's Spec; an invalid Form raises LibraryEntryInvalid."""
        return self._require_form(entry).spec()

    # ------------------------------------------------------------------
    # S2 — display-only baseline capture (read-only, single owner)
    # ------------------------------------------------------------------

    @staticmethod
    def _format_value_summary(value: Any) -> str:
        if value is None:
            return "—"
        if isinstance(value, float):
            return f"{value:.6g}"
        if isinstance(value, complex):
            return f"{value.real:.4g}{value.imag:+.4g}j"
        if isinstance(value, (list, tuple)):
            return WritebackDraft._format_sequence_summary(value)
        if isinstance(value, dict):
            return f"map[{len(value)}]"
        text = repr(value)
        return text if len(text) <= 48 else f"{text[:45]}..."

    @staticmethod
    def _format_sequence_summary(value: list[object] | tuple[object, ...]) -> str:
        """Summarize rectangular nonempty scalar rows as a matrix, else as a list."""
        rows: list[list[object] | tuple[object, ...]] = []
        for row in value:
            if not isinstance(row, (list, tuple)):
                return f"list[{len(value)}]"
            rows.append(row)
        if not rows:
            return f"list[{len(value)}]"
        row_lengths = {len(row) for row in rows}
        scalar_cells = all(
            not isinstance(cell, (list, tuple, dict)) for row in rows for cell in row
        )
        if len(row_lengths) == 1 and next(iter(row_lengths)) > 0 and scalar_cells:
            return f"{len(value)} × {next(iter(row_lengths))} matrix"
        return f"list[{len(value)}]"

    def _capture_baseline(
        self, entry: _DraftEntry, parameters: ParameterContext | None
    ) -> None:
        """Capture display-only summaries from the creation context."""
        item = entry.item
        if isinstance(item, RecordWriteback):
            entry.current_summary = "entry record"
            entry.proposed_summary = self._format_value_summary(item.record)
        elif isinstance(item, ParameterWriteback):
            current = (
                None
                if parameters is None
                else destination_value(
                    item, parameters, destination_path(item, parameters)
                )
            )
            entry.current_summary = self._format_value_summary(current)
            entry.proposed_summary = self._format_value_summary(item.proposed_value)
        elif isinstance(item, ModuleWriteback):
            exists = False
            if parameters is not None:
                path = destination_path(item, parameters)
                exists = path in parameters.library.module_keys()
            entry.current_summary = "present" if exists else "— not present"
            action = "update" if exists else "create"
            role = item.role_id
            entry.proposed_summary = (
                f"{action} {role}" if role else f"{action} → {item.target_name}"
            )

    # App-local Qt presentation projection (S2) — not wire/persistence.
    def get_summaries(self, session_id: str) -> tuple[str | None, str | None]:
        """Return (current_summary, proposed_summary) for one item."""
        self._require_active()
        entry = self._find_draft_entry(session_id)
        return entry.current_summary, entry.proposed_summary

    def get_all_summaries(self) -> dict[str, tuple[str | None, str | None]]:
        """Return mapping session_id -> (current, proposed) for the draft."""
        self._require_active()
        return {
            e.item.session_id: (e.current_summary, e.proposed_summary)
            for e in self._entries
        }

    def get_all_applied(self) -> dict[str, bool]:
        """Return the draft-owned applied state for presentation."""
        self._require_active()
        return {entry.item.session_id: entry.applied for entry in self._entries}

    @staticmethod
    def _values_equal(left: Any, right: Any) -> bool:
        if left is right:
            return True
        try:
            return bool(left == right)
        except (TypeError, ValueError):
            return False

    @staticmethod
    def _mark_unapplied(entry: _DraftEntry) -> None:
        entry.applied = False


class WritebackService:
    """Create independent drafts and coordinate ordered acceptance across drafts.

    Persistent writes use the existing ContextControlPort supplied at creation.
    Per-draft reads, edits, Apply and teardown belong to WritebackDraft.
    """

    def __init__(
        self,
        write_port: ContextControlPort,
        *,
        arb_names: Callable[[], Sequence[str]],
    ) -> None:
        """Write through write_port; read arb data options from arb_names.

        arb_names returns the arb waveform asset keys available to module
        Forms. It is called once per draft that has a module item, when the
        draft is created; every arb data field in that draft accepts only
        those names for its lifetime.
        """
        self._write = write_port
        self._arb_names = arb_names

    # ------------------------------------------------------------------
    # Opaque draft lifecycle
    # ------------------------------------------------------------------

    def create_draft(
        self,
        items: Iterable[WritebackItem],
        *,
        parameters: ParameterContext | None,
        experiment_name: str,
        run_id: str | None = None,
        device_snapshot: (
            Mapping[str, JsonObject]
            | Callable[[], Mapping[str, JsonObject] | None]
            | None
        ) = None,
    ) -> WritebackDraft:
        """Create a draft with one Form per module item at the creation capture.

        The input is a proposal collection, not an adapter or stage. Items are
        deep-copied so draft-local ``session_id``, selection, target and value
        edits cannot mutate the adapter's proposal objects. Author element
        identities are detached at creation and never resolved against aliases.
        Initial selection and the optional display note come from each proposal.
        Quality decisions belong to the adapter; edits and Apply retain the note.
        Each module proposal
        is projected into a library entry Form (ModuleEntryForm): child library
        keys stay keys, and the optional hardware fields the shared Form
        generator leaves out return to Spec defaults. A proposal that is not a
        concrete persistent ModuleSpec raises TypeError and no draft is returned.

        parameters fixes expression/library resolution and baseline display,
        not the write destination. Preview and Apply use the active document set
        through write_port at that call. None provides no historical values;
        concrete parameter/record proposals can still write to an open entry.
        Module items without parameters have no Form and cannot be applied.
        experiment_name is the captured experiment name, never a frontend/event kind.
        run_id is the source RunMetadata.run_id, including unsaved Runs and
        historical native Loads; None means no native acquisition evidence.
        It stays fixed through Save, selection changes, edits and application.
        Application copies it into accepted evidence.
        device_snapshot maps Run-fixed device names to their JSON settings, or
        supplies a reader over detached Run evidence. Readers run only when
        draft.device_snapshot is requested and must preserve owner validation
        errors. None means no device evidence. Records are detached adapter-owned JSON;
        this service neither reads device facts nor adds coordinate fields.
        The append owner requires run_id and supplies only the provenance
        envelope (run_id/point/at/source). Record format follows adapter policy.
        """
        if not experiment_name.strip():
            raise InvalidInputError("experiment_name must be nonempty")
        proposals = list(items)
        arb_names: tuple[str, ...] = ()
        if parameters is not None and any(
            isinstance(item, ModuleWriteback) for item in proposals
        ):
            arb_names = tuple(self._arb_names())
        return WritebackDraft(
            self,
            self._write,
            proposals,
            parameters,
            arb_names=arb_names,
            run_source=_RunSource(experiment_name, run_id, device_snapshot),
        )

    def accept_drafts(
        self,
        selections: tuple[WritebackSelection, ...],
        *,
        origin: CommandOrigin = _GUI_ORIGIN,
    ) -> WritebackAcceptance:
        """Check every captured group before the first write, then apply in order.

        Call synchronously on the owner thread; no yield separates validation
        from application. Unknown/duplicate IDs, foreign/closed/repeated drafts
        raise before any write. Use current draft values and destinations;
        do not edit or restore proposals here. Write failures
        return the confirmed prefix and diagnostic, without retry or rollback.
        """
        seen: set[WritebackDraft] = set()
        for selection in selections:
            draft = selection.draft
            # A seen draft already passed owner/activity checks in this
            # synchronous call. Preserve repeated-draft priority over bad IDs.
            if draft in seen:
                raise InvalidInputError("Repeated captured writeback draft")
            draft.validate_selection(self, selection.ids)
            seen.add(draft)
        applied: list[tuple[str, ...]] = [() for _ in selections]
        written: list[tuple[WritebackApplied, ...]] = [() for _ in selections]
        for index, selection in enumerate(selections):
            try:
                result = selection.draft.accept(selection.ids, origin=origin)
            except Exception as error:
                # Isolate admission/ledger failure; earlier groups remain confirmed.
                logger.exception("Captured writeback failed after confirmed groups")
                return WritebackAcceptance(tuple(applied), str(error), tuple(written))
            applied[index] = result.applied_ids
            written[index] = result.values
            if result.error is not None:
                return WritebackAcceptance(
                    tuple(applied), str(result.error), tuple(written)
                )
        return WritebackAcceptance(tuple(applied), written=tuple(written))
