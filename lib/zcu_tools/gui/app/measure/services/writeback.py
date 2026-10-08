from __future__ import annotations

import copy
import logging
import uuid
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal

from zcu_tools.cfg_model import (
    ModuleSpec,
    WaveformSpec,
    module_spec_from_raw,
    waveform_spec_from_raw,
)
from zcu_tools.gui.app.measure.adapter import (
    MetaDictWriteback,
    ModuleWriteback,
    SessionEnv,
    WaveformWriteback,
    WritebackItem,
)
from zcu_tools.gui.app.measure.adapter.lowering import schema_to_raw_dict
from zcu_tools.gui.cfg import CfgSchema
from zcu_tools.gui.expected_error import FailedPreconditionError, InvalidInputError
from zcu_tools.gui.session.context_control import ContextControlPort
from zcu_tools.gui.session.context_types import CommandOrigin
from zcu_tools.gui.session.types import ParameterContext
from zcu_tools.resources.entry.ledger import RecordsLedger
from zcu_tools.resources.entry.ledger_models import (
    AcceptedPayload,
    AcceptedWrite,
    LedgerEvent,
)
from zcu_tools.resources.entry.provenance import Provenance

from .ports import CfgEdit, CfgEditorPort

logger = logging.getLogger(__name__)
_GUI_ORIGIN = CommandOrigin("gui", None)

if TYPE_CHECKING:
    from zcu_tools.gui.cfg.binding import CfgDraft

# Kind prefixes are deliberately draft-local. Two independent drafts can both
# expose ``md-1`` without sharing any editor session or mutable item state.
_KIND_PREFIX = {
    MetaDictWriteback: "md",
    ModuleWriteback: "ml",
    WaveformWriteback: "wf",
}

# Sentinel for "argument not supplied" in set_item_field (None is a real value).
_UNSET: Any = object()


@dataclass(frozen=True)
class WritebackEdit:
    """One ordered agent edit; omitted values differ from explicit None."""

    session_id: str
    target_name: str | None = None
    proposed_value: Any = _UNSET
    edits: list[dict[str, object]] | None = None


@dataclass(frozen=True)
class WritebackWritten:
    """Actual destination values around a successful context write."""

    id: str
    kind: Literal["md", "module", "waveform"]
    target: str
    before: object
    after: object


def _destination_value(item: WritebackItem, context: SessionEnv) -> object:
    if isinstance(item, MetaDictWriteback):
        value = context.md.get(item.target_name, None)
    else:
        collection = (
            context.ml.modules
            if isinstance(item, ModuleWriteback)
            else context.ml.waveforms
        )
        target = collection.get(item.target_name)
        value = None if target is None else target.to_dict()
    return copy.deepcopy(value)


@dataclass(frozen=True)
class WritebackValues:
    """Complete values at the active destination and in the current draft."""

    current: object
    proposed: object


@dataclass
class _DraftEntry:
    """Writeback's private item/session association.

    ``WritebackItem`` is an adapter proposal and intentionally does not carry a
    cfg-editor handle or presentation summaries. Handles and S2 display-only
    baseline summaries stay here, behind ``WritebackDraft`` and
    ``WritebackService`` (app/service-owned read model, not adapter contract).
    """

    item: WritebackItem
    editor_id: str | None = None
    # S2 display-only baseline — not on WritebackItem, not persisted/wire.
    current_summary: str | None = None
    proposed_summary: str | None = None
    applied: bool = False


class WritebackDraft:
    """Opaque, service-owned writeback draft.

    A draft is the only handle a workflow needs after handing proposal items to
    :class:`WritebackService`. Its item list is a read-model projection; cfg
    editor identities and teardown bookkeeping remain private to the service.
    Mutations should use the service (or the small forwarding methods below), so
    the same draft is shared by the UI and remote editing surfaces.
    """

    __slots__ = ("_service", "_identity", "_entries", "_closed", "_parameters")

    def __init__(
        self,
        service: WritebackService,
        identity: str,
        entries: list[_DraftEntry],
        parameters: ParameterContext | None,
    ) -> None:
        self._service = service
        self._identity = identity
        self._entries = entries
        self._parameters = parameters
        self._closed = False

    @property
    def parameters(self) -> ParameterContext | None:
        """Fixed creation capture, or None for a context-free empty draft.

        This read-only source owns role bindings, library values and the write
        target. Later selection changes never replace it.
        """
        return self._parameters

    @property
    def items(self) -> tuple[WritebackItem, ...]:
        """Current item projection, without any editor identity."""
        return tuple(entry.item for entry in self._entries)

    @property
    def is_active(self) -> bool:
        return not self._closed

    def preview(self) -> list[WritebackItem]:
        return self._service.preview_draft(self)

    def edit(
        self,
        session_id: str,
        *,
        selected: bool | None = None,
        target_name: str | None = None,
        proposed_value: Any = _UNSET,
        edits: list[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        return self._service.edit_draft(
            self,
            session_id,
            selected=selected,
            target_name=target_name,
            proposed_value=proposed_value,
            edits=edits,
        )

    def apply(
        self,
        *,
        item_ids: tuple[str, ...] | None = None,
        origin: CommandOrigin = _GUI_ORIGIN,
    ) -> dict[str, Any]:
        return self._service.apply_draft(self, item_ids=item_ids, origin=origin)

    def teardown(self) -> None:
        self._service.teardown_draft(self)


def _entries_to_apply(
    entries: list[_DraftEntry], item_ids: tuple[str, ...] | None
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


class WritebackService:
    """Own opaque transactional drafts and their cfg-editor sessions.

    The primary interface is ``create_draft`` → ``preview_draft`` / ``edit_draft``
    → ``apply_draft`` / ``teardown_draft``. It accepts already-computed proposal
    items and has no knowledge of analysis stages, subtabs, figures, or adapter
    hooks.
    """

    def __init__(
        self,
        cfg_editor: CfgEditorPort,
        write_port: ContextControlPort,
        entry_ledger: Callable[[str], RecordsLedger],
    ) -> None:
        self._cfg_editor = cfg_editor
        self._write = write_port
        self._entry_ledger = entry_ledger

    # ------------------------------------------------------------------
    # Opaque draft lifecycle
    # ------------------------------------------------------------------

    def create_draft(
        self,
        items: Iterable[WritebackItem],
        *,
        parameters: ParameterContext | None,
    ) -> WritebackDraft:
        """Create a draft and all item-local cfg sessions transactionally.

        The input is a proposal collection, not an adapter or stage. Items are
        shallow-copied so draft-local ``session_id``, selection, target and value
        edits cannot mutate the adapter's proposal objects. If any editor session
        fails to open, every session opened for this draft is torn down before the
        original exception is re-raised; no partially-created draft escapes.

        parameters fixes both expression/library resolution and destination
        target for the draft lifetime. None is valid for an empty/context-free
        proposal, but nonempty application fails without a destination. Baseline
        display uses this capture; no live selection provider is retained.
        """
        identity = uuid.uuid4().hex
        entries: list[_DraftEntry] = []
        try:
            for raw_item in items:
                item = self._copy_item(raw_item)
                prefix = self._kind_prefix(item)
                item.session_id = self._next_session_id(entries, prefix)
                item.selected = True
                entry = _DraftEntry(item)
                # Baseline capture — display-only, app/service-owned.
                self._capture_baseline(entry, parameters)
                entries.append(entry)
                if isinstance(item, (ModuleWriteback, WaveformWriteback)):
                    if item.edit_schema is None:
                        continue
                    editor_id, _ = self._cfg_editor.open_seeded(
                        item.edit_schema,
                        parameters=parameters,
                        gc=False,
                        owner_key=f"writeback:{identity}:{item.session_id}",
                    )
                    entry.editor_id = editor_id
                    editor_draft = self._cfg_editor.get_draft(editor_id)
                    editor_draft.on_change.connect(
                        lambda *_args, draft_entry=entry: self._mark_unapplied(
                            draft_entry
                        )
                    )
        except BaseException:
            # Cleanup is best-effort per session so one teardown failure cannot
            # strand the remaining sessions. The creation failure remains the
            # observable error and no draft is returned.
            self._teardown_entries(entries)
            raise
        return WritebackDraft(self, identity, entries, parameters)

    def preview_draft(self, draft: WritebackDraft) -> list[WritebackItem]:
        self._require_draft(draft)
        return list(draft.items)

    def preview_values(
        self, draft: WritebackDraft, context: SessionEnv
    ) -> dict[str, WritebackValues]:
        """Read live destination values; do not reuse presentation baselines.

        Missing targets have current=None. Missing context or invalid cfg raises;
        this is not a promise that later writes will see the same context.
        """
        self._require_draft(draft)
        md, ml = context.md, context.ml
        if not context.has_context():
            raise FailedPreconditionError("No experiment context.")
        values = {}
        for item in draft.items:
            if isinstance(item, MetaDictWriteback):
                current = md.get(item.target_name, None)
                proposed = item.proposed_value
            elif isinstance(item, (ModuleWriteback, WaveformWriteback)):
                collection = (
                    ml.modules if isinstance(item, ModuleWriteback) else ml.waveforms
                )
                target = collection.get(item.target_name)
                current = None if target is None else target.to_dict()
                schema = self.get_item_draft(draft, item.session_id).snapshot()
                proposed = schema_to_raw_dict(schema, md, ml)
            else:
                raise RuntimeError(f"Unsupported writeback item type: {type(item)}")
            values[item.session_id] = WritebackValues(
                current=copy.deepcopy(current), proposed=copy.deepcopy(proposed)
            )
        return values

    def edit_draft(
        self,
        draft: WritebackDraft,
        session_id: str,
        *,
        selected: bool | None = None,
        target_name: str | None = None,
        proposed_value: Any = _UNSET,
        edits: list[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        """Apply one draft-local edit; agent cfg changes use aggregate grammar."""
        self._require_draft(draft)
        entry = self._find_draft_entry(draft, session_id)
        item = entry.item
        if selected is not None:
            item.selected = selected
        if target_name is not None and target_name != item.target_name:
            item.target_name = target_name
            entry.applied = False
            # Keep proposed summary in sync for module/waveform retarget (bounded)
            if isinstance(item, (ModuleWriteback, WaveformWriteback)):
                # Preserve original current_summary; recompute proposed_summary
                # from new target (still bounded, not full cfg).
                base_current = entry.current_summary
                is_update = base_current == "present"
                action = "update" if is_update else "create"
                role = getattr(item, "role_id", None)
                if role:
                    entry.proposed_summary = f"{action} {role}"
                else:
                    entry.proposed_summary = f"{action} → {item.target_name}"
        if proposed_value is not _UNSET:
            if not isinstance(item, MetaDictWriteback):
                raise InvalidInputError(
                    f"{session_id!r} is not a metadict item; proposed_value invalid"
                )
            if not self._values_equal(proposed_value, item.proposed_value):
                item.proposed_value = proposed_value
                entry.proposed_summary = self._format_value_summary(proposed_value)
                entry.applied = False

        result = None
        if edits is not None:
            if not isinstance(item, (ModuleWriteback, WaveformWriteback)):
                raise InvalidInputError(
                    f"{session_id!r} is not a module/waveform item; edits invalid"
                )
            if entry.editor_id is None:
                raise FailedPreconditionError(
                    f"{session_id!r} has no editable cfg model to apply edits to"
                )
            typed_edits: list[CfgEdit] = []
            for i, edit in enumerate(edits):
                if "path" not in edit or "value" not in edit:
                    raise InvalidInputError(
                        f"edits[{i}] must be an object with 'path' and 'value'"
                    )
                typed_edits.append(CfgEdit(str(edit["path"]), edit["value"]))
            result = self._cfg_editor.set_fields(
                entry.editor_id, typed_edits, agent_edit=True
            )
        if result is None:
            return {"valid": True, "removed": [], "added": []}
        return result.to_wire()

    def get_item_draft(self, draft: WritebackDraft, session_id: str) -> CfgDraft:
        """Return an item model for a viewer without exposing its editor id."""
        self._require_draft(draft)
        entry = self._find_draft_entry(draft, session_id)
        if entry.editor_id is None:
            raise FailedPreconditionError(
                f"{session_id!r} has no editable cfg model to attach"
            )
        return self._cfg_editor.get_draft(entry.editor_id)

    def apply_draft(
        self,
        draft: WritebackDraft,
        *,
        item_ids: tuple[str, ...] | None = None,
        origin: CommandOrigin = _GUI_ORIGIN,
    ) -> dict[str, Any]:
        """Lower against the draft capture, append accepted, then set in order.

        Role paths resolve through the creation capture, never current focus.
        All selected values lower before the ledger append. Append failure writes
        nothing; setter failure stops immediately, retaining accepted evidence
        and successful writes. Empty selection is a no-op without context.
        Nonempty writes require a capture and its still-open entry. GUI defaults
        to a native GUI origin; MCP callers supply the native call ID explicitly.
        Unknown/duplicate IDs or concrete destinations fail before any write.
        """
        self._require_draft(draft)
        selected_entries = _entries_to_apply(draft._entries, item_ids)
        written: dict[str, list[str]] = {"md": [], "ml_modules": [], "ml_waveforms": []}
        if not selected_entries:
            return {"applied_ids": [], "written": written}
        parameters = draft.parameters
        if parameters is None:
            raise FailedPreconditionError("Writeback requires a captured context.")
        ledger = self._entry_ledger(parameters.target.entry_id)
        destinations: set[str] = set()
        writes: list[AcceptedWrite] = []
        modules: dict[str, ModuleSpec] = {}
        waveforms: dict[str, WaveformSpec] = {}
        for entry in selected_entries:
            item = entry.item
            if isinstance(item, MetaDictWriteback):
                path = parameters.resolve_path(item.target_name)
                write = AcceptedWrite.model_validate(
                    {"path": path, "value": item.proposed_value}
                )
                written["md"].append(path)
            else:
                path = parameters.library_key(item.target_name)
                raw = schema_to_raw_dict(self._entry_schema(entry), parameters)
                if isinstance(item, ModuleWriteback):
                    spec = module_spec_from_raw(raw)
                    modules[path] = spec
                    written["ml_modules"].append(path)
                else:
                    spec = waveform_spec_from_raw(raw)
                    waveforms[path] = spec
                    written["ml_waveforms"].append(path)
                write = AcceptedWrite.model_validate(
                    {"path": path, "value": spec.model_dump(mode="json")}
                )
            if path in destinations:
                raise InvalidInputError(f"duplicate writeback destination: {path!r}")
            destinations.add(path)
            writes.append(write)

        event = LedgerEvent(
            id=str(uuid.uuid4()),
            kind="accepted",
            at=datetime.now(timezone.utc).isoformat(),
            entry_id=parameters.target.entry_id,
            origin=origin.origin,
            call_id=origin.call_id,
            payload=AcceptedPayload(writes=tuple(writes), analysis_summary={}),
        )
        ledger.append(event)
        source = Provenance(
            source=event.id,
            kind=event.kind,
            run_id=None,
            at=event.at,
            stderr=None,
        )
        applied_ids: list[str] = []
        for entry, write in zip(selected_entries, writes, strict=True):
            item = entry.item
            if isinstance(item, MetaDictWriteback):
                self._write.set_parameter(
                    parameters.target, write.path, write.value, source=source
                )
            elif isinstance(item, ModuleWriteback):
                self._write.set_module(
                    parameters.target, write.path, modules[write.path], source=source
                )
            else:
                self._write.set_waveform(
                    parameters.target, write.path, waveforms[write.path], source=source
                )
            entry.applied = True
            applied_ids.append(item.session_id)
        return {"applied_ids": applied_ids, "written": written}

    def write_draft(
        self,
        draft: WritebackDraft,
        changes: tuple[WritebackEdit, ...],
        context: SessionEnv,
    ) -> list[WritebackWritten]:
        """Edit sequentially, then apply only these IDs once.

        An edit failure keeps the successful draft prefix but starts no context
        write. Context write failures retain the existing write-port semantics.
        The caller supplies the live context on the owner thread.
        """
        self._require_draft(draft)
        if not context.has_context():
            raise FailedPreconditionError("No experiment context.")
        ids = tuple(change.session_id for change in changes)
        if len(set(ids)) != len(ids):
            raise InvalidInputError("Repeated writeback IDs")
        items = []
        for change in changes:
            result = self.edit_draft(
                draft,
                change.session_id,
                target_name=change.target_name,
                proposed_value=change.proposed_value,
                edits=change.edits,
            )
            if not result["valid"]:
                raise FailedPreconditionError(
                    f"Invalid writeback draft: {change.session_id!r}"
                )
            items.append(self._find_draft_entry(draft, change.session_id).item)
        before = [_destination_value(item, context) for item in items]
        self.apply_draft(draft, item_ids=ids)
        return [
            WritebackWritten(
                id=item.session_id,
                kind=(
                    "md"
                    if isinstance(item, MetaDictWriteback)
                    else "module"
                    if isinstance(item, ModuleWriteback)
                    else "waveform"
                ),
                target=item.target_name,
                before=previous,
                after=_destination_value(item, context),
            )
            for item, previous in zip(items, before, strict=True)
        ]

    def teardown_draft(self, draft: WritebackDraft) -> None:
        """Tear down a draft at most once; cleanup errors never cause a retry."""
        self._require_draft(draft, allow_closed=True)
        if draft._closed:
            return
        # Mark closed before calling driven cleanup. This makes teardown
        # idempotent even if a cfg editor's close implementation raises.
        draft._closed = True
        self._teardown_entries(draft._entries)

    # Short behavior-oriented aliases for callers that already hold the opaque
    # draft. They keep the public surface independent of stage/subtab vocabulary.
    def preview(self, draft: WritebackDraft) -> list[WritebackItem]:
        return self.preview_draft(draft)

    def edit(
        self, draft: WritebackDraft, session_id: str, **changes: Any
    ) -> dict[str, object]:
        return self.edit_draft(draft, session_id, **changes)

    def apply(self, draft: WritebackDraft) -> dict[str, Any]:
        return self.apply_draft(draft)

    def teardown(self, draft: WritebackDraft) -> None:
        self.teardown_draft(draft)

    def _require_draft(
        self, draft: WritebackDraft, *, allow_closed: bool = False
    ) -> None:
        if not isinstance(draft, WritebackDraft) or draft._service is not self:
            raise InvalidInputError("unknown writeback draft")
        if draft._closed and not allow_closed:
            raise FailedPreconditionError("writeback draft has been torn down")

    @staticmethod
    def _copy_item(item: WritebackItem) -> WritebackItem:
        copied = copy.copy(item)
        # Proposal items never carry cfg-editor identity. Treat a dynamic
        # attribute as an invalid caller contract instead of silently lowering it.
        if "editor_id" in vars(item):
            raise InvalidInputError("writeback proposal must not expose editor_id")
        copied.session_id = ""
        copied.selected = True
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

    def _find_draft_entry(self, draft: WritebackDraft, session_id: str) -> _DraftEntry:
        for entry in draft._entries:
            if entry.item.session_id == session_id:
                return entry
        raise InvalidInputError(f"unknown writeback session_id: {session_id!r}")

    def _entry_schema(self, entry: _DraftEntry) -> CfgSchema:
        if entry.editor_id is not None:
            return self._cfg_editor.get_draft(entry.editor_id).snapshot()
        item = entry.item
        schema = (
            item.edit_schema
            if isinstance(item, (ModuleWriteback, WaveformWriteback))
            else None
        )
        if schema is None:
            raise FailedPreconditionError(
                f"writeback '{entry.item.session_id}' has no editable schema"
            )
        return schema

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
            if value and all(isinstance(row, (list, tuple)) for row in value):
                row_lengths = {len(row) for row in value}
                scalar_cells = all(
                    not isinstance(cell, (list, tuple, dict))
                    for row in value
                    for cell in row
                )
                if (
                    len(row_lengths) == 1
                    and next(iter(row_lengths)) > 0
                    and scalar_cells
                ):
                    return f"{len(value)} × {next(iter(row_lengths))} matrix"
            return f"list[{len(value)}]"
        if isinstance(value, dict):
            return f"map[{len(value)}]"
        text = repr(value)
        return text if len(text) <= 48 else f"{text[:45]}..."

    def _capture_baseline(
        self, entry: _DraftEntry, parameters: ParameterContext | None
    ) -> None:
        """Capture display-only summaries from the creation context."""
        item = entry.item
        if isinstance(item, MetaDictWriteback):
            current = (
                None if parameters is None else parameters.lookup(item.target_name)
            )
            entry.current_summary = self._format_value_summary(
                current.value if current is not None and current.present else None
            )
            entry.proposed_summary = self._format_value_summary(item.proposed_value)
        elif isinstance(item, (ModuleWriteback, WaveformWriteback)):
            exists = False
            if parameters is not None:
                path = parameters.library_key(item.target_name)
                keys = (
                    parameters.library.module_keys()
                    if isinstance(item, ModuleWriteback)
                    else parameters.library.waveform_keys()
                )
                exists = path in keys
            entry.current_summary = "present" if exists else "— not present"
            action = "update" if exists else "create"
            role = item.role_id
            entry.proposed_summary = (
                f"{action} {role}" if role else f"{action} → {item.target_name}"
            )

    # App-local Qt presentation projection (S2) — not wire/persistence.
    def get_summaries(
        self, draft: WritebackDraft, session_id: str
    ) -> tuple[str | None, str | None]:
        """Return (current_summary, proposed_summary) for one item."""
        self._require_draft(draft)
        entry = self._find_draft_entry(draft, session_id)
        return entry.current_summary, entry.proposed_summary

    def get_all_summaries(
        self, draft: WritebackDraft
    ) -> dict[str, tuple[str | None, str | None]]:
        """Return mapping session_id -> (current, proposed) for the draft."""
        self._require_draft(draft)
        return {
            e.item.session_id: (e.current_summary, e.proposed_summary)
            for e in draft._entries
        }

    def get_all_applied(self, draft: WritebackDraft) -> dict[str, bool]:
        """Return the draft-owned applied state for presentation."""
        self._require_draft(draft)
        return {entry.item.session_id: entry.applied for entry in draft._entries}

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

    def _teardown_entries(self, entries: Iterable[_DraftEntry]) -> None:
        for entry in reversed(list(entries)):
            editor_id = entry.editor_id
            entry.editor_id = None
            if editor_id is None:
                continue
            try:
                self._cfg_editor.teardown(editor_id)
            except Exception:
                logger.exception(
                    "writeback draft editor teardown failed: %s", editor_id
                )
