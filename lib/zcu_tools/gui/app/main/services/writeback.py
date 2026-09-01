from __future__ import annotations

import copy
import logging
import uuid
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

from zcu_tools.gui.app.main.adapter import (
    MetaDictWriteback,
    ModuleWriteback,
    WaveformWriteback,
    WritebackItem,
)
from zcu_tools.gui.cfg import CfgSchema
from zcu_tools.gui.expected_error import FailedPreconditionError, InvalidInputError

from .ports import CfgEdit, CfgEditorPort, ContextWrites

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from zcu_tools.gui.cfg.binding import CfgDraft

    from .ports import ContextWritePort

# Kind prefixes are deliberately draft-local. Two independent drafts can both
# expose ``md-1`` without sharing any editor session or mutable item state.
_KIND_PREFIX = {
    MetaDictWriteback: "md",
    ModuleWriteback: "ml",
    WaveformWriteback: "wf",
}

# Sentinel for "argument not supplied" in set_item_field (None is a real value).
_UNSET: Any = object()


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

    __slots__ = ("_service", "_identity", "_entries", "_closed")

    def __init__(
        self,
        service: WritebackService,
        identity: str,
        entries: list[_DraftEntry],
    ) -> None:
        self._service = service
        self._identity = identity
        self._entries = entries
        self._closed = False

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

    def apply(self) -> dict[str, Any]:
        return self._service.apply_draft(self)

    def teardown(self) -> None:
        self._service.teardown_draft(self)


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
        write_port: ContextWritePort,
    ) -> None:
        self._cfg_editor = cfg_editor
        self._write = write_port

    # ------------------------------------------------------------------
    # Opaque draft lifecycle
    # ------------------------------------------------------------------

    def create_draft(self, items: Iterable[WritebackItem]) -> WritebackDraft:
        """Create a draft and all item-local cfg sessions transactionally.

        The input is a proposal collection, not an adapter or stage. Items are
        shallow-copied so draft-local ``session_id``, selection, target and value
        edits cannot mutate the adapter's proposal objects. If any editor session
        fails to open, every session opened for this draft is torn down before the
        original exception is re-raised; no partially-created draft escapes.

        S2 — captures a display-only baseline from the destination context at
        creation time (read-only, one owner, not persisted/wire). Scalar
        MetaDict items show concrete current vs proposed values; module/waveform
        items show bounded target/change summaries.
        """
        identity = uuid.uuid4().hex
        entries: list[_DraftEntry] = []
        # Snapshot the destination context once at draft creation (read-only,
        # S2). Summaries are stored in the service-owned _DraftEntry, not on
        # the public WritebackItem (adapter contract unchanged).
        ctx = self._snapshot_context()
        try:
            for raw_item in items:
                item = self._copy_item(raw_item)
                prefix = self._kind_prefix(item)
                item.session_id = self._next_session_id(entries, prefix)
                item.selected = True
                entry = _DraftEntry(item)
                # Baseline capture — display-only, app/service-owned.
                self._capture_baseline(entry, ctx)
                entries.append(entry)
                if isinstance(item, (ModuleWriteback, WaveformWriteback)):
                    if item.edit_schema is None:
                        continue
                    editor_id, _ = self._cfg_editor.open_seeded(
                        item.edit_schema,
                        gc=False,
                        owner_key=f"writeback:{identity}:{item.session_id}",
                    )
                    entry.editor_id = editor_id
        except BaseException:
            # Cleanup is best-effort per session so one teardown failure cannot
            # strand the remaining sessions. The creation failure remains the
            # observable error and no draft is returned.
            self._teardown_entries(entries)
            raise
        return WritebackDraft(self, identity, entries)

    def preview_draft(self, draft: WritebackDraft) -> list[WritebackItem]:
        self._require_draft(draft)
        return list(draft.items)

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
        """Apply one draft-local edit, preserving ordered fail-fast semantics."""
        self._require_draft(draft)
        entry = self._find_draft_entry(draft, session_id)
        item = entry.item
        if selected is not None:
            item.selected = selected
        if target_name is not None:
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
            item.proposed_value = proposed_value
            entry.proposed_summary = self._format_scalar(proposed_value)
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
            result = self._cfg_editor.set_fields(entry.editor_id, typed_edits)
            entry.applied = False
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

    def apply_draft(self, draft: WritebackDraft) -> dict[str, Any]:
        """Apply selected entries through exactly one ``ContextWritePort`` call."""
        self._require_draft(draft)
        applied_ids: list[str] = []
        md: dict[str, Any] = {}
        ml_modules: dict[str, CfgSchema] = {}
        ml_waveforms: dict[str, CfgSchema] = {}

        for entry in draft._entries:
            item = entry.item
            if not item.selected:
                continue
            if isinstance(item, MetaDictWriteback):
                md[item.target_name] = item.proposed_value
            elif isinstance(item, ModuleWriteback):
                ml_modules[item.target_name] = self._entry_schema(entry)
            elif isinstance(item, WaveformWriteback):
                ml_waveforms[item.target_name] = self._entry_schema(entry)
            else:
                raise RuntimeError(f"Unsupported writeback item type: {type(item)}")
            applied_ids.append(item.session_id)

        written = {
            "md": list(md),
            "ml_modules": list(ml_modules),
            "ml_waveforms": list(ml_waveforms),
        }
        if md or ml_modules or ml_waveforms:
            self._write.apply_writes(
                ContextWrites(
                    md=md,
                    ml_modules=ml_modules,
                    ml_waveforms=ml_waveforms,
                )
            )
            applied_id_set = set(applied_ids)
            for entry in draft._entries:
                if entry.item.session_id in applied_id_set:
                    entry.applied = True
        return {"applied_ids": applied_ids, "written": written}

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
        if isinstance(copied, MetaDictWriteback):
            copied.proposed_value = copy.deepcopy(copied.proposed_value)
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
        schema = getattr(entry.item, "edit_schema", None)
        if schema is None:
            raise FailedPreconditionError(
                f"writeback '{entry.item.session_id}' has no editable schema"
            )
        return schema

    # ------------------------------------------------------------------
    # S2 — display-only baseline capture (read-only, single owner)
    # ------------------------------------------------------------------

    def _snapshot_context(self) -> Any | None:
        """Best-effort read of the live destination ExpContext.

        In production ``_write`` is the Controller, which exposes
        ``get_exp_context``. In tests it is a MagicMock, so we probe
        defensively and return None when unavailable — the draft remains
        usable with fallback summaries.
        """
        for attr in ("get_exp_context",):
            if hasattr(self._write, attr):
                try:
                    getter = getattr(self._write, attr)
                    ctx = getter() if callable(getter) else getter
                    if ctx is not None and hasattr(ctx, "md") and hasattr(ctx, "ml"):
                        return ctx
                except Exception:
                    continue
        return None

    @staticmethod
    def _format_scalar(value: Any) -> str:
        if value is None:
            return "—"
        if isinstance(value, float):
            return f"{value:.6g}"
        if isinstance(value, complex):
            return f"{value.real:.4g}{value.imag:+.4g}j"
        return repr(value)

    def _capture_baseline(self, entry: _DraftEntry, ctx: Any | None) -> None:
        """Populate entry.current_summary / entry.proposed_summary.

        Stored in service-owned _DraftEntry, not on public WritebackItem.
        When ``ctx`` is unavailable (tests or early init), fall back to a
        neutral placeholder so the draft remains displayable. The summaries are
        display-only and never alter persistence/wire/MCP/adapter contracts.
        """
        item = entry.item
        try:
            if isinstance(item, MetaDictWriteback):
                # Current from live md
                has_current = False
                current: Any = None
                if ctx is not None:
                    try:
                        has_current = item.target_name in ctx.md.keys()  # type: ignore[attr-defined]
                        current = ctx.md.get(item.target_name, None)  # type: ignore[attr-defined]
                    except Exception:
                        has_current = False
                        current = None
                if has_current:
                    entry.current_summary = self._format_scalar(current)
                else:
                    entry.current_summary = "—"
                entry.proposed_summary = self._format_scalar(item.proposed_value)
            elif isinstance(item, (ModuleWriteback, WaveformWriteback)):
                exists = False
                if ctx is not None:
                    try:
                        if isinstance(item, ModuleWriteback):
                            exists = item.target_name in ctx.ml.modules  # type: ignore[attr-defined]
                        else:
                            exists = item.target_name in ctx.ml.waveforms  # type: ignore[attr-defined]
                    except Exception:
                        exists = False
                entry.current_summary = "present" if exists else "— not present"
                action = "update" if exists else "create"
                role = getattr(item, "role_id", None)
                if role:
                    entry.proposed_summary = f"{action} {role}"
                else:
                    entry.proposed_summary = f"{action} → {item.target_name}"
            else:
                entry.current_summary = None
                entry.proposed_summary = None
        except Exception:
            logger.debug(
                "baseline capture failed for %r", item.target_name, exc_info=True
            )
            entry.current_summary = None
            entry.proposed_summary = None

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
