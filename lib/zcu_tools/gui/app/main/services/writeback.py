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
    WritebackRequest,
)
from zcu_tools.gui.cfg import CfgSchema
from zcu_tools.gui.expected_error import FailedPreconditionError, InvalidInputError

from .guard import WritebackPermit
from .ports import CfgEdit, CfgEditorPort, ContextWrites

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from zcu_tools.gui.app.main.state import State
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
    cfg-editor handle. The handle stays here, behind ``WritebackDraft`` and
    ``WritebackService``.
    """

    item: WritebackItem
    editor_id: str | None = None


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
    hooks. The tab-named methods at the bottom are temporary A4 adapters retained
    for the current primary analyze/UI path until ticket 06 migrates the caller.
    """

    def __init__(
        self,
        state: State,
        cfg_editor: CfgEditorPort,
        write_port: ContextWritePort,
    ) -> None:
        self._state = state
        self._cfg_editor = cfg_editor
        self._write = write_port
        # Temporary tab caller adapter. The draft itself remains the authority;
        # this map only lets the current tab-level services find that draft until
        # their state contract is migrated.
        self._tab_drafts: dict[str, WritebackDraft] = {}

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
        if proposed_value is not _UNSET:
            if not isinstance(item, MetaDictWriteback):
                raise InvalidInputError(
                    f"{session_id!r} is not a metadict item; proposed_value invalid"
                )
            item.proposed_value = proposed_value

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
        return {"applied_ids": applied_ids, "written": written}

    def teardown_draft(self, draft: WritebackDraft) -> None:
        """Tear down a draft at most once; cleanup errors never cause a retry."""
        self._require_draft(draft, allow_closed=True)
        if draft._closed:
            return
        # Mark closed before calling driven cleanup. This makes teardown
        # idempotent even if a cfg editor's close implementation raises.
        draft._closed = True
        self._detach_draft_from_state(draft)
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
        # A proposal is not allowed to carry a handle from another draft. This
        # also keeps the temporary pre-refactor dynamic attribute from leaking
        # into a new opaque draft when old callers reuse an item instance.
        copied.session_id = ""
        copied.selected = True
        if isinstance(copied, MetaDictWriteback):
            copied.proposed_value = copy.deepcopy(copied.proposed_value)
        if "editor_id" in vars(copied):
            delattr(copied, "editor_id")
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

    def _detach_draft_from_state(self, draft: WritebackDraft) -> None:
        """Ensure State never retains a draft after its editor sessions close."""
        self._tab_drafts = {
            tab_id: candidate
            for tab_id, candidate in self._tab_drafts.items()
            if candidate is not draft
        }
        for tab in getattr(self._state, "tabs", {}).values():
            if getattr(tab, "writeback_draft", None) is draft:
                setattr(tab, "writeback_draft", None)
                tab.writeback_items = []

    # ------------------------------------------------------------------
    # Temporary tab caller adapters (A4; removed by the later caller migration)
    # ------------------------------------------------------------------

    def compute_items_for_tab(
        self,
        tab_id: str,
        analyze_result: Any,
        *,
        proposal_items: Iterable[WritebackItem] | None = None,
    ) -> list[WritebackItem]:
        """Temporary tab adapter around ``create_draft``.

        New workflows compute proposals themselves and pass them in. The
        no-argument proposal path remains only for the current caller contract;
        it is deliberately isolated here so the opaque draft API never needs an
        adapter or stage parameter.
        """
        tab = self._state.get_tab(tab_id)
        run_result = tab.run_result
        if run_result is None or analyze_result is None:
            return []
        if proposal_items is None:
            proposal_items = tab.adapter.get_writeback_items(
                WritebackRequest(
                    run_result=run_result,
                    analyze_result=analyze_result,
                    ctx=self._state.exp_context,
                )
            )
        draft = self.create_draft(proposal_items)
        self._tab_drafts[tab_id] = draft
        return draft.preview()

    def get_tab_writeback_draft(self, tab_id: str) -> WritebackDraft | None:
        """Temporary lookup for the current tab-level state adapter."""
        draft = self._tab_drafts.get(tab_id)
        if draft is not None and draft.is_active:
            return draft
        state_draft = getattr(self._state.get_tab(tab_id), "writeback_draft", None)
        if isinstance(state_draft, WritebackDraft) and state_draft.is_active:
            return state_draft
        return None

    def get_tab_writeback_item_draft(self, tab_id: str, session_id: str) -> CfgDraft:
        """Viewer adapter that resolves an item model without returning its id."""
        draft = self.get_tab_writeback_draft(tab_id)
        if draft is not None:
            return self.get_item_draft(draft, session_id)
        item = self._find_item(tab_id, session_id)
        editor_id = getattr(item, "editor_id", None)
        if editor_id is None:
            raise FailedPreconditionError(
                f"{session_id!r} has no editable cfg model to attach"
            )
        return self._cfg_editor.get_draft(editor_id)

    def teardown_tab_items(self, tab_id: str) -> None:
        """Temporary tab teardown adapter; safe to call repeatedly."""
        tab = self._state.get_tab(tab_id)
        draft = self._tab_drafts.pop(tab_id, None)
        state_draft = getattr(tab, "writeback_draft", None)
        if draft is None and isinstance(state_draft, WritebackDraft):
            draft = state_draft
        if draft is not None:
            self.teardown_draft(draft)
            if getattr(tab, "writeback_draft", None) is draft:
                setattr(tab, "writeback_draft", None)
            return

        # Compatibility for pre-draft state injected by the old caller/tests.
        for item in tab.writeback_items:
            editor_id = getattr(item, "editor_id", None)
            if not editor_id:
                continue
            try:
                self._cfg_editor.teardown(editor_id)
            finally:
                # Prevent a second call if this old projection is torn down
                # again before the later state clear.
                delattr(item, "editor_id")

    def get_tab_writeback_items(self, tab_id: str) -> list[WritebackItem]:
        """Temporary read projection of the current tab draft."""
        draft = self.get_tab_writeback_draft(tab_id)
        if draft is not None:
            return draft.preview()
        return list(self._state.get_tab(tab_id).writeback_items)

    def set_item_field(
        self,
        tab_id: str,
        session_id: str,
        *,
        selected: bool | None = None,
        target_name: str | None = None,
        proposed_value: Any = _UNSET,
        edits: list[dict[str, object]] | None = None,
    ) -> dict[str, object]:
        """Temporary tab adapter for ``edit_draft`` (A4)."""
        draft = self.get_tab_writeback_draft(tab_id)
        if draft is not None:
            return self.edit_draft(
                draft,
                session_id,
                selected=selected,
                target_name=target_name,
                proposed_value=proposed_value,
                edits=edits,
            )

        # Compatibility for a state list populated by the old caller/tests.
        item = self._find_item(tab_id, session_id)
        if selected is not None:
            item.selected = selected
        if target_name is not None:
            item.target_name = target_name
        if proposed_value is not _UNSET:
            if not isinstance(item, MetaDictWriteback):
                raise InvalidInputError(
                    f"{session_id!r} is not a metadict item; proposed_value invalid"
                )
            item.proposed_value = proposed_value

        result = None
        if edits is not None:
            if not isinstance(item, (ModuleWriteback, WaveformWriteback)):
                raise InvalidInputError(
                    f"{session_id!r} is not a module/waveform item; edits invalid"
                )
            editor_id = getattr(item, "editor_id", None)
            if editor_id is None:
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
            result = self._cfg_editor.set_fields(editor_id, typed_edits)
        if result is None:
            return {"valid": True, "removed": [], "added": []}
        return result.to_wire()

    def _find_item(self, tab_id: str, session_id: str) -> WritebackItem:
        for item in self._state.get_tab(tab_id).writeback_items:
            if item.session_id == session_id:
                return item
        raise InvalidInputError(f"unknown writeback session_id: {session_id!r}")

    def apply_tab_writeback(self, permit: WritebackPermit) -> dict[str, Any]:
        """Temporary guarded tab adapter around ``apply_draft`` (A4)."""
        tab_id = permit.tab_id
        draft = self.get_tab_writeback_draft(tab_id)
        if draft is not None:
            logger.info("writeback apply: tab_id=%r", tab_id)
            result = self.apply_draft(draft)
            logger.info("writeback applied: tab_id=%r", tab_id)
            return result

        # Compatibility for state lists populated by the old caller/tests.
        logger.info("writeback apply: tab_id=%r", tab_id)
        tab = self._state.get_tab(tab_id)
        applied_ids: list[str] = []
        md: dict[str, Any] = {}
        ml_modules: dict[str, CfgSchema] = {}
        ml_waveforms: dict[str, CfgSchema] = {}

        for item in tab.writeback_items:
            if not item.selected:
                continue
            if isinstance(item, MetaDictWriteback):
                md[item.target_name] = item.proposed_value
            elif isinstance(item, ModuleWriteback):
                ml_modules[item.target_name] = self._legacy_item_schema(item)
            elif isinstance(item, WaveformWriteback):
                ml_waveforms[item.target_name] = self._legacy_item_schema(item)
            else:
                raise RuntimeError(f"Unsupported writeback item type: {type(item)}")
            applied_ids.append(item.session_id)

        written = {
            "md": list(md),
            "ml_modules": list(ml_modules),
            "ml_waveforms": list(ml_waveforms),
        }
        if not (md or ml_modules or ml_waveforms):
            return {"applied_ids": applied_ids, "written": written}

        self._write.apply_writes(
            ContextWrites(md=md, ml_modules=ml_modules, ml_waveforms=ml_waveforms)
        )
        logger.info(
            "writeback applied: tab_id=%r md=%d ml_modules=%d ml_waveforms=%d",
            tab_id,
            len(md),
            len(ml_modules),
            len(ml_waveforms),
        )
        return {"applied_ids": applied_ids, "written": written}

    def _legacy_item_schema(
        self, item: ModuleWriteback | WaveformWriteback
    ) -> CfgSchema:
        editor_id = getattr(item, "editor_id", None)
        if editor_id is not None:
            return self._cfg_editor.get_draft(editor_id).snapshot()
        schema = item.edit_schema
        if schema is None:
            raise FailedPreconditionError(
                f"writeback '{item.session_id}' has no editable schema"
            )
        return schema

    # Preserve the old private helper name for narrow in-repo compatibility.
    def _item_schema(self, item: ModuleWriteback | WaveformWriteback) -> CfgSchema:
        return self._legacy_item_schema(item)
