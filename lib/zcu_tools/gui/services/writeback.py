from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any, Optional

from zcu_tools.gui.adapter import (
    CfgSchema,
    MetaDictWriteback,
    ModuleWriteback,
    WaveformWriteback,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.event_bus import (
    GuiEvent,
    TabContentChangedPayload,
)
from zcu_tools.gui.services.guard import WritebackPermit
from zcu_tools.gui.services.ports import ContextWrites

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from zcu_tools.gui.event_bus import EventBus
    from zcu_tools.gui.services.cfg_editor import CfgEditorService
    from zcu_tools.gui.services.ports import ContextWritePort
    from zcu_tools.gui.state import State

# kind prefix for the stable per-item session_id (``<kind>-<n>``).
_KIND_PREFIX = {
    MetaDictWriteback: "md",
    ModuleWriteback: "ml",
    WaveformWriteback: "wf",
}

# Sentinel for "argument not supplied" in set_item_field (None is a real value).
_UNSET: Any = object()


class WritebackService:
    """Owns the persistent writeback draft for each tab (ADR-0010).

    Writeback items are computed **once** when analyze finishes (``compute_items``)
    and stored on ``TabState.writeback_items``. Each module/waveform item gets a
    gc=False CfgEditorService model (seeded from its ``edit_schema``); the agent
    edits it via ``editor.set_field`` and the user's Edit dialog attaches to the
    same model (WYSIWYG). ``get_tab_writeback_items`` is a pure read of that
    persistent list — it never recomputes (which would discard live edits).
    """

    def __init__(
        self,
        state: "State",
        bus: "EventBus",
        cfg_editor: "CfgEditorService",
        write_port: "ContextWritePort",
    ) -> None:
        self._state = state
        self._bus = bus
        self._cfg_editor = cfg_editor
        self._write = write_port

    # ------------------------------------------------------------------
    # Compute once (analyze sink) + read
    # ------------------------------------------------------------------

    def compute_items_for_tab(self, tab_id: str) -> list[WritebackItem]:
        """Compute the tab's writeback items once (analyze sink calls this).

        Calls the adapter, stamps a stable per-kind ``session_id``, and for each
        module/waveform item opens a gc=False CfgEditorService model seeded from
        its ``edit_schema`` (storing the ``editor_id``). The returned list is
        stored on ``TabState.writeback_items`` by the analyze sink.
        """
        tab = self._state.get_tab(tab_id)
        run_result = tab.run_result
        analyze_result = tab.analyze_result
        if run_result is None or analyze_result is None:
            return []
        items = list(
            tab.adapter.get_writeback_items(
                WritebackRequest(
                    run_result=run_result,
                    analyze_result=analyze_result,
                    ctx=self._state.exp_context,
                )
            )
        )
        counter: dict[str, int] = {}
        for item in items:
            prefix = _KIND_PREFIX[type(item)]
            counter[prefix] = counter.get(prefix, 0) + 1
            item.session_id = f"{prefix}-{counter[prefix]}"
            item.selected = True
            if isinstance(item, (ModuleWriteback, WaveformWriteback)):
                if item.edit_schema is not None:
                    editor_id, _ = self._cfg_editor.open_seeded(
                        item.edit_schema,
                        gc=False,
                        owner_key=f"writeback:{tab_id}:{item.session_id}",
                    )
                    item.editor_id = editor_id
        return items

    def teardown_tab_items(self, tab_id: str) -> None:
        """Tear down every per-item editor model for a tab (on reanalyze/rerun)."""
        tab = self._state.get_tab(tab_id)
        for item in tab.writeback_items:
            if (
                isinstance(item, (ModuleWriteback, WaveformWriteback))
                and item.editor_id
            ):
                self._cfg_editor.teardown(item.editor_id)

    def get_tab_writeback_items(self, tab_id: str) -> list[WritebackItem]:
        """Pure read of the persistent items (no recompute)."""
        return list(self._state.get_tab(tab_id).writeback_items)

    # ------------------------------------------------------------------
    # Edit a persistent item (agent / UI writeback.set)
    # ------------------------------------------------------------------

    def set_item_field(
        self,
        tab_id: str,
        session_id: str,
        *,
        selected: Optional[bool] = None,
        target_name: Optional[str] = None,
        proposed_value: Any = _UNSET,
        edited_schema: Optional[CfgSchema] = _UNSET,  # type: ignore[assignment]
    ) -> None:
        item = self._find_item(tab_id, session_id)
        if selected is not None:
            item.selected = selected
        if target_name is not None:
            item.target_name = target_name
        if proposed_value is not _UNSET:
            if not isinstance(item, MetaDictWriteback):
                raise RuntimeError(
                    f"{session_id!r} is not a metadict item; proposed_value invalid"
                )
            item.proposed_value = proposed_value
        if edited_schema is not _UNSET:
            if not isinstance(item, (ModuleWriteback, WaveformWriteback)):
                raise RuntimeError(
                    f"{session_id!r} is not a module/waveform item; "
                    "edited_schema invalid"
                )
            item.edited_schema = edited_schema

    def _find_item(self, tab_id: str, session_id: str) -> WritebackItem:
        for item in self._state.get_tab(tab_id).writeback_items:
            if item.session_id == session_id:
                return item
        raise RuntimeError(f"unknown writeback session_id: {session_id!r}")

    # ------------------------------------------------------------------
    # Apply (execute the persistent draft)
    # ------------------------------------------------------------------

    def apply_tab_writeback(self, permit: WritebackPermit) -> list[str]:
        # Context + analyze-result preconditions are proven by the
        # WritebackPermit. ADR-0011: writeback no longer writes ctx.md/ml itself —
        # it collects the selected items into a ContextWrites batch and hands them
        # to the single write authority (ContextService), which lowers + registers
        # + bumps "context" once + emits at most one MD/ML_CHANGED. Writeback only
        # owns the per-tab bookkeeping (applied ids + TAB_CONTENT_CHANGED).
        tab_id = permit.tab_id
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
                ml_modules[item.target_name] = self._item_schema(item)
            elif isinstance(item, WaveformWriteback):
                ml_waveforms[item.target_name] = self._item_schema(item)
            else:
                raise RuntimeError(f"Unsupported writeback item type: {type(item)}")
            applied_ids.append(item.session_id)

        if not (md or ml_modules or ml_waveforms):
            return applied_ids

        self._write.apply_writes(
            ContextWrites(md=md, ml_modules=ml_modules, ml_waveforms=ml_waveforms)
        )
        tab.applied_session_ids.update(applied_ids)
        self._bus.emit(
            GuiEvent.TAB_CONTENT_CHANGED, TabContentChangedPayload(tab_id=tab_id)
        )
        return applied_ids

    def _item_schema(self, item: "ModuleWriteback | WaveformWriteback") -> CfgSchema:
        """The live draft to apply: snapshot the item's editor model if present.

        The persistent draft lives in the service-owned model (``editor_id``); a
        snapshot of it is the authoritative edited schema. Falls back to
        ``edited_schema`` / ``edit_schema`` when there is no model.
        """
        if item.editor_id is not None:
            root = self._cfg_editor.get_root(item.editor_id)
            return CfgSchema(spec=root.spec, value=root.get_value())
        schema = item.edited_schema or item.edit_schema
        if schema is None:
            raise RuntimeError(f"writeback '{item.session_id}' has no editable schema")
        return schema
