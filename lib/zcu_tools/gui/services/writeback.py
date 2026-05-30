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
    schema_to_dict,
)
from zcu_tools.gui.event_bus import (
    GuiEvent,
    MdChangedPayload,
    MlChangedPayload,
    TabContentChangedPayload,
)
from zcu_tools.gui.services.guard import WritebackPermit
from zcu_tools.program.v2 import ModuleCfgFactory, WaveformCfgFactory

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from zcu_tools.gui.adapter import ExpContext
    from zcu_tools.gui.event_bus import EventBus
    from zcu_tools.gui.services.cfg_editor import CfgEditorService
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
        self, state: "State", bus: "EventBus", cfg_editor: "CfgEditorService"
    ) -> None:
        self._state = state
        self._bus = bus
        self._cfg_editor = cfg_editor

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
        # WritebackPermit.
        tab_id = permit.tab_id
        tab = self._state.get_tab(tab_id)
        ctx = self._state.exp_context
        applied_ids: list[str] = []
        touched_ml = False
        touched_md = False

        for item in tab.writeback_items:
            if not item.selected:
                continue
            if isinstance(item, MetaDictWriteback):
                setattr(ctx.md, item.target_name, item.proposed_value)
                touched_md = True
            elif isinstance(item, ModuleWriteback):
                module = self._resolve_module_item(item)
                ctx.ml.register_module(**{item.target_name: module})
                touched_ml = True
            elif isinstance(item, WaveformWriteback):
                waveform = self._resolve_waveform_item(item)
                ctx.ml.register_waveform(**{item.target_name: waveform})
                touched_ml = True
            else:
                raise RuntimeError(f"Unsupported writeback item type: {type(item)}")
            applied_ids.append(item.session_id)

        if touched_ml and ctx.ml.has_persistence:
            ctx.ml.dump()
        if touched_md or touched_ml:
            # Writeback writes md/ml directly (it does not go through
            # ContextService), so it must bump the context resource version
            # itself — same semantics as ContextService's md/ml writers — so a
            # later context-dependent op (run / editor.commit / another
            # writeback) detects this change. This is path 2 of 3; the canonical
            # anchor listing all three is on ContextService.set_md_attr.
            self._state.version.bump("context")
        if touched_md:
            self._bus.emit(GuiEvent.MD_CHANGED, MdChangedPayload(md=ctx.md))
        if touched_ml:
            self._bus.emit(GuiEvent.ML_CHANGED, MlChangedPayload(ml=ctx.ml))
        if touched_md or touched_ml:
            tab.applied_session_ids.update(applied_ids)
            self._bus.emit(
                GuiEvent.TAB_CONTENT_CHANGED, TabContentChangedPayload(tab_id=tab_id)
            )
        return applied_ids

    def _resolve_module_item(self, item: ModuleWriteback) -> Any:
        raw = schema_to_dict(
            self._item_schema(item),
            self._state.exp_context.ml,
            self._state.exp_context.md,
        )
        return ModuleCfgFactory.from_raw(raw, ml=self._state.exp_context.ml)

    def _resolve_waveform_item(self, item: WaveformWriteback) -> Any:
        raw = schema_to_dict(
            self._item_schema(item),
            self._state.exp_context.ml,
            self._state.exp_context.md,
        )
        return WaveformCfgFactory.from_raw(raw, ml=self._state.exp_context.ml)

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
