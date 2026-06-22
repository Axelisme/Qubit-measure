from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from zcu_tools.gui.app.main.adapter import (
    CfgSchema,
    MetaDictWriteback,
    ModuleWriteback,
    WaveformWriteback,
    WritebackItem,
    WritebackRequest,
)
from zcu_tools.gui.app.main.events.tab import TabContentChangedPayload

from .guard import WritebackPermit
from .ports import CfgEditorPort, ContextWrites

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from zcu_tools.gui.app.main.state import State
    from zcu_tools.gui.event_bus import BaseEventBus as EventBus

    from .ports import ContextWritePort

# kind prefix for the stable per-item session_id (``<kind>-<n>``).
_KIND_PREFIX = {
    MetaDictWriteback: "md",
    ModuleWriteback: "ml",
    WaveformWriteback: "wf",
}

# Sentinel for "argument not supplied" in set_item_field (None is a real value).
_UNSET: Any = object()


class WritebackService:
    """Owns the persistent writeback draft for each tab (ADR-0008).

    Writeback items are computed **once** when analyze finishes (``compute_items``)
    and stored on ``Session.writeback_items``. Each module/waveform item gets a
    gc=False cfg-editor model (seeded from its ``edit_schema``); the agent edits
    it via this service's ``set_item_field(edits=...)`` (which writes through the
    item's ``editor_id`` on ``CfgEditorPort``, ADR-0008) and the user's Edit
    dialog attaches to the same model (WYSIWYG). ``get_tab_writeback_items`` is a
    pure read of that persistent list — it never recomputes (which would discard
    live edits).
    """

    def __init__(
        self,
        state: State,
        bus: EventBus,
        cfg_editor: CfgEditorPort,
        write_port: ContextWritePort,
    ) -> None:
        self._state = state
        self._bus = bus
        self._cfg_editor = cfg_editor
        self._write = write_port

    # ------------------------------------------------------------------
    # Compute once (analyze sink) + read
    # ------------------------------------------------------------------

    def compute_items_for_tab(
        self, tab_id: str, analyze_result: Any
    ) -> list[WritebackItem]:
        """Compute the tab's writeback items once (analyze sink calls this).

        The fresh ``analyze_result`` is passed in explicitly (not read from
        State): the analyze sink computes the draft *before* committing the
        result through ``update_tab_analyze``, so State must not be written
        early just to make this readable. Calls the adapter, stamps a stable
        per-kind ``session_id``, and for each module/waveform item opens a
        gc=False CfgEditorService model seeded from its ``edit_schema`` (storing
        the ``editor_id``). The returned list is stored on
        ``Session.writeback_items`` by the analyze sink.
        """
        tab = self._state.get_tab(tab_id)
        run_result = tab.run_result
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
    # Edit a persistent item (agent / UI tab.writeback_set)
    # ------------------------------------------------------------------

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
        """Edit a persistent writeback item by id.

        ``selected`` / ``target_name`` apply to any item. ``proposed_value`` is a
        metadict-only facet (the proposed md scalar). ``edits`` is the
        module/waveform-only facet: an ORDERED list of ``{path, value}`` cfg edits
        applied to the item's service-owned editor model via
        ``CfgEditorPort.set_field`` — the agent never sees the ``editor_id``
        (ADR-0008). Returns the aggregated ``{valid, removed, added}`` of the
        applied edits (empty lists / valid=True when no ``edits`` are given), the
        same shape as tab cfg's batch set. Edits are fail-fast and non-atomic:
        a failing edit raises and earlier edits in the batch stay applied.
        """
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

        valid = True
        removed: list[str] = []
        added: list[str] = []
        if edits is not None:
            if not isinstance(item, (ModuleWriteback, WaveformWriteback)):
                raise RuntimeError(
                    f"{session_id!r} is not a module/waveform item; edits invalid"
                )
            if item.editor_id is None:
                raise RuntimeError(
                    f"{session_id!r} has no editable cfg model to apply edits to"
                )
            for i, edit in enumerate(edits):
                if "path" not in edit or "value" not in edit:
                    raise RuntimeError(
                        f"edits[{i}] must be an object with 'path' and 'value'"
                    )
                result = self._cfg_editor.set_field(
                    item.editor_id, str(edit["path"]), edit["value"]
                )
                valid = bool(result.get("valid", True))
                r = result.get("removed", [])
                a = result.get("added", [])
                if isinstance(r, list):
                    removed.extend(str(p) for p in r)
                if isinstance(a, list):
                    added.extend(str(p) for p in a)
        return {"valid": valid, "removed": removed, "added": added}

    def _find_item(self, tab_id: str, session_id: str) -> WritebackItem:
        for item in self._state.get_tab(tab_id).writeback_items:
            if item.session_id == session_id:
                return item
        raise RuntimeError(f"unknown writeback session_id: {session_id!r}")

    # ------------------------------------------------------------------
    # Apply (execute the persistent draft)
    # ------------------------------------------------------------------

    def apply_tab_writeback(self, permit: WritebackPermit) -> dict[str, Any]:
        """Apply the tab's persistent draft and echo what was actually written.

        Returns ``{applied_ids, written}`` where ``written`` lists the destination
        names actually pushed to context, split by kind
        (``{md, ml_modules, ml_waveforms}``). All lists are empty on a no-op draft
        (nothing selected); ``applied_ids`` still reflects the selected item ids.
        """
        # Context + analyze-result preconditions are proven by the
        # WritebackPermit. ADR-0006: writeback no longer writes ctx.md/ml itself —
        # it collects the selected items into a ContextWrites batch and hands them
        # to the single write authority (ContextService), which lowers + registers
        # + bumps "context" once + emits at most one MD/ML_CHANGED. Writeback only
        # owns the per-tab bookkeeping (applied ids + TAB_CONTENT_CHANGED).
        tab_id = permit.tab_id
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
                ml_modules[item.target_name] = self._item_schema(item)
            elif isinstance(item, WaveformWriteback):
                ml_waveforms[item.target_name] = self._item_schema(item)
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
        tab.applied_session_ids.update(applied_ids)
        self._bus.emit(TabContentChangedPayload(tab_id=tab_id))
        logger.info(
            "writeback applied: tab_id=%r md=%d ml_modules=%d ml_waveforms=%d",
            tab_id,
            len(md),
            len(ml_modules),
            len(ml_waveforms),
        )
        return {"applied_ids": applied_ids, "written": written}

    def _item_schema(self, item: ModuleWriteback | WaveformWriteback) -> CfgSchema:
        """The live draft to apply: snapshot the item's editor model if present.

        The persistent draft lives in the service-owned model (``editor_id``); a
        snapshot of it is the authoritative edited schema. Falls back to
        ``edit_schema`` when there is no model.
        """
        if item.editor_id is not None:
            root = self._cfg_editor.get_root(item.editor_id)
            return CfgSchema(spec=root.spec, value=root.get_value())
        schema = item.edit_schema
        if schema is None:
            raise RuntimeError(f"writeback '{item.session_id}' has no editable schema")
        return schema
