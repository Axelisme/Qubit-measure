from __future__ import annotations

import logging
from typing import TYPE_CHECKING, Any

from zcu_tools.gui.adapter import (
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
    from zcu_tools.gui.event_bus import EventBus
    from zcu_tools.gui.state import State


class WritebackService:
    """Encapsulates writeback proposal retrieval and persistence."""

    def __init__(self, state: "State", bus: "EventBus") -> None:
        self._state = state
        self._bus = bus

    def get_tab_writeback_items(self, tab_id: str) -> list[WritebackItem]:
        tab = self._state.get_tab(tab_id)
        # Direct None-checks here (not the has_* predicates): the values are used
        # immediately below, so the checker must narrow them to non-None.
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
        for item in items:
            item.selected = item.key not in tab.applied_writeback_keys
        return items

    def apply_tab_writeback_items(
        self,
        permit: WritebackPermit,
        items: list[WritebackItem],
    ) -> list[str]:
        # Context + analyze-result preconditions are proven by the
        # WritebackPermit.
        tab_id = permit.tab_id
        tab = self._state.get_tab(tab_id)
        ctx = self._state.exp_context
        applied_keys: list[str] = []
        touched_ml = False
        touched_md = False

        for item in items:
            if not item.selected:
                continue
            if isinstance(item, MetaDictWriteback):
                setattr(ctx.md, item.md_key, item.proposed_value)
                touched_md = True
            elif isinstance(item, ModuleWriteback):
                module = self._resolve_module_item(item)
                ctx.ml.register_module(**{item.module_name: module})
                touched_ml = True
            elif isinstance(item, WaveformWriteback):
                waveform = self._resolve_waveform_item(item)
                ctx.ml.register_waveform(**{item.waveform_name: waveform})
                touched_ml = True
            else:
                raise RuntimeError(f"Unsupported writeback item type: {type(item)}")
            applied_keys.append(item.key)

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
            tab.applied_writeback_keys.update(applied_keys)
            self._bus.emit(
                GuiEvent.TAB_CONTENT_CHANGED, TabContentChangedPayload(tab_id=tab_id)
            )
        return applied_keys

    def _resolve_module_item(self, item: ModuleWriteback) -> Any:
        if item.edited_schema is not None:
            raw = schema_to_dict(
                item.edited_schema,
                self._state.exp_context.ml,
                self._state.exp_context.md,
            )
            return ModuleCfgFactory.from_raw(raw, ml=self._state.exp_context.ml)
        if item.proposed_module is not None:
            return item.proposed_module
        raise RuntimeError(f"Module writeback '{item.key}' has no proposal to apply")

    def _resolve_waveform_item(self, item: WaveformWriteback) -> Any:
        if item.edited_schema is not None:
            raw = schema_to_dict(
                item.edited_schema,
                self._state.exp_context.ml,
                self._state.exp_context.md,
            )
            return WaveformCfgFactory.from_raw(raw, ml=self._state.exp_context.ml)
        if item.proposed_waveform is not None:
            return item.proposed_waveform
        raise RuntimeError(f"Waveform writeback '{item.key}' has no proposal to apply")
