from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import TYPE_CHECKING

from zcu_tools.gui.app.main.adapter import LoadDataRequest
from zcu_tools.gui.app.main.events.tab import TabInteractionChangedPayload

from .guard import LoadPermit

if TYPE_CHECKING:
    from zcu_tools.gui.app.main.state import State
    from zcu_tools.gui.event_bus import BaseEventBus as EventBus

    from .ports import WritebackLifecyclePort

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class LoadTabResultOutcome:
    tab_id: str
    data_path: str
    result_type: str
    has_cfg_snapshot: bool
    has_analyze_params: bool
    source_kind: str = "loaded"


class LoadService:
    """Synchronous canonical result load boundary.

    The service owns the adapter call and state replacement only. The Controller
    initializes analyze params afterward so run-finish and load share that policy.
    """

    def __init__(
        self,
        state: State,
        bus: EventBus,
        writeback: WritebackLifecyclePort,
    ) -> None:
        self._state = state
        self._bus = bus
        self._writeback = writeback

    def load_result(self, permit: LoadPermit, data_path: str) -> LoadTabResultOutcome:
        tab_id = permit.tab_id
        if self._state.is_tab_busy(tab_id):
            raise RuntimeError(f"Tab {tab_id!r} is busy")

        tab = self._state.get_tab(tab_id)
        ctx = self._state.exp_context
        request = LoadDataRequest(data_path=data_path, md=ctx.md, ml=ctx.ml)
        logger.info("load_result: tab_id=%r data_path=%r", tab_id, data_path)
        result = tab.adapter.load(request)

        self._writeback.teardown_tab_items(tab_id)
        self._state.update_tab_loaded_result(tab_id, result, data_path)
        self._bus.emit(TabInteractionChangedPayload(tab_id=tab_id))
        return LoadTabResultOutcome(
            tab_id=tab_id,
            data_path=data_path,
            result_type=type(result).__name__,
            has_cfg_snapshot=getattr(result, "cfg_snapshot", None) is not None,
            has_analyze_params=False,
        )
