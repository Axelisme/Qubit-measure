from __future__ import annotations

import dataclasses
from typing import TYPE_CHECKING, Any, Optional

if TYPE_CHECKING:
    from zcu_tools.gui.event_bus import EventBus
    from zcu_tools.gui.state import State


class ConnectionService:
    """Encapsulates SOC connection and Predictor settings."""

    def __init__(self, state: "State", bus: Optional["EventBus"] = None) -> None:
        self._state = state
        self._bus = bus
        self._predictor_path: Optional[str] = None

    def has_soc(self) -> bool:
        return self._state.exp_context.soc is not None

    def get_soccfg(self) -> Any:
        return self._state.exp_context.soccfg

    def set_connection(self, soc: Any, soccfg: Any) -> None:
        new_ctx = dataclasses.replace(self._state.exp_context, soc=soc, soccfg=soccfg)
        self._state.set_context(new_ctx)

    def set_predictor(
        self, predictor: Optional[Any], path: Optional[str] = None
    ) -> None:
        self._predictor_path = path
        new_ctx = dataclasses.replace(self._state.exp_context, predictor=predictor)
        self._state.set_context(new_ctx)

    def get_predictor(self) -> Optional[Any]:
        return self._state.exp_context.predictor

    def get_predictor_info(self) -> Optional[dict]:
        predictor = self._state.exp_context.predictor
        if predictor is None:
            return None
        return {"path": self._predictor_path, "flux_bias": predictor.flux_bias}
