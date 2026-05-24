from __future__ import annotations

import dataclasses
import logging
from typing import TYPE_CHECKING, Optional

logger = logging.getLogger(__name__)

from zcu_tools.gui.adapter import SocCfgHandle, SocHandle
from zcu_tools.gui.event_bus import GuiEvent, PredictorChangedPayload, SocChangedPayload
from zcu_tools.simulate.fluxonium.predict import FluxoniumPredictor

if TYPE_CHECKING:
    from zcu_tools.gui.event_bus import EventBus
    from zcu_tools.gui.state import State


class ConnectionService:
    """Encapsulates SOC connection and Predictor settings."""

    def __init__(self, state: "State", bus: "EventBus") -> None:
        self._state = state
        self._bus = bus
        self._predictor_path: Optional[str] = None

    def has_soc(self) -> bool:
        return self._state.exp_context.soc is not None

    def get_soccfg(self) -> Optional[SocCfgHandle]:
        return self._state.exp_context.soccfg

    def set_connection(
        self, soc: Optional[SocHandle], soccfg: Optional[SocCfgHandle]
    ) -> None:
        logger.info(
            "set_connection: soc=%s soccfg=%s",
            type(soc).__name__,
            type(soccfg).__name__,
        )
        new_ctx = dataclasses.replace(self._state.exp_context, soc=soc, soccfg=soccfg)
        self._state.set_context(new_ctx)
        self._bus.emit(GuiEvent.SOC_CHANGED, SocChangedPayload(soc=soc, soccfg=soccfg))

    def set_predictor(
        self, predictor: Optional[FluxoniumPredictor], path: Optional[str] = None
    ) -> None:
        logger.info(
            "set_predictor: path=%r predictor=%s", path, type(predictor).__name__
        )
        self._predictor_path = path
        new_ctx = dataclasses.replace(self._state.exp_context, predictor=predictor)
        self._state.set_context(new_ctx)
        self._bus.emit(GuiEvent.PREDICTOR_CHANGED, PredictorChangedPayload())

    def get_predictor(self) -> Optional[FluxoniumPredictor]:
        return self._state.exp_context.predictor

    def get_predictor_info(self) -> Optional[dict]:
        predictor = self._state.exp_context.predictor
        if predictor is None:
            return None
        return {"path": self._predictor_path, "flux_bias": predictor.flux_bias}
