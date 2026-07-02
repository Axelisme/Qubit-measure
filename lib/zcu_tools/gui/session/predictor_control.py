"""Narrow predictor-control facet for shared UI and remote driving adapters."""

from __future__ import annotations

from collections.abc import Callable
from typing import TYPE_CHECKING, Protocol

from zcu_tools.gui.event_bus import BaseEventBus
from zcu_tools.gui.session.events import PredictorChangedPayload

if TYPE_CHECKING:
    from zcu_tools.gui.session.services.predictor import (
        LoadPredictorRequest,
        PredictCurveRequest,
        PredictCurveResult,
        PredictFreqRequest,
        PredictMatrixCurveRequest,
        PredictMatrixCurveResult,
        PredictorService,
        SetModelParamsRequest,
    )


class PredictorControlPort(Protocol):
    """Predictor lifecycle/query/compute surface for shared consumers."""

    def on_predictor_changed(
        self, handler: Callable[[PredictorChangedPayload], None]
    ) -> Callable[[], None]: ...

    def load_predictor(self, req: LoadPredictorRequest) -> None: ...
    def set_predictor_model_params(self, req: SetModelParamsRequest) -> None: ...
    def clear_predictor(self) -> None: ...

    def predict_freq(self, req: PredictFreqRequest) -> float: ...
    def predict_freq_curve(self, req: PredictCurveRequest) -> PredictCurveResult: ...
    def predict_matrix_element_curve(
        self, req: PredictMatrixCurveRequest
    ) -> PredictMatrixCurveResult: ...
    def get_predictor_info(self) -> dict | None: ...


class PredictorControlFacet:
    """Composite adapter over predictor service and event bus."""

    def __init__(
        self,
        *,
        bus: BaseEventBus,
        predictor: PredictorService,
    ) -> None:
        self._bus = bus
        self._predictor = predictor

    def on_predictor_changed(
        self, handler: Callable[[PredictorChangedPayload], None]
    ) -> Callable[[], None]:
        return self._bus.subscribe(PredictorChangedPayload, handler).unsubscribe

    def load_predictor(self, req: LoadPredictorRequest) -> None:
        self._predictor.load_predictor(req)

    def set_predictor_model_params(self, req: SetModelParamsRequest) -> None:
        self._predictor.set_model_params(req)

    def clear_predictor(self) -> None:
        self._predictor.clear_predictor()

    def predict_freq(self, req: PredictFreqRequest) -> float:
        return self._predictor.predict_freq(req)

    def predict_freq_curve(self, req: PredictCurveRequest) -> PredictCurveResult:
        return self._predictor.predict_freq_curve(req)

    def predict_matrix_element_curve(
        self, req: PredictMatrixCurveRequest
    ) -> PredictMatrixCurveResult:
        return self._predictor.predict_matrix_element_curve(req)

    def get_predictor_info(self) -> dict | None:
        return self._predictor.get_predictor_info()
