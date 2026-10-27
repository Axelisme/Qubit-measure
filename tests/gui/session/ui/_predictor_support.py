"""Device cache collaborator for detached predictor UI trials."""

from unittest.mock import MagicMock

import numpy as np
from zcu_tools.analysis.flux import flux_from_device_values
from zcu_tools.device.fake import FakeDeviceInfo
from zcu_tools.gui.session.device_models import DeviceEntry
from zcu_tools.gui.session.services.predictor import (
    PredictCurveRequest,
    PredictCurveResult,
    PredictFreqRequest,
    PredictMatrixCurveRequest,
    PredictMatrixCurveResult,
    PredictorService,
)
from zcu_tools.simulate.predictor import PredictorParams


def make_device_control(
    *,
    entries: list[DeviceEntry] | None = None,
    cached: dict[str, float | None] | None = None,
    units: dict[str, str] | None = None,
    live: dict[str, FakeDeviceInfo | None] | None = None,
) -> MagicMock:
    ctrl = MagicMock()
    value_cache = cached if cached is not None else {}
    unit_map = units if units is not None else {}
    live_map = live if live is not None else {}
    ctrl.list_devices.return_value = list(entries or [])
    ctrl.get_cached_device_value.side_effect = lambda name: value_cache.get(name)
    ctrl.get_device_unit.side_effect = lambda name: unit_map.get(name, "none")
    ctrl.get_device_info.side_effect = lambda name: live_map.get(name)
    ctrl.try_poll_device_info.return_value = True
    ctrl.on_device_changed.return_value = MagicMock(name="unsubscribe_device")
    return ctrl


class SyntheticCurves(PredictorService):
    """Supply deterministic visible curves without numerical solvers or devices."""

    def predict_freq_curve(
        self, params: PredictorParams, req: PredictCurveRequest
    ) -> PredictCurveResult:
        labels = tuple(f"{frm}→{to}" for frm, to in req.transitions)
        series = np.array(
            [1000 * (to - frm) + 100 * req.values for frm, to in req.transitions]
        )
        return PredictCurveResult(labels, req.values, req.values, series)

    def predict_matrix_element_curve(
        self, params: PredictorParams, req: PredictMatrixCurveRequest
    ) -> PredictMatrixCurveResult:
        labels = tuple(f"{frm}→{to}" for frm, to in req.transitions)
        scale = 0.1 if req.operator == "n" else 0.2
        series = np.array(
            [scale * (to - frm) + 0.01 * req.values for frm, to in req.transitions]
        )
        return PredictMatrixCurveResult(labels, req.values, req.values, series)

    def predict_freq(self, params: PredictorParams, req: PredictFreqRequest) -> float:
        return 1000 * (
            req.transition[1] - req.transition[0]
        ) + 100 * flux_from_device_values(req.frame, req.device_values)
