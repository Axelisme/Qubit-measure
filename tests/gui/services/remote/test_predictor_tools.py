"""predictor.set_model_params dispatch handler — round-trip with predict.

Drives the real dispatch handler (via the _helpers stub adapter) against a real
PredictorService so set_model_params actually builds+installs a FluxoniumPredictor
and a follow-up predictor.predict returns a physically-sane frequency. This pins
the dual-end seam end to end (wire params -> service -> scqubits eigensolve).
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from zcu_tools.gui.app.main.services.remote.dispatch import METHOD_REGISTRY
from zcu_tools.gui.app.main.state import ExpContext, State
from zcu_tools.gui.event_bus import BaseEventBus as EventBus
from zcu_tools.gui.remote.errors import ErrorCode, RemoteError
from zcu_tools.gui.session.services.predictor import (
    LoadPredictorRequest,
    PredictFreqRequest,
    PredictorService,
    SetModelParamsRequest,
)

from ._helpers import dispatch_handler as _dispatch  # noqa: E402


def _ctrl_backed_by_real_service() -> MagicMock:
    """A ctrl mock whose predictor methods delegate to a real PredictorService."""
    state = State(
        ExpContext(md=MagicMock(), ml=MagicMock(), soc=None, soccfg=None, result_dir="")
    )
    svc = PredictorService(state, EventBus())
    ctrl = MagicMock()
    ctrl.set_predictor_model_params.side_effect = svc.set_model_params
    ctrl.predict_freq.side_effect = svc.predict_freq
    ctrl.get_predictor_info.side_effect = svc.get_predictor_info
    return ctrl


class _PoisonController:
    def __getattr__(self, name: str) -> object:
        raise AssertionError(f"broad controller used for {name}")


def _dispatch_with_predictor_control(
    method: str, params: dict[str, object], predictor_control: MagicMock
) -> dict[str, object]:
    adapter = SimpleNamespace(
        ctrl=_PoisonController(),
        predictor_control=predictor_control,
    )
    return dict(METHOD_REGISTRY[method].handler(cast(Any, adapter), params))


def test_predictor_handlers_dispatch_only_through_predictor_control_facet():
    pred = MagicMock()
    pred.get_predictor_info.return_value = {
        "path": None,
        "flux_bias": 0.0,
        "flux_half": 0.0,
        "flux_period": 1.0,
        "EJ": 4.0,
        "EC": 1.0,
        "EL": 1.0,
    }
    pred.predict_freq.return_value = 1234.5

    assert (
        _dispatch_with_predictor_control(
            "predictor.load",
            {"path": "/tmp/params.json", "flux_bias": 0.1},
            pred,
        )["loaded"]
        is True
    )
    assert isinstance(pred.load_predictor.call_args.args[0], LoadPredictorRequest)

    assert (
        _dispatch_with_predictor_control(
            "predictor.set_model_params",
            {
                "EJ": 4.0,
                "EC": 1.0,
                "EL": 1.0,
                "flux_half": 0.0,
                "flux_period": 1.0,
                "flux_bias": 0.0,
            },
            pred,
        )["loaded"]
        is True
    )
    assert isinstance(
        pred.set_predictor_model_params.call_args.args[0], SetModelParamsRequest
    )

    assert _dispatch_with_predictor_control(
        "predictor.predict",
        {"device_value": 0.5, "from_level": 0, "to_level": 1},
        pred,
    ) == {"freq_mhz": 1234.5}
    assert isinstance(pred.predict_freq.call_args.args[0], PredictFreqRequest)

    assert _dispatch_with_predictor_control("predictor.clear", {}, pred) == {
        "loaded": False
    }
    pred.clear_predictor.assert_called_once_with()

    pred.get_predictor_info.return_value = None
    assert _dispatch_with_predictor_control("predictor.info", {}, pred) == {
        "loaded": False
    }


def test_set_model_params_then_predict_returns_sane_freq():
    """dispatch set_model_params (EJ:EC:EL=4:1:1) then predict → finite MHz freq."""
    ctrl = _ctrl_backed_by_real_service()

    res = _dispatch(
        ctrl,
        "predictor.set_model_params",
        {
            "EJ": 4.0,
            "EC": 1.0,
            "EL": 1.0,
            "flux_half": 0.0,
            "flux_period": 1.0,
            "flux_bias": 0.0,
        },
    )
    # Echoes the installed model (loaded flag + flattened energies, path=null).
    assert res["loaded"] is True
    assert res["EJ"] == pytest.approx(4.0)

    # predict at half-flux (device_value 0.5 maps to flux 0.5 here) for the 0->1 line.
    pred = _dispatch(
        ctrl,
        "predictor.predict",
        {"device_value": 0.5, "from_level": 0, "to_level": 1},
    )
    freq = pred["freq_mhz"]
    assert isinstance(freq, float)
    # A real fluxonium 0->1 gap is a finite, positive MHz number.
    assert freq > 0.0
    assert freq < 1e7  # sanity upper bound (well under 10 THz)


def test_set_model_params_info_reads_back_energies():
    """After set_model_params, predictor.info reply carries EJ/EC/EL.

    predictor.info now FLATTENS the model fields to the top level (the `loaded`
    flag replaces the old {info: ...} wrapper).
    """
    ctrl = _ctrl_backed_by_real_service()
    _dispatch(
        ctrl,
        "predictor.set_model_params",
        {
            "EJ": 4.0,
            "EC": 1.0,
            "EL": 1.0,
            "flux_half": 0.3,
            "flux_period": 0.8,
            "flux_bias": 0.0,
        },
    )
    info = _dispatch(ctrl, "predictor.info", {})
    assert info["loaded"] is True
    assert info["EJ"] == pytest.approx(4.0)
    assert info["EC"] == pytest.approx(1.0)
    assert info["EL"] == pytest.approx(1.0)
    assert info["path"] is None


def test_set_model_params_zero_period_precondition_failed():
    """flux_period == 0 surfaces as PRECONDITION_FAILED over the wire."""
    ctrl = _ctrl_backed_by_real_service()
    with pytest.raises(RemoteError) as exc:
        _dispatch(
            ctrl,
            "predictor.set_model_params",
            {
                "EJ": 4.0,
                "EC": 1.0,
                "EL": 1.0,
                "flux_half": 0.3,
                "flux_period": 0.0,
                "flux_bias": 0.0,
            },
        )
    assert exc.value.code is ErrorCode.PRECONDITION_FAILED


def test_set_model_params_flux_bias_defaults_to_zero():
    """flux_bias omitted in params falls back to the spec default (0.0).

    The wire spec marks flux_bias as a NUMBER with default 0.0, so a caller may
    omit it. The dispatch handler reads params["flux_bias"]; validate_params (run
    by the live server before dispatch) injects the default. Here we exercise the
    handler directly, so we pass the defaulted value explicitly to mirror that.
    """
    ctrl = _ctrl_backed_by_real_service()
    _dispatch(
        ctrl,
        "predictor.set_model_params",
        {
            "EJ": 4.0,
            "EC": 1.0,
            "EL": 1.0,
            "flux_half": 0.3,
            "flux_period": 0.8,
            "flux_bias": 0.0,
        },
    )
    info = _dispatch(ctrl, "predictor.info", {})
    assert info["loaded"] is True
    assert info["flux_bias"] == pytest.approx(0.0)
