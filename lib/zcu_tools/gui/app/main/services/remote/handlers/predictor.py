"""Predictor remote handlers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING

from zcu_tools.gui.remote.errors import ErrorCode, RemoteError

if TYPE_CHECKING:
    from ..service import RemoteControlAdapter


def _h_predictor_load(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    from zcu_tools.gui.session.services.predictor import LoadPredictorRequest

    path = str(params["path"])
    flux_bias = float(params["flux_bias"])  # type: ignore[arg-type]
    adapter.predictor_control.load_predictor(
        LoadPredictorRequest(path=path, flux_bias=flux_bias)
    )
    # Echo the installed model so the agent verifies the load without a follow-up
    # read; get_predictor_info() is non-None right after a successful install (a
    # None here is a broken invariant, so raise rather than echo a half-shape).
    info = adapter.predictor_control.get_predictor_info()
    if info is None:
        raise RuntimeError("predictor missing immediately after a successful load")
    return {"loaded": True, **info}


def _h_predictor_set_model_params(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    from zcu_tools.gui.session.services.predictor import SetModelParamsRequest

    req = SetModelParamsRequest(
        EJ=float(params["EJ"]),  # type: ignore[arg-type]
        EC=float(params["EC"]),  # type: ignore[arg-type]
        EL=float(params["EL"]),  # type: ignore[arg-type]
        flux_half=float(params["flux_half"]),  # type: ignore[arg-type]
        flux_period=float(params["flux_period"]),  # type: ignore[arg-type]
        flux_bias=float(params["flux_bias"]),  # type: ignore[arg-type]
    )
    adapter.predictor_control.set_predictor_model_params(req)
    # Echo the installed model (path is null — in-memory install has no file); a
    # None right after a successful install is a broken invariant, so raise.
    info = adapter.predictor_control.get_predictor_info()
    if info is None:
        raise RuntimeError("predictor missing immediately after a successful install")
    return {"loaded": True, **info}


def _h_predictor_clear(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    adapter.predictor_control.clear_predictor()
    return {"loaded": False}


def _h_predictor_predict(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    from zcu_tools.gui.session.services.predictor import PredictFreqRequest

    device_value = float(params["device_value"])  # type: ignore[arg-type]
    from_level = int(params["from_level"])  # type: ignore[arg-type]
    to_level = int(params["to_level"])  # type: ignore[arg-type]
    freq = adapter.predictor_control.predict_freq(
        PredictFreqRequest(value=device_value, transition=(from_level, to_level))
    )
    return {"freq_mhz": freq}


def _h_predictor_info(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    # Flatten the model fields to the top level; the `loaded` flag replaces a null
    # payload so the agent never has to distinguish {info: null} from a real read.
    info = adapter.predictor_control.get_predictor_info()
    if info is None:
        return {"loaded": False}
    return {"loaded": True, **info}
