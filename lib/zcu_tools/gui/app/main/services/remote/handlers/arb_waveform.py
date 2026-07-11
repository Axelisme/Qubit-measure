"""Arb Waveform remote handlers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, NoReturn

from zcu_tools.gui.remote.errors import ErrorCode, RemoteError
from zcu_tools.gui.remote.wire import optional_bool, require_str
from zcu_tools.meta_tool import ArbWaveformError

if TYPE_CHECKING:
    from ..service import RemoteControlAdapter


_ARB_INVALID_PARAM_REASONS = frozenset(
    {
        "invalid_data_key",
        "invalid_recipe",
        "invalid_recipe_json",
        "formula_parse_failed",
        "formula_unsafe",
        "formula_unknown_symbol",
        "formula_conditional_not_supported",
        "formula_evaluation_failed",
        "formula_not_numeric",
        "formula_shape_mismatch",
        "formula_non_finite",
        "amplitude_out_of_range",
        "sample_count_too_small",
        "sample_count_too_large",
        "data_key_not_found",
    }
)


def _raise_arb_waveform_error(exc: ArbWaveformError) -> NoReturn:
    code = (
        ErrorCode.INVALID_PARAMS
        if exc.reason in _ARB_INVALID_PARAM_REASONS
        else ErrorCode.PRECONDITION_FAILED
    )
    raise RemoteError(code, str(exc), reason=exc.reason, data=exc.data) from exc


def _h_arb_waveform_list(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    return {"waveforms": adapter.ctrl.list_arb_waveforms()}


def _h_arb_waveform_preview(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    name = require_str(params, "name")
    try:
        return adapter.ctrl.get_arb_waveform_preview(name)
    except ArbWaveformError as exc:
        _raise_arb_waveform_error(exc)


def _h_arb_waveform_set(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    name = require_str(params, "name")
    overwrite = optional_bool(params, "overwrite", False)
    recipe = params["recipe"]
    try:
        result = dict(adapter.ctrl.set_arb_waveform(name, recipe, overwrite=overwrite))
    except ArbWaveformError as exc:
        _raise_arb_waveform_error(exc)
    result["success"] = True
    return result
