"""Arb Waveform remote handlers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import TYPE_CHECKING, NoReturn

from zcu_tools.gui.remote.errors import ErrorCode, RemoteError
from zcu_tools.gui.remote.wire import optional_bool, require_str
from zcu_tools.resources.waveform_assets import ArbWaveformError, FormulaRecipe

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


def h_arb_waveform_list(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    del params
    return {"waveforms": adapter.ctrl.arb_waveforms.list_data_keys()}


def h_arb_waveform_preview(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    name = require_str(params, "name")
    try:
        preview = adapter.ctrl.arb_waveforms.get_preview(name)
        return {
            "recipe": preview.recipe.to_dict() if preview.recipe is not None else None,
            "preview_figure": preview.figure_path,
        }
    except ArbWaveformError as exc:
        _raise_arb_waveform_error(exc)


def h_arb_waveform_set(
    adapter: RemoteControlAdapter, params: Mapping[str, object]
) -> Mapping[str, object]:
    name = require_str(params, "name")
    overwrite = optional_bool(params, "overwrite", False)
    try:
        recipe = FormulaRecipe.from_raw(params["recipe"])
        status = adapter.ctrl.arb_waveforms.set_formula(
            name, recipe, overwrite=overwrite
        )
    except ArbWaveformError as exc:
        _raise_arb_waveform_error(exc)
    return {"success": True, "status": status}
