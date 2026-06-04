"""Shared building blocks for the per-role default factories.

These are the primitives every ``defaults/<role>.py`` uses to assemble a value
tree: in-place field patchers, a trig-offset builder, and re-exports of the
library-lookup selector and the blank-value builder. The role files import only
from here (plus ctx_helpers and the gui specs).
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Union

from zcu_tools.gui.app.main.adapter import (
    CfgSectionValue,
    DirectValue,
    EvalValue,
    ModuleRefValue,
    ScalarValue,
    WaveformRefValue,
    make_default_value,
)

from ..ctx_helpers import md_has_key
from .module_defaults import (
    NamedModuleValue,
    select_named_module_value,
    select_named_waveform_value,
)

if TYPE_CHECKING:
    from zcu_tools.gui.app.main.adapter import ExpContext

__all__ = [
    "make_default_value",
    "select_named_module_value",
    "select_named_waveform_value",
    "NamedModuleValue",
    "patch_pulse_fields",
    "patch_ro_cfg_fields",
    "make_trig_offset",
]


def patch_pulse_fields(
    value: CfgSectionValue,
    *,
    freq: Union[float, ScalarValue],
    ch: Union[int, ScalarValue],
    gain: float,
    length: float,
) -> None:
    """Patch a pulse CfgSectionValue (flat fields) in-place with sensible values."""
    waveform_ref = value.fields.get("waveform")
    if isinstance(waveform_ref, (ModuleRefValue, WaveformRefValue)):
        waveform_ref.value.fields["length"] = DirectValue(length)

    value.fields["ch"] = (
        ch if isinstance(ch, (DirectValue, EvalValue)) else DirectValue(ch)
    )
    value.fields["nqz"] = DirectValue(2)
    value.fields["freq"] = (
        freq if isinstance(freq, (DirectValue, EvalValue)) else DirectValue(freq)
    )
    value.fields["gain"] = DirectValue(gain)


def patch_ro_cfg_fields(
    value: CfgSectionValue,
    *,
    ro_freq: Union[float, ScalarValue],
    ro_ch: Union[int, ScalarValue],
    trig_offset: Union[float, ScalarValue],
    ro_length: Union[float, ScalarValue] = 0.9,
) -> None:
    """Patch a DirectReadout CfgSectionValue in-place with sensible values."""
    value.fields["ro_freq"] = (
        ro_freq
        if isinstance(ro_freq, (DirectValue, EvalValue))
        else DirectValue(ro_freq)
    )
    value.fields["ro_ch"] = (
        ro_ch if isinstance(ro_ch, (DirectValue, EvalValue)) else DirectValue(ro_ch)
    )
    value.fields["ro_length"] = (
        ro_length
        if isinstance(ro_length, (DirectValue, EvalValue))
        else DirectValue(ro_length)
    )
    value.fields["trig_offset"] = (
        trig_offset
        if isinstance(trig_offset, (DirectValue, EvalValue))
        else DirectValue(trig_offset)
    )


def make_trig_offset(
    ctx: ExpContext,
    *,
    trig_expr: str,
    trig_fallback: float,
) -> ScalarValue:
    """Build a trig_offset ScalarValue: EvalValue if timeFly exists, else DirectValue.

    When ``timeFly`` is present the EvalValue carries only ``trig_expr``; lowering
    resolves it against md at render time.
    """
    if md_has_key(ctx, "timeFly"):
        return EvalValue(expr=trig_expr)
    return DirectValue(trig_fallback)
