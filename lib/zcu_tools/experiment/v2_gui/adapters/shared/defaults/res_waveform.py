"""Default factories for the ``res_waveform`` role (resonator-pulse waveform).

Mirror of qub_waveform for the resonator side. The ref prefers res library
waveforms (res_flat / res_const); the blank is a plain const.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from typing_extensions import Literal, Optional, overload

from zcu_tools.gui.adapter import DirectValue, WaveformRefValue
from zcu_tools.gui.specs.waveform import make_const_waveform_spec

from .helpers import make_default_value
from .qub_waveform import _select_waveform

if TYPE_CHECKING:
    from zcu_tools.gui.adapter import ExpContext

RES_WAVEFORM_NAMES = ["res_flat", "res_const"]


def make_res_waveform_default(ctx: ExpContext) -> WaveformRefValue:  # noqa: ARG001
    """Blank resonator-pulse waveform (a plain const), no library lookup.

    The caller sets ``length`` via ``.with_field("length", ...)`` when needed.
    """
    value = make_default_value(make_const_waveform_spec())
    value.fields["length"] = DirectValue(1.0)
    return WaveformRefValue(chosen_key="<Custom:Const>", value=value)


@overload
def make_res_waveform_ref_default(
    ctx: ExpContext,
    preferred_names: list[str] = ...,
    *,
    optional: Literal[False] = ...,
) -> WaveformRefValue: ...


@overload
def make_res_waveform_ref_default(
    ctx: ExpContext,
    preferred_names: list[str] = ...,
    *,
    optional: Literal[True],
) -> Optional[WaveformRefValue]: ...


def make_res_waveform_ref_default(
    ctx: ExpContext,
    preferred_names: list[str] = RES_WAVEFORM_NAMES,
    *,
    optional: bool = False,
) -> Optional[WaveformRefValue]:
    """Reference a library resonator waveform (res_flat / res_const), else blank."""
    selected = _select_waveform(ctx, preferred_names)
    if selected is not None:
        return selected
    if optional:
        return None
    return make_res_waveform_default(ctx)
