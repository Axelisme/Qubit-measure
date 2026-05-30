"""Default factories for the ``qub_waveform`` role (qubit-pulse waveform).

Produces a WaveformRefValue for a pulse's ``waveform`` sub-field. The ref prefers
library waveforms (qub_flat / qub_cos); the blank is a plain cosine.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from typing_extensions import Literal, Optional, overload

from zcu_tools.gui.adapter import DirectValue, WaveformRefValue
from zcu_tools.gui.specs.waveform import make_cosine_waveform_spec

from .helpers import make_default_value

if TYPE_CHECKING:
    from zcu_tools.gui.adapter import ExpContext

QUB_WAVEFORM_NAMES = ["qub_flat", "qub_cos"]


def make_qub_waveform_default(
    ctx: ExpContext, *, length: float = 0.1
) -> WaveformRefValue:  # noqa: ARG001
    """Blank qubit-pulse waveform (a plain cosine), no library lookup."""
    value = make_default_value(make_cosine_waveform_spec())
    value.fields["length"] = DirectValue(length)
    return WaveformRefValue(chosen_key="<Custom:Cosine>", value=value)


def _select_waveform(ctx: ExpContext, preferred_names: list[str]):
    from zcu_tools.gui.cfg_schemas import waveform_cfg_to_value

    waveforms = getattr(ctx.ml, "waveforms", {})
    for name in preferred_names:
        if name in waveforms:
            _, wav_val = waveform_cfg_to_value(waveforms[name])
            return WaveformRefValue(chosen_key=name, value=wav_val)
    return None


@overload
def make_qub_waveform_ref_default(
    ctx: ExpContext,
    preferred_names: list[str] = ...,
    *,
    optional: Literal[False] = ...,
) -> WaveformRefValue: ...


@overload
def make_qub_waveform_ref_default(
    ctx: ExpContext,
    preferred_names: list[str] = ...,
    *,
    optional: Literal[True],
) -> Optional[WaveformRefValue]: ...


def make_qub_waveform_ref_default(
    ctx: ExpContext,
    preferred_names: list[str] = QUB_WAVEFORM_NAMES,
    *,
    optional: bool = False,
) -> Optional[WaveformRefValue]:
    """Reference a library qubit waveform (qub_flat / qub_cos), else blank cosine."""
    selected = _select_waveform(ctx, preferred_names)
    if selected is not None:
        return selected
    if optional:
        return None
    return make_qub_waveform_default(ctx)
