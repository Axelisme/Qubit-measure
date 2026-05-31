"""Default factories for the ``qub_waveform`` role (qubit-pulse waveform).

Produces a WaveformRefValue for a pulse's ``waveform`` sub-field. The ref prefers
library waveforms (qub_flat / qub_cos); the blank is a plain cosine.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from typing_extensions import Literal, Union, overload

from zcu_tools.gui.adapter import DirectValue, DisabledRefValue, WaveformRefValue
from zcu_tools.gui.specs.waveform import make_cosine_waveform_spec

from .helpers import make_default_value, select_named_waveform_value

if TYPE_CHECKING:
    from zcu_tools.gui.adapter import ExpContext

QUB_WAVEFORM_NAMES = ["qub_flat", "qub_cos"]


def make_qub_waveform_default(ctx: ExpContext) -> WaveformRefValue:  # noqa: ARG001
    """Blank qubit-pulse waveform (a plain cosine), no library lookup.

    The caller sets ``length`` via ``.with_field("length", ...)`` when needed.
    """
    value = make_default_value(make_cosine_waveform_spec())
    value.fields["length"] = DirectValue(0.1)
    return WaveformRefValue(chosen_key="<Custom:Cosine>", value=value)


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
) -> Union[WaveformRefValue, DisabledRefValue]: ...


def make_qub_waveform_ref_default(
    ctx: ExpContext,
    preferred_names: list[str] = QUB_WAVEFORM_NAMES,
    *,
    optional: bool = False,
) -> Union[WaveformRefValue, DisabledRefValue]:
    """Reference a library qubit waveform (qub_flat / qub_cos), else blank cosine."""
    selected = select_named_waveform_value(ctx.ml, preferred_names)
    if selected is not None:
        return selected
    if optional:
        return DisabledRefValue()  # ADR-0012: present-but-disabled marker
    return make_qub_waveform_default(ctx)
