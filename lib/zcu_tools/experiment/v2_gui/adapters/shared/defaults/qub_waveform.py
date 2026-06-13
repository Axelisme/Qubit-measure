"""Default factories for the ``qub_waveform`` role (qubit-pulse waveform).

Produces a WaveformRefValue for a pulse's ``waveform`` sub-field. The ref prefers
library waveforms (qub_flat / qub_cos); the blank is a plain cosine.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload

from zcu_tools.gui.app.main.adapter import (
    DirectValue,
    WaveformRefValue,
)
from zcu_tools.gui.app.main.specs.waveform import make_cosine_waveform_spec

from .helpers import make_default_value, select_named_waveform_value

if TYPE_CHECKING:
    from zcu_tools.gui.app.main.adapter import ExpContext

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
) -> WaveformRefValue | None: ...


def make_qub_waveform_ref_default(
    ctx: ExpContext,
    preferred_names: list[str] = QUB_WAVEFORM_NAMES,
    *,
    optional: bool = False,
) -> WaveformRefValue | None:
    """Reference a library qubit waveform (qub_flat / qub_cos), else blank cosine."""
    selected = select_named_waveform_value(ctx.ml, preferred_names)
    if selected is not None:
        return selected
    if optional:
        return None  # optional ref disabled (ADR-0010)
    return make_qub_waveform_default(ctx)
