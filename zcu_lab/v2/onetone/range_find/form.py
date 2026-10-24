"""Pulse-readout initial values for a wide absolute-frequency survey."""

from collections.abc import Sequence
from math import isfinite

from zcu_tools.cfg_model import Range
from zcu_tools.gui.form_adapter import Form, FormContext
from zcu_tools.gui.session.types import ParameterContext

from zcu_lab.v2.onetone._form_shared import initial_spectrum_tree
from zcu_lab.v2.support.measure.pulse_readout_metadata import (
    pulse_readout_seed_metadata,
)

from .core import RangeFindCfg


def resonator_frequencies(
    parameters: ParameterContext, names: Sequence[str]
) -> list[float] | None:
    """Read captured MHz in name order, or None if any frequency is absent/null.

    parameters is the fixed Open/Reset or Run/Load capture. Every set frequency
    must be finite numeric MHz (not bool), even if another name is missing;
    invalid values raise ValueError and source errors propagate unchanged.
    No sorting, live lookup or source mutation occurs.
    """
    frequencies: list[float] = []
    missing_frequency = False
    for name in names:
        source = parameters.lookup_point(f"{name}.freq")
        value = source.value
        if not source.present or value is None:
            missing_frequency = True
            continue
        if (
            isinstance(value, bool)
            or not isinstance(value, (int, float))
            or not isfinite(value)
        ):
            raise ValueError(f"{name}.freq must be finite MHz to seed resonator order")
        frequencies.append(float(value))
    return None if missing_frequency else frequencies


def make_form(ctx: FormContext) -> Form[RangeFindCfg]:
    """Leave sweep endpoints blank with 301 points.

    ctx is the Open/Reset capture. The operator must enter absolute MHz endpoints.
    Other defaults match freq linear. Dip naming is a later Analyze choice.
    Timing failures propagate; no hardware I/O or live refresh occurs here.
    """
    return Form(
        RangeFindCfg,
        field_order=(
            "reset",
            "readout",
            "freq",
            "reps",
            "rounds",
            "relax_delay",
        ),
        tree={**initial_spectrum_tree(ctx), "freq": Range(None, None, expts=301)},
    ).with_seed_metadata(pulse_readout_seed_metadata(gain=0.05))
