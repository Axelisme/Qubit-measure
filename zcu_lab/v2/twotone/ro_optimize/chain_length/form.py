"""Starting values for ro_optimize/chain_length; Qt-free data only."""

from dataclasses import replace
from math import isfinite

from pydantic import ValidationError
from zcu_tools.cfg_model import Hidden, Range, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.support.measure.ctx_helpers import source, t1_initial_time
from zcu_lab.v2.support.measure.seed_metadata import pulse_readout_seed_metadata
from zcu_lab.v2.twotone.ro_optimize.chain_length.core import ChainLengthCfg


def _initial_time_of_fly(ctx: FormContext) -> float | None:
    """Capture finite nonnegative resonator flight time in us, or leave it blank."""
    if ctx.parameters is None:
        return None
    try:
        result = ctx.parameters.lookup("$resonator.time_of_fly")
    except ValueError as exc:
        # Raw point lookup wraps leaf schema errors with the parameter path.
        if isinstance(exc.__cause__, ValidationError):
            return None
        raise
    value = result.value
    if (
        not result.present
        or isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not isfinite(value)
        or value < 0
    ):
        return None
    return float(value)


def make_form(ctx: FormContext) -> Form[ChainLengthCfg]:
    """Capture the readout and flight time and seed relaxation K from 5*T1.

    A missing or invalid flight time leaves a required hidden value blank.
    Source faults propagate; this form does not read live parameters.
    """
    readout = source(
        ctx,
        "$resonator.module.readout",
        Template(
            "readout",
            bindings={"resonator": "$resonator"},
            seed_metadata=pulse_readout_seed_metadata(gain=0.1),
        ),
    )
    # Run replaces both lengths for each point in the ADC window sweep.
    readout = replace(
        readout,
        set={
            **readout.set,
            "ro_length": Hidden(0.0),
            "waveform.length": Hidden(0.0),
        },
    )
    return Form(
        ChainLengthCfg,
        tree={
            "reps": 2000,
            "rounds": 320,
            "readout": readout,
            "time_of_fly": Hidden(_initial_time_of_fly(ctx)),
            "relax_delay": t1_initial_time(ctx.parameters, multiplier=5),
            "length": Range(0.2, 2.2, expts=11),
        },
    )
