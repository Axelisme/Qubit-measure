"""Starting values for ro_optimize/chain_length; Qt-free data only."""

from dataclasses import replace

from zcu_tools.cfg_model import FieldMeta, FormNode, Hidden, Range, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.support.measure.ctx_helpers import source, t1_initial_time
from zcu_lab.v2.support.measure.seed_metadata import pulse_readout_seed_metadata
from zcu_lab.v2.twotone.ro_optimize.chain_length.core import ChainLengthCfg

_FLIGHT_TIME_SOURCE = "$resonator.time_of_fly"


def _initial_time_of_fly(ctx: FormContext) -> float | None:
    """Capture the resonator's flight time in us, or leave its field blank."""
    parameters = ctx.parameters
    if parameters is None:
        return None
    path = parameters.resolve_path(_FLIGHT_TIME_SOURCE)
    point = parameters.point_content()
    # Navigate before validating the leaf so source and structure errors escape.
    try:
        point.node(path)
    except AttributeError:
        return None
    try:
        value = point.value(path)
    except ValueError:
        return None
    if value is not None and not isinstance(value, float):
        raise TypeError(f"{path}: expected a resonator flight time")
    return value


def make_form(ctx: FormContext) -> Form[ChainLengthCfg]:
    """Capture the readout and flight time and seed relaxation K from 5*T1.

    A missing or invalid flight time leaves the readonly field blank.
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
    flight_time = _initial_time_of_fly(ctx)
    tree: dict[str, FormNode] = {
        "reps": 2000,
        "rounds": 320,
        "readout": readout,
        "relax_delay": t1_initial_time(ctx.parameters, multiplier=5),
        "length": Range(0.2, 2.2, expts=11),
    }
    if flight_time is not None:
        tree["time_of_fly"] = flight_time
    return Form(ChainLengthCfg, tree).with_seed_metadata(
        {
            ("time_of_fly",): FieldMeta(
                description=f"From captured {_FLIGHT_TIME_SOURCE}"
            )
        }
    )
