"""Starting values for ro_optimize/rpi_length; Qt-free data only."""

from dataclasses import replace

from zcu_tools.cfg_model import FormNode, Hidden, Range, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.support.measure.ctx_helpers import source
from zcu_lab.v2.support.measure.readout_initial_values import (
    pulse_readout_length_margin,
)
from zcu_lab.v2.support.measure.seed_metadata import pulse_readout_seed_metadata
from zcu_lab.v2.twotone.ro_optimize.rpi_length.core import RPiLengthCfg


def make_form(ctx: FormContext) -> Form[RPiLengthCfg]:
    """Use the captured readout and x180 slots, or fresh templates when absent.

    Seed the editable length margin in us from the selected readout's pulse
    length minus its ADC window. Use the template only when there is no captured
    readout. Leave the margin required and blank if either length cannot be
    resolved. The pi pulse is an operator-calibrated input.
    The ADC window sweep starts at 11 points from 0.2 to 2.2 us.
    Reps defaults to 250 cycles of 64 R-pi pairs; rounds defaults to 20.
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
    margin = pulse_readout_length_margin(ctx, readout)
    # Run replaces both lengths for each point in the ADC window sweep.
    readout = replace(
        readout,
        set={
            **readout.set,
            "ro_length": Hidden(0.0),
            "waveform.length": Hidden(0.0),
        },
    )
    tree: dict[str, FormNode] = {
        "reps": 250,
        "rounds": 20,
        "readout": readout,
        "pi_pulse": source(
            ctx,
            "$qubit.module.x180",
            Template("pi_pulse", bindings={"qubit": "$qubit"}),
        ),
        "length": Range(0.2, 2.2, expts=11),
    }
    if margin.value is not None:
        tree["length_margin"] = margin.value
    return Form(RPiLengthCfg, tree).with_seed_metadata(
        {("length_margin",): margin.metadata}
    )
