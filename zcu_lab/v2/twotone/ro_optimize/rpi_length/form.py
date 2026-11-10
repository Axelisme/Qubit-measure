"""Starting values for ro_optimize/rpi_length; Qt-free data only."""

from dataclasses import replace

from zcu_tools.cfg_model import Hidden, Range, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.support.measure.ctx_helpers import source
from zcu_lab.v2.support.measure.seed_metadata import pulse_readout_seed_metadata
from zcu_lab.v2.twotone.ro_optimize.rpi_length.core import RPiLengthCfg


def make_form(ctx: FormContext) -> Form[RPiLengthCfg]:
    """Use the captured readout and x180 slots, or fresh templates when absent.

    The pi pulse is an operator-calibrated input: this Form does not check it.
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
    # Run sets the ADC window from the sweep and the pulse length to that window
    # plus 0.11 us.
    readout = replace(
        readout,
        set={
            **readout.set,
            "ro_length": Hidden(0.0),
            "waveform.length": Hidden(0.0),
        },
    )
    return Form(
        RPiLengthCfg,
        field_order=("readout", "pi_pulse", "length", "reps", "rounds"),
        tree={
            "readout": readout,
            "pi_pulse": source(
                ctx,
                "$qubit.module.x180",
                Template("pi_pulse", bindings={"qubit": "$qubit"}),
            ),
            "length": Range(0.2, 2.2, expts=11),
        },
    )
