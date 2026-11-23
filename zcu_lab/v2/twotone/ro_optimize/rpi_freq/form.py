"""Starting values for ro_optimize/rpi_freq; Qt-free data only."""

from dataclasses import replace

from zcu_tools.cfg_model import Hidden, Range, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.support.measure.ctx_helpers import source
from zcu_lab.v2.support.measure.seed_metadata import pulse_readout_seed_metadata
from zcu_lab.v2.twotone.ro_optimize.rpi_freq.core import RPiFreqCfg


def make_form(ctx: FormContext) -> Form[RPiFreqCfg]:
    """Use the captured readout and x180 slots, or fresh templates when absent.

    The pi pulse is an operator-calibrated input: this Form does not check it.
    The frequency sweep starts at 31 points around $resonator.freq with half
    width $resonator.kappa. Reps defaults to 250 cycles of 64 R-pi pairs;
    rounds defaults to 20.
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
    # Run replaces the readout frequency with the sweep.
    readout = replace(readout, set={**readout.set, "freq": Hidden(0.0)})
    return Form(
        RPiFreqCfg,
        field_order=("readout", "pi_pulse", "freq", "reps", "rounds"),
        tree={
            "reps": 250,
            "rounds": 20,
            "readout": readout,
            "pi_pulse": source(
                ctx,
                "$qubit.module.x180",
                Template("pi_pulse", bindings={"qubit": "$qubit"}),
            ),
            "freq": Range.around("$resonator.freq", "$resonator.kappa", expts=31),
        },
    )
