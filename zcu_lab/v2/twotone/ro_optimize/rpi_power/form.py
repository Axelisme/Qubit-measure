"""Starting values for ro_optimize/rpi_power; Qt-free data only."""

from dataclasses import replace

from zcu_tools.cfg_model import Hidden, Range, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.support.measure.ctx_helpers import source
from zcu_lab.v2.twotone.ro_optimize.rpi_power.core import RPiPowerCfg


def make_form(ctx: FormContext) -> Form[RPiPowerCfg]:
    """Use the captured readout and x180 slots, or fresh templates when absent.

    The pi pulse is an operator-calibrated input: this Form does not check it.
    Reps defaults to 250 cycles of 64 R-pi pairs; rounds defaults to 20.
    """
    readout = source(
        ctx,
        "$resonator.module.readout",
        Template("readout", bindings={"resonator": "$resonator"}),
    )
    # Run replaces the readout gain with the sweep.
    readout = replace(readout, set={**readout.set, "gain": Hidden(0.0)})
    return Form(
        RPiPowerCfg,
        field_order=("readout", "pi_pulse", "gain", "reps", "rounds"),
        tree={
            "reps": 250,
            "rounds": 20,
            "readout": readout,
            "pi_pulse": source(
                ctx,
                "$qubit.module.x180",
                Template("pi_pulse", bindings={"qubit": "$qubit"}),
            ),
            "gain": Range(0.02, 0.2, expts=31),
        },
    )
