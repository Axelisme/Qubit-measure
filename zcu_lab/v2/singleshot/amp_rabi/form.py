"""Starting values for singleshot/amp_rabi; sources are captured at Open/Reset."""

from dataclasses import replace

from zcu_tools.cfg_model import Expr, Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2._support.measure.ctx_helpers import t1_initial_time
from zcu_lab.v2._support.measure.rabi_initial_values import amp_rabi_initial_pulse
from zcu_lab.v2.singleshot.amp_rabi.core import AmpRabiCfg


def make_form(ctx: FormContext) -> Form[AmpRabiCfg]:
    """Build starting values and shape from the captured source availability."""
    pulse = amp_rabi_initial_pulse(ctx.parameters)
    # Keep shape/calibration admission; Run supplies gain itself.
    pulse = replace(pulse, set={**pulse.set, "gain": Hidden(0.0)})
    return Form(
        AmpRabiCfg,
        tree={
            "reset": Slot("$qubit.module.reset"),
            "qub_pulse": pulse,
            "readout": Slot("$qubit.module.readout")
            if ctx.has("$qubit.module.readout")
            else Template("readout", bindings={"resonator": "$resonator"}),
            "gain": Range.between(-0.3, 1.0, expts=51),
            "g_center": Expr("$qubit.ext.readout_cal.g_center"),
            "e_center": Expr("$qubit.ext.readout_cal.e_center"),
            "radius": Expr("$qubit.ext.readout_cal.radius"),
            "relax_delay": t1_initial_time(ctx.parameters, multiplier=5),
            "reps": 1000,
            "rounds": 10,
        },
    )
