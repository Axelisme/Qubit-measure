"""Starting values for singleshot/len_rabi; sources are captured at Open/Reset."""

from dataclasses import replace

from zcu_tools.cfg_model import Expr, Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.singleshot.len_rabi.core import LenRabiCfg
from zcu_lab.v2.support.measure.ctx_helpers import t1_initial_time
from zcu_lab.v2.support.measure.rabi_initial_values import len_rabi_initial_values


def make_form(ctx: FormContext) -> Form[LenRabiCfg]:
    """Build starting values and shape from the captured source availability."""
    qub_pulse, length_stop = len_rabi_initial_values(ctx.parameters)
    # This Run accepts only const/flat_top and replaces total pulse duration.
    qub_pulse = replace(
        qub_pulse, set={**qub_pulse.set, "waveform.length": Hidden(0.0)}
    )
    return Form(
        LenRabiCfg,
        tree={
            "reset": Slot("$qubit.module.reset"),
            "qub_pulse": qub_pulse,
            "readout": Slot("$qubit.module.readout")
            if ctx.has("$qubit.module.readout")
            else Template("readout", bindings={"resonator": "$resonator"}),
            "length": Range(0.03, length_stop, expts=51),
            "g_center": Expr("$qubit.ext.readout_cal.g_center"),
            "e_center": Expr("$qubit.ext.readout_cal.e_center"),
            "radius": Expr("$qubit.ext.readout_cal.radius"),
            "relax_delay": t1_initial_time(ctx.parameters, multiplier=5),
            "shots": 1000,
            "initial_delay": Hidden(),
        },
    )
