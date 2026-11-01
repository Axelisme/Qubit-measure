"""Starting values for singleshot/reset_check; sources are captured at Open/Reset."""

from zcu_tools.cfg_model import Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.singleshot.reset_check.core import ResetCheckCfg
from zcu_lab.v2.support.measure.ctx_helpers import t1_initial_time


def make_form(ctx: FormContext) -> Form[ResetCheckCfg]:
    """Build starting values and shape from the captured source availability."""
    return Form(
        ResetCheckCfg,
        tree={
            "reset": Slot("$qubit.module.reset"),
            "rabi_pulse": Slot("$qubit.module.x180", set={"gain": Hidden(0.0)}, fallback=Template(
                "pi_pulse", bindings={"qubit": "$qubit"}, set={"gain": Hidden(0.0)}
            )),
            "tested_reset": Slot("$qubit.module.reset", fallback=Template("reset", bindings={"qubit": "$qubit"})),
            "readout": Slot("$resonator.module.readout", fallback=Template("readout", bindings={"resonator": "$resonator"})),
            "gain": Range(0.0, "4.0 * $qubit.module.x180.gain", expts=51),
            "relax_delay": t1_initial_time(ctx.parameters, multiplier=5),
            "reps": 1000,
            "rounds": 10,
        },
    )
