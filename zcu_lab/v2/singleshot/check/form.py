"""Starting values for singleshot/check; sources are captured at Open/Reset."""

from zcu_tools.cfg_model import Hidden, Locked, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2._support.measure.ctx_helpers import t1_initial_time
from zcu_lab.v2.singleshot.check.core import CheckCfg


def make_form(ctx: FormContext) -> Form[CheckCfg]:
    """Build starting values and shape from the captured source availability."""
    return Form(
        CheckCfg,
        tree={
            "modules": {
                "reset": Slot("$qubit.module.reset"),
                "init_pulse": Slot("$qubit.module.x180"),
                "probe_pulse": Template("qub_probe"),
                "readout": Slot("$qubit.module.readout")
                if ctx.has("$qubit.module.readout")
                else Template("readout"),
            },
            "relax_delay": t1_initial_time(ctx.parameters, multiplier=5),
            "shots": 5000,
            "reps": Locked(1),
            "rounds": Locked(1),
            "initial_delay": Hidden(),
        },
    )
