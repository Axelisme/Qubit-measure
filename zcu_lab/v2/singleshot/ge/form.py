"""Starting values for singleshot/ge; sources are captured at Open/Reset."""

from zcu_tools.cfg_model import Hidden, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2._support.measure.ctx_helpers import t1_initial_time
from zcu_lab.v2.singleshot.ge.core import GE_Cfg


def make_form(ctx: FormContext) -> Form[GE_Cfg]:
    """Build starting values and shape from the captured source availability."""
    return Form(
        GE_Cfg,
        tree={
            "modules": {
                "reset": Slot("$qubit.module.reset"),
                "init_pulse": Slot("$qubit.module.x180"),
                "probe_pulse": Slot("$qubit.module.x180")
                if ctx.has("$qubit.module.x180")
                else Template("pi_pulse"),
                "readout": Slot("$qubit.module.readout")
                if ctx.has("$qubit.module.readout")
                else Template("readout"),
            },
            "relax_delay": t1_initial_time(ctx.parameters, multiplier=5),
            "shots": 100000,
            "initial_delay": Hidden(),
        },
    )
