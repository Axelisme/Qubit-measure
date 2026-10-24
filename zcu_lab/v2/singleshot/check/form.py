"""Starting values for singleshot/check; sources are captured at Open/Reset."""

from zcu_tools.cfg_model import Hidden, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.singleshot.check.core import CheckCfg
from zcu_lab.v2.support.measure.ctx_helpers import t1_initial_time


def make_form(ctx: FormContext) -> Form[CheckCfg]:
    """Build starting values and shape from the captured source availability."""
    return Form(
        CheckCfg,
        tree={
            "reset": Slot("$qubit.module.reset"),
            "init_pulse": Slot("$qubit.module.x180"),
            "probe_pulse": Template("qub_probe", bindings={"qubit": "$qubit"}),
            "readout": Slot("$qubit.module.readout")
            if ctx.has("$qubit.module.readout")
            else Template("readout", bindings={"resonator": "$resonator"}),
            "relax_delay": t1_initial_time(ctx.parameters, multiplier=5),
            "shots": 5000,
            "initial_delay": Hidden(),
        },
    )
