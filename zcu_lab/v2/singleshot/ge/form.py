"""Starting values for singleshot/ge; sources are captured at Open/Reset."""

from zcu_tools.cfg_model import Hidden, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.singleshot.ge.core import GE_Cfg
from zcu_lab.v2.support.measure.ctx_helpers import t1_initial_time
from zcu_lab.v2.support.measure.pulse_readout_metadata import (
    pulse_readout_seed_metadata,
)


def make_form(ctx: FormContext) -> Form[GE_Cfg]:
    """Build starting values and shape from the captured source availability."""
    return Form(
        GE_Cfg,
        field_order=(
            "reset",
            "init_pulse",
            "probe_pulse",
            "readout",
            "shots",
            "relax_delay",
            "initial_delay",
        ),
        tree={
            "reset": Slot("$qubit.module.reset"),
            "init_pulse": Slot("$qubit.module.x180"),
            "probe_pulse": Slot("$qubit.module.x180")
            if ctx.has("$qubit.module.x180")
            else Template("pi_pulse", bindings={"qubit": "$qubit"}),
            "readout": Slot("$qubit.module.readout")
            if ctx.has("$qubit.module.readout")
            else Template("readout", bindings={"resonator": "$resonator"}),
            "relax_delay": t1_initial_time(ctx.parameters, multiplier=5),
            "shots": 100000,
            "initial_delay": Hidden(),
        },
    ).with_seed_metadata(
        (
            {}
            if ctx.has("$qubit.module.readout")
            else pulse_readout_seed_metadata(gain=0.1)
        )
    )
