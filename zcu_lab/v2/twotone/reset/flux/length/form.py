"""Starting values for the fixed-gain flux reset length scan."""

from zcu_tools.cfg_model import Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.support.measure.ctx_helpers import t1_initial_time
from zcu_lab.v2.twotone.reset.flux.length.core import LengthCfg


def make_form(ctx: FormContext) -> Form[LengthCfg]:
    """Capture fixed flux gain; the operator supplies the permitted hold lengths."""
    return Form(
        LengthCfg,
        tree={
            "reset": Slot("$qubit.module.reset"),
            "pi_pulse": Slot("$qubit.module.x180")
            if ctx.has("$qubit.module.x180")
            else Template("pi_pulse", bindings={"qubit": "$qubit"}),
            "tested_reset": Template(
                "flux_reset",
                bindings={"qubit": "$qubit"},
                set={"pulse_cfg.waveform.length": Hidden(0.1)},
            ),
            "readout": Slot("$resonator.module.readout")
            if ctx.has("$resonator.module.readout")
            else Template("readout", bindings={"resonator": "$resonator"}),
            "length": Range(0.1, 0.1, expts=1),
            "relax_delay": t1_initial_time(ctx.parameters, multiplier=5),
            "reps": 1000,
            "rounds": 10,
        },
    )
