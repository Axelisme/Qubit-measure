"""Starting values for fastflux/mist, captured at Open and Reset."""

from zcu_tools.cfg_model import Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2._support.measure.ctx_helpers import t1_initial_time
from zcu_lab.v2.fastflux.mist.core import MistCfg


def make_form(ctx: FormContext) -> Form[MistCfg]:
    """Map flux and resonator-probe gains using editable pulse settings."""
    return Form(
        MistCfg,
        tree={
            "modules": {
                "reset": Slot("$qubit.module.reset"),
                "init_pulse": Slot("$qubit.module.x180"),
                # The template supplies wiring, not a calibrated flux pulse.
                "flux_pulse": Template("flux_pulse", set={"waveform.length": 1.0}),
                "mist_pulse": Template("res_probe"),
                "readout": Slot("$qubit.module.readout")
                if ctx.has("$qubit.module.readout")
                else Template("readout"),
            },
            "sweep": {
                "flux_gain": Range.between(0.0, 0.1, expts=101),
                "mist_gain": Range.between(0.0, 0.1, expts=101),
            },
            "relax_delay": t1_initial_time(ctx.parameters, multiplier=5.0),
            "initial_delay": Hidden(),
            "reps": 1000,
            "rounds": 100,
        },
    )
