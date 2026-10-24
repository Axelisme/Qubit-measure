"""Starting values for accumulated-phase distortion measurements."""

from zcu_tools.cfg_model import Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2._support.measure.ctx_helpers import t1_initial_time

from .core import AccPhaseCfg


def make_form(ctx: FormContext) -> Form[AccPhaseCfg]:
    """Use captured drive/readout calibrations and the shared flux pulse role."""
    return Form(
        AccPhaseCfg,
        tree={
            "modules": {
                "reset": Slot("$qubit.module.reset"),
                "flux_pulse": Template("flux_pulse", bindings={"qubit": "$qubit"}),
                "pi2_pulse": Slot("$qubit.module.x90")
                if ctx.has("$qubit.module.x90")
                else Template("pi2_pulse", bindings={"qubit": "$qubit"}),
                "readout": Slot("$qubit.module.readout")
                if ctx.has("$qubit.module.readout")
                else Template("readout", bindings={"resonator": "$resonator"}),
            },
            "sweep": {
                "length": Range(0.0, 2.0, expts=101),
                "phase": Range(0.0, 360.0, expts=21),
            },
            "readout_t": 2.0,
            "relax_delay": t1_initial_time(ctx.parameters, multiplier=5),
            "reps": 1000,
            "rounds": 10,
            "initial_delay": Hidden(),
        },
    )
