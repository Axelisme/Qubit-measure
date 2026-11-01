"""Starting values for accumulated-phase distortion measurements."""

from zcu_tools.cfg_model import Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.support.measure.ctx_helpers import source, t1_initial_time

from .core import AccPhaseCfg


def make_form(ctx: FormContext) -> Form[AccPhaseCfg]:
    """Use captured drive/readout calibrations and the shared flux pulse role."""
    return Form(
        AccPhaseCfg,
        tree={
            "reset": Slot("$qubit.module.reset"),
            "flux_pulse": Template("flux_pulse", bindings={"qubit": "$qubit"}),
            "pi2_pulse": source(
                ctx,
                "$qubit.module.x90",
                Template("pi2_pulse", bindings={"qubit": "$qubit"}),
            ),
            "readout": source(
                ctx,
                "$resonator.module.readout",
                Template("readout", bindings={"resonator": "$resonator"}),
            ),
            "length": Range(0.0, 2.0, expts=101),
            "phase": Range(0.0, 360.0, expts=21),
            "readout_t": 2.0,
            "relax_delay": t1_initial_time(ctx.parameters, multiplier=5),
            "reps": 1000,
            "rounds": 10,
        },
    )
