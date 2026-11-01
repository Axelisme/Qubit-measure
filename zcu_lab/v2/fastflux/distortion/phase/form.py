"""Starting values for moving a tomography pulse pair through a flux transient."""

from zcu_tools.cfg_model import Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.support.measure.ctx_helpers import source, t1_initial_time

from .core import PhaseCfg


def make_form(ctx: FormContext) -> Form[PhaseCfg]:
    """Use captured calibration pulses and the shared fast-flux pulse role."""
    return Form(
        PhaseCfg,
        tree={
            "reset": Slot("$qubit.module.reset"),
            "flux_pulse": Template(
                "flux_pulse",
                bindings={"qubit": "$qubit"},
                set={"waveform.length": 1.0},
            ),
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
            "length": Range(0.0, 1.0, expts=101),
            "phase": Range(-180.0, 180.0, expts=61),
            "readout_t": 2.0,
            "relax_delay": t1_initial_time(ctx.parameters, multiplier=5),
            "reps": 1000,
            "rounds": 10,
        },
    )
