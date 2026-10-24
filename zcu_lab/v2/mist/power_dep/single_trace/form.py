"""Starting values for the averaged-IQ MIST probe-gain trace."""

from zcu_tools.cfg_model import Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.mist.power_dep.single_trace.core import PowerDepCfg
from zcu_lab.v2.support.measure.ctx_helpers import t1_initial_time


def make_form(ctx: FormContext) -> Form[PowerDepCfg]:
    """Seed probe/readout and optional preparation from captured Open/Reset sources."""
    return Form(
        PowerDepCfg,
        tree={
            "reset": Slot("$qubit.module.reset"),
            "init_pulse": Slot("$qubit.module.x180"),
            "probe_pulse": Template(
                "res_probe",
                bindings={"resonator": "$resonator"},
                # Run supplies these probe scan coordinates.
                set={"gain": Hidden(0.0)},
            ),
            "readout": Slot("$qubit.module.readout")
            if ctx.has("$qubit.module.readout")
            else Template("readout", bindings={"resonator": "$resonator"}),
            "gain": Range(0.0, 1.0, expts=151),
            "relax_delay": t1_initial_time(ctx.parameters, multiplier=5),
            "reps": 1000,
            "rounds": 10,
        },
    )
