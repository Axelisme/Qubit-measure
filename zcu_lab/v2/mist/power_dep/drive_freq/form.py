"""Open/Reset defaults for the MIST probe frequency/gain map."""

from zcu_tools.cfg_model import Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2._support.measure.ctx_helpers import t1_initial_time

from .core import DriveFreqCfg


def make_form(ctx: FormContext) -> Form[DriveFreqCfg]:
    """Seed the map from captured qubit/resonator sources, without hardware I/O."""
    return Form(
        DriveFreqCfg,
        tree={
            "modules": {
                "reset": Slot("$qubit.module.reset"),
                "init_pulse": Slot("$qubit.module.x180"),
                "probe_pulse": Template(
                    "res_probe", bindings={"resonator": "$resonator"}
                ),
                "readout": Slot("$qubit.module.readout")
                if ctx.has("$qubit.module.readout")
                else Template("readout", bindings={"resonator": "$resonator"}),
            },
            "sweep": {
                "freq": Range.around(
                    "$resonator.freq", "1.5 * $resonator.kappa", expts=51
                ),
                "gain": Range.between(0.0, 1.0, expts=51),
            },
            "relax_delay": t1_initial_time(ctx.parameters, multiplier=5),
            "reps": 1000,
            "rounds": 100,
            "initial_delay": Hidden(),
        },
    )
