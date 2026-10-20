"""Starting values and form deviations for twotone/freq; Qt-free data only."""

from zcu_tools.cfg_model import Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.twotone.freq.core import FreqCfg


def make_form(ctx: FormContext) -> Form[FreqCfg]:
    """Build starting values and shape from the captured source availability."""
    return Form(
        FreqCfg,
        tree={
            "modules": {
                "reset": Slot("$qubit.module.reset"),
                "init_pulse": Hidden(),
                "qub_pulse": Template(
                    "qub_probe",
                    bindings={"qubit": "$qubit"},
                    set={"gain": 0.1, "waveform.length": 1.0},
                ),
                "readout": Slot("$qubit.module.readout")
                if ctx.has("$qubit.module.readout")
                else Template("readout", bindings={"resonator": "$resonator"}),
            },
            "sweep": {
                "freq": Range.around("$qubit.freq", "1.5 * $qubit.kappa", expts=301)
            },
            "relax_delay": 1.0,
            "reps": 1000,
            "rounds": 100,
            "initial_delay": Hidden(),
        },
    )
