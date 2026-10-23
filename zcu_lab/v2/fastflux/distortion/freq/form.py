"""Starting values for delayed spectroscopy of a fast-flux transient."""

from zcu_tools.cfg_model import Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.fastflux.distortion.freq.core import FreqCfg


def make_form(ctx: FormContext) -> Form[FreqCfg]:
    """Choose pulse sources from the fixed Open/Reset capture."""
    return Form(
        FreqCfg,
        tree={
            "modules": {
                "reset": Slot("$qubit.module.reset"),
                "flux_pulse": Template(
                    "flux_pulse",
                    bindings={"qubit": "$qubit"},
                    set={"waveform.length": 0.5},
                ),
                "qub_pulse": Template(
                    "qub_probe",
                    bindings={"qubit": "$qubit"},
                    # Run supplies the spectroscopy frequency scan.
                    set={"freq": Hidden(0.0), "gain": 0.1, "waveform.length": 0.05},
                ),
                "readout": Slot("$qubit.module.readout")
                if ctx.has("$qubit.module.readout")
                else Template("readout", bindings={"resonator": "$resonator"}),
            },
            "sweep": {
                "length": Range.between(0.0, 1.0, expts=101),
                "freq": Range.around("$qubit.freq", "1.5 * $qubit.kappa", expts=101),
            },
            "readout_t": 2.0,
            "relax_delay": 1.0,
            "reps": 1000,
            "rounds": 10,
            "initial_delay": Hidden(),
        },
    )
