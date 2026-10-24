"""Fast-flux relaxation starting values from fixed Open/Reset captures."""

from zcu_tools.cfg_model import Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2._support.measure.ctx_helpers import t1_initial_time

from .core import T1Cfg


def make_form(ctx: FormContext) -> Form[T1Cfg]:
    """Prepare a zero-gain flux scan; operators supply the calibrated gain range."""
    relax_delay = t1_initial_time(ctx.parameters, multiplier=5.0)
    return Form(
        T1Cfg,
        tree={
            "modules": {
                "reset": Slot("$qubit.module.reset"),
                "pi_pulse": Slot("$qubit.module.x180")
                if ctx.has("$qubit.module.x180")
                else Template("pi_pulse", bindings={"qubit": "$qubit"}),
                "flux_pulse": Template(
                    "flux_pulse",
                    bindings={"qubit": "$qubit"},
                    # Run supplies gain; length remains a waveform-shape baseline.
                    set={"gain": Hidden(0.0), "waveform.length": 0.1},
                ),
                "readout": Slot("$qubit.module.readout")
                if ctx.has("$qubit.module.readout")
                else Template("readout", bindings={"resonator": "$resonator"}),
            },
            "sweep": {
                "gain": Range(0.0, 0.0, expts=1),
                "length": Range(0.1, relax_delay, expts=101),
            },
            "relax_delay": relax_delay,
            "initial_delay": Hidden(),
            "reps": 1000,
            "rounds": 10,
        },
    )
