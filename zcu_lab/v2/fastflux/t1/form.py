"""Fast-flux relaxation starting values from fixed Open/Reset captures."""

from zcu_tools.cfg_model import Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.support.measure.ctx_helpers import source, t1_initial_time

from .core import T1Cfg


def make_form(ctx: FormContext) -> Form[T1Cfg]:
    """Prepare a zero-gain flux scan; operators supply the calibrated gain range."""
    relax_delay = t1_initial_time(ctx.parameters, multiplier=5.0)
    return Form(
        T1Cfg,
        tree={
            "reset": Slot("$qubit.module.reset"),
            "pi_pulse": source(
                ctx,
                "$qubit.module.x180",
                Template("pi_pulse", bindings={"qubit": "$qubit"}),
            ),
            "flux_pulse": Template(
                "flux_pulse",
                bindings={"qubit": "$qubit"},
                # Run supplies gain; length remains a waveform-shape baseline.
                set={"gain": Hidden(0.0), "waveform.length": 0.1},
            ),
            "readout": source(
                ctx,
                "$resonator.module.readout",
                Template("readout", bindings={"resonator": "$resonator"}),
            ),
            "gain": Range(0.0, 0.0, expts=1),
            "length": Range(0.1, relax_delay, expts=101),
            "relax_delay": relax_delay,
            "reps": 1000,
            "rounds": 10,
        },
    )
