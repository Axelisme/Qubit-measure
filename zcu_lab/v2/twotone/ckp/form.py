"""CKP starting values, using fixed sources and injected templates."""

from math import pi

from zcu_tools.cfg_model import Expr, Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2._support.measure.ctx_helpers import t1_initial_time

from .core import CKP_Cfg


def make_form(ctx: FormContext) -> Form[CKP_Cfg]:
    """Build starting values and shape from the captured source availability."""
    return Form(
        CKP_Cfg,
        tree={
            "modules": {
                "reset": Slot("$qubit.module.reset"),
                "pi_pulse": Slot("$qubit.module.x180")
                if ctx.has("$qubit.module.x180")
                else Template("pi_pulse", bindings={"qubit": "$qubit"}),
                "res_pulse": Template(
                    "res_probe",
                    bindings={"resonator": "$resonator"},
                    set={
                        # Run supplies the resonator-frequency sweep.
                        "freq": Hidden(0.0),
                        "gain": 0.015,
                        "waveform.length": Expr(
                            f"1.5 + {5.1 / (2 * pi)} / $resonator.kappa"
                        ),
                    },
                ),
                "qub_pulse": Template(
                    "qub_probe",
                    bindings={"qubit": "$qubit"},
                    set={
                        # Run supplies the qubit-frequency sweep.
                        "freq": Hidden(0.0),
                        "gain": 0.01,
                        "waveform.length": 1.5,
                        "pre_delay": Expr(f"{5.0 / (2 * pi)} / $resonator.kappa"),
                        "post_delay": Expr(f"{3.1 / (2 * pi)} / $resonator.kappa"),
                    },
                ),
                "readout": Slot("$qubit.module.readout")
                if ctx.has("$qubit.module.readout")
                else Template("readout", bindings={"resonator": "$resonator"}),
            },
            "sweep": {
                "res_freq": Range.around(
                    "$resonator.freq", "1.5 * $resonator.kappa", expts=101
                ),
                "qub_freq": Range.between(
                    "$qubit.freq - 10", "$qubit.freq + 5", expts=101
                ),
            },
            "relax_delay": t1_initial_time(ctx.parameters, multiplier=5),
            "reps": 100,
            "rounds": 100,
            "initial_delay": Hidden(),
        },
    )
