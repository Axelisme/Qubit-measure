"""CKP starting values, using fixed sources and injected templates."""

from math import pi

from zcu_tools.cfg_model import Expr, Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form

from .core import CKP_Cfg

FORM = Form(
    CKP_Cfg,
    tree={
        "modules": {
            "reset": Slot("$qubit.module.reset"),
            "pi_pulse": Slot("$qubit.module.x180", fallback=Template("pi_pulse")),
            "res_pulse": Template(
                "res_probe",
                set={
                    "gain": 0.015,
                    "waveform.length": Expr(
                        f"1.5 + {5.1 / (2 * pi)} / $resonator.kappa"
                    ),
                },
            ),
            "qub_pulse": Template(
                "qub_probe",
                set={
                    "gain": 0.01,
                    "waveform.length": 1.5,
                    "pre_delay": Expr(f"{5.0 / (2 * pi)} / $resonator.kappa"),
                    "post_delay": Expr(f"{3.1 / (2 * pi)} / $resonator.kappa"),
                },
            ),
            "readout": Slot("$qubit.module.readout", fallback=Template("readout")),
        },
        "sweep": {
            "res_freq": Range.around(
                "$resonator.freq", "1.5 * $resonator.kappa", expts=101
            ),
            "qub_freq": Range.between("$qubit.freq - 10", "$qubit.freq + 5", expts=101),
        },
        "relax_delay": 10.1,
        "reps": 100,
        "rounds": 100,
        "initial_delay": Hidden(),
    },
)
