"""One-tone starting values; the Run sweep owns the shared generator/ADC frequency."""

from zcu_tools.cfg_model import Expr, Hidden, Range, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.onetone.freq.core import FreqCfg


def make_form(ctx: FormContext) -> Form[FreqCfg]:
    """Build starting values and shape from the captured source availability."""
    del ctx
    return Form(
        FreqCfg,
        tree={
            "modules": {
                "reset": Hidden(),
                "readout": Template(
                    "readout",
                    set={
                        "gain": 0.05,
                        "ro_length": Expr("$resonator.ext.default_probe_length - 0.1"),
                    },
                ),
            },
            "sweep": {
                "freq": Range.around(
                    "$resonator.freq", "1.5 * $resonator.kappa", expts=301
                )
            },
            "sampling_mode": "linear",
            "relax_delay": 1.0,
            "reps": 100,
            "rounds": 100,
            "initial_delay": Hidden(),
        },
    )
