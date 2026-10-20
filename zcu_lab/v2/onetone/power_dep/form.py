"""One-tone power map defaults; Run sweeps frequency and pulse gain."""

from zcu_tools.cfg_model import Hidden, Range, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.onetone._shared import initial_ro_length
from zcu_lab.v2.onetone.power_dep.core import PowerDepCfg


def make_form(ctx: FormContext) -> Form[PowerDepCfg]:
    """Build starting values and shape from the captured source availability."""
    del ctx
    return Form(
        PowerDepCfg,
        tree={
            "modules": {
                "reset": Hidden(),
                "readout": Template(
                    "readout",
                    bindings={"resonator": "$resonator"},
                    set={"ro_length": initial_ro_length()},
                ),
            },
            "sweep": {
                "gain": Range.between(0.001, 1.0, expts=101),
                "freq": Range.around(
                    "$resonator.freq", "1.5 * $resonator.kappa", expts=201
                ),
            },
            "relax_delay": 1.0,
            "reps": 1000,
            "rounds": 100,
            "initial_delay": Hidden(),
        },
    )
