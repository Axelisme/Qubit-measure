"""Starting values for jpa/flux_onetone; Qt-free declarations."""

from zcu_tools.cfg_model import AliasName, Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.jpa.flux_onetone.core import OneToneFluxCfg


def make_form(ctx: FormContext) -> Form[OneToneFluxCfg]:
    """Build starting values and shape from the captured source availability."""
    del ctx
    return Form(
        OneToneFluxCfg,
        tree={
            "modules": {
                "reset": Slot("$qubit.module.reset"),
                "readout": Template("readout", bindings={"resonator": "$resonator"}),
            },
            "jpa_flux_dev": AliasName("flux_dev"),
            "flux_unit": "A",
            "sweep": {
                "jpa_flux": Range.around("$jpa.flux", 0.005, expts=101),
                "freq": Range.around(
                    "$resonator.freq", "1.5 * $resonator.kappa", expts=101
                ),
            },
            "relax_delay": 0.1,
            "reps": 100,
            "rounds": 10,
            "initial_delay": Hidden(),
        },
    )
