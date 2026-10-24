"""Starting values for jpa/flux_onetone; Qt-free declarations."""

from zcu_tools.cfg_model import Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.jpa.flux_onetone.core import OneToneFluxCfg


def make_form(ctx: FormContext) -> Form[OneToneFluxCfg]:
    """Build starting values and shape from the captured source availability."""
    flux_source = (
        ctx.parameters.lookup_point("$jpa.flux_source").value
        if ctx.parameters is not None
        else None
    )
    if flux_source is not None and not isinstance(flux_source, str):
        raise ValueError("$jpa.flux_source must name a device")
    return Form(
        OneToneFluxCfg,
        tree={
            "reset": Slot("$qubit.module.reset"),
            "readout": Template(
                "readout",
                bindings={"resonator": "$resonator"},
                set={"freq": Hidden(0.0)},
            ),
            "jpa_flux_dev": flux_source,
            "flux_unit": "A",
            "jpa_flux": Range.around("$jpa.flux", 0.005, expts=101),
            "freq": Range.around(
                "$resonator.freq", "1.5 * $resonator.kappa", expts=101
            ),
            "relax_delay": 0.1,
            "reps": 1000,
            "rounds": 10,
        },
    )
