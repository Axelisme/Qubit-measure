"""Starting values for jpa/flux; Qt-free declarations."""

from zcu_tools.cfg_model import AliasName, Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.jpa.flux.core import FluxCfg


def make_form(ctx: FormContext) -> Form[FluxCfg]:
    """Build starting values and shape from the captured source availability."""
    return Form(
        FluxCfg,
        tree={
            "modules": {
                "reset": Slot("$qubit.module.reset"),
                "pi_pulse": Slot("$qubit.module.x180")
                if ctx.has("$qubit.module.x180")
                else Template("pi_pulse"),
                "readout": Slot("$qubit.module.readout")
                if ctx.has("$qubit.module.readout")
                else Template("readout"),
            },
            "jpa_flux_dev": AliasName("flux_dev"),
            "flux_unit": "A",
            "sweep": {"jpa_flux": Range.around("$jpa.flux", 0.005, expts=101)},
            "skew_penalty": 0.0,
            "relax_delay": 0.5,
            "reps": 10000,
            "rounds": 1,
            "initial_delay": Hidden(),
        },
    )
