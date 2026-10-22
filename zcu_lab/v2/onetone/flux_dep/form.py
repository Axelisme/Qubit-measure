"""Starting values and form deviations for onetone/flux_dep; Qt-free data only."""

from zcu_tools.cfg_model import AliasName, Hidden, Range, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.onetone._form_shared import initial_readout_timing
from zcu_lab.v2.onetone.flux_dep.core import FluxDepCfg


def make_form(ctx: FormContext) -> Form[FluxDepCfg]:
    """Build starting values and shape from the captured source availability."""
    timing = initial_readout_timing(ctx)
    return Form(
        FluxDepCfg,
        tree={
            "modules": {
                "reset": Hidden(),
                "readout": Template(
                    "readout",
                    bindings={"resonator": "$resonator"},
                    set={
                        # Run supplies the frequency sweep at every flux point.
                        "freq": Hidden(0.0),
                        "gain": 0.05,
                        **timing,
                    },
                ),
            },
            "flux_dev": AliasName("flux_dev"),
            "flux_unit": "A",
            "sweep": {
                "freq": Range.around(
                    "$resonator.freq", "1.0 * $resonator.kappa", expts=101
                ),
                "flux": Range.between(
                    "1.1 * $qubit.flux_int - 0.1 * $qubit.flux_half",
                    "1.1 * $qubit.flux_half - 0.1 * $qubit.flux_int",
                    expts=101,
                ),
            },
            "relax_delay": 1.0,
            "reps": 1000,
            "rounds": 1,
            "initial_delay": Hidden(),
        },
    )
