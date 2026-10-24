"""Starting values and form deviations for onetone/flux_dep; Qt-free data only."""

from zcu_tools.cfg_model import Hidden, Range, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2._support.measure.seed_metadata import (
    pulse_readout_seed_metadata,
)
from zcu_lab.v2.onetone._form_shared import initial_readout_timing
from zcu_lab.v2.onetone.flux_dep.core import FluxDepCfg


def make_form(ctx: FormContext) -> Form[FluxDepCfg]:
    """Build starting values and shape from the captured source availability."""
    timing = initial_readout_timing(ctx)
    flux_source = (
        ctx.parameters.lookup_point("$qubit.flux_source").value
        if ctx.parameters is not None
        else None
    )
    if flux_source is not None and not isinstance(flux_source, str):
        raise ValueError("$qubit.flux_source must name a device")
    return Form(
        FluxDepCfg,
        field_order=(
            "modules",
            "flux_dev",
            "flux_unit",
            "sweep",
            "reps",
            "rounds",
            "relax_delay",
            "initial_delay",
        ),
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
            "flux_dev": flux_source,
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
            "rounds": 10,
            "initial_delay": Hidden(),
        },
    ).with_seed_metadata(pulse_readout_seed_metadata(gain=0.05))
