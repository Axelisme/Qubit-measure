"""Starting values and form deviations for onetone/flux_dep; Qt-free data only."""

from zcu_tools.cfg_model import Hidden, Range, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.onetone._form_shared import initial_readout_timing
from zcu_lab.v2.onetone.flux_dep.core import FluxDepCfg
from zcu_lab.v2.support.measure.flux_defaults import qubit_flux_initial_values
from zcu_lab.v2.support.measure.seed_metadata import (
    pulse_readout_seed_metadata,
)


def make_form(ctx: FormContext) -> Form[FluxDepCfg]:
    """Build starting values and shape from the captured source availability."""
    timing = initial_readout_timing(ctx)
    return Form(
        FluxDepCfg,
        initial_values=qubit_flux_initial_values,
        field_order=(
            "reset",
            "readout",
            "flux_dev",
            "flux_unit",
            "freq",
            "flux",
            "reps",
            "rounds",
            "relax_delay",
        ),
        tree={
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
            "freq": Range.around(
                "$resonator.freq", "1.0 * $resonator.kappa", expts=101
            ),
            "flux": Range(None, None, expts=101),
            "relax_delay": 1.0,
            "reps": 1000,
            "rounds": 10,
        },
    ).with_seed_metadata(pulse_readout_seed_metadata(gain=0.05))
