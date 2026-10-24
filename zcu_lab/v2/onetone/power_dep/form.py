"""One-tone power map defaults; Run sweeps frequency and pulse gain."""

from zcu_tools.cfg_model import Hidden, Range, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.onetone._form_shared import initial_readout_timing
from zcu_lab.v2.onetone.power_dep.core import PowerDepCfg


def make_form(ctx: FormContext) -> Form[PowerDepCfg]:
    """Build starting values and shape from the captured source availability."""
    timing = initial_readout_timing(ctx)
    return Form(
        PowerDepCfg,
        field_order=(
            "modules",
            "sweep",
            "earlystop_snr",
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
                    # Run supplies both scan coordinates, not the module literals.
                    set={
                        "freq": Hidden(0.0),
                        "gain": Hidden(0.0),
                        **timing,
                    },
                ),
            },
            "sweep": {
                "gain": Range(0.001, 1.0, expts=101),
                "freq": Range.around(
                    "$resonator.freq", "1.5 * $resonator.kappa", expts=201
                ),
            },
            "relax_delay": 1.0,
            "reps": 1000,
            "rounds": 10,
            "initial_delay": Hidden(),
        },
    )
