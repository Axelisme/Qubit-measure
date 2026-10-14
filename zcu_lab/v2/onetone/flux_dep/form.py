"""Starting values and form deviations for onetone/flux_dep; Qt-free data only."""

from zcu_tools.cfg_model import AliasName, Expr, Hidden, Range, Template
from zcu_tools.gui.form_adapter import Form

from zcu_lab.v2.onetone.flux_dep.core import FluxDepCfg

FORM = Form(
    FluxDepCfg,
    tree={
        "modules": {
            "reset": Hidden(),
            # A fresh inline readout; the freq sweep overwrites both freqs at Run.
            "readout": Template(
                "readout",
                set={
                    "pulse_cfg.gain": 0.05,
                    "ro_cfg.ro_length": Expr(
                        "$resonator.ext.default_probe_length - 0.1"
                    ),
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
