"""Starting values for jpa/flux_onetone; Qt-free declarations."""

from zcu_tools.cfg_model import AliasName, Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form

from zcu_lab.v2.jpa.flux_onetone.core import OneToneFluxCfg

FORM = Form(
    OneToneFluxCfg,
    tree={
        "modules": {
            "reset": Slot("$qubit.module.reset"),
            # A fresh readout; Run overwrites both frequencies from the sweep.
            "readout": Template("readout"),
        },
        "jpa_flux_dev": AliasName("flux_dev"),
        "flux_unit": "A",
        "sweep": {
            "jpa_flux": Range.around("$jpa.flux", 5.0e-3, expts=101),
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
