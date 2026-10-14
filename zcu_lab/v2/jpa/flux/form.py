"""Starting values for jpa/flux; Qt-free declarations."""

from zcu_tools.cfg_model import AliasName, Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form

from zcu_lab.v2.jpa.flux.core import FluxCfg

FORM = Form(
    FluxCfg,
    tree={
        "modules": {
            "reset": Slot("$qubit.module.reset"),
            "pi_pulse": Slot("$qubit.module.x180", fallback=Template("pi_pulse")),
            "readout": Slot("$qubit.module.readout", fallback=Template("readout")),
        },
        "jpa_flux_dev": AliasName("flux_dev"),
        "flux_unit": "A",
        "sweep": {
            "jpa_flux": Range.around("$jpa.flux", 5.0e-3, expts=101),
        },
        "skew_penalty": 0.0,
        "relax_delay": 0.5,
        "reps": 10000,
        "rounds": 1,
        "initial_delay": Hidden(),
    },
)
