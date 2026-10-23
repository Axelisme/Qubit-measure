"""Starting values for jpa/check; Qt-free declarations."""

from zcu_tools.cfg_model import AliasName, Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.jpa.check.core import CheckCfg


def make_form(ctx: FormContext) -> Form[CheckCfg]:
    """Build starting values and shape from the captured source availability."""
    del ctx
    return Form(
        CheckCfg,
        tree={
            "modules": {
                "reset": Slot("$qubit.module.reset"),
                # Run supplies readout frequency for each external-device setting.
                "readout": Template(
                    "readout",
                    bindings={"resonator": "$resonator"},
                    set={"freq": Hidden(0.0)},
                ),
            },
            "jpa_rf_dev": AliasName("jpa_rf_dev"),
            "sweep": {
                "freq": Range.around(
                    "$resonator.freq", "1.5 * $resonator.kappa", expts=101
                )
            },
            "relax_delay": 0.5,
            "reps": 1000,
            "rounds": 10,
            "initial_delay": Hidden(),
        },
    )
