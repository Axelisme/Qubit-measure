"""Starting values for jpa/check; Qt-free declarations."""

from zcu_tools.cfg_model import Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.jpa.check.core import CheckCfg


def make_form(ctx: FormContext) -> Form[CheckCfg]:
    """Build starting values and shape from the captured source availability."""
    rf_source = (
        ctx.parameters.lookup_point("$jpa.rf_source").value
        if ctx.parameters is not None
        else None
    )
    if rf_source is not None and not isinstance(rf_source, str):
        raise ValueError("$jpa.rf_source must name a device")
    return Form(
        CheckCfg,
        tree={
            "reset": Slot("$qubit.module.reset"),
            "readout": Template(
                "readout",
                bindings={"resonator": "$resonator"},
                set={"freq": Hidden(0.0)},
            ),
            "jpa_rf_dev": rf_source,
            "freq": Range.around(
                "$resonator.freq", "1.5 * $resonator.kappa", expts=101
            ),
            "relax_delay": 0.5,
            "reps": 1000,
            "rounds": 10,
        },
    )
