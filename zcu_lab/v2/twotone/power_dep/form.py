"""Power-dependent spectroscopy starting values."""

from zcu_tools.cfg_model import Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from .core import PowerCfg


def make_form(ctx: FormContext) -> Form[PowerCfg]:
    """Build starting values and shape from the captured source availability."""
    return Form(
        PowerCfg,
        tree={
            "modules": {
                "reset": Slot("$qubit.module.reset"),
                "init_pulse": Hidden(),
                "qub_pulse": Template("qub_probe"),
                "readout": Slot("$qubit.module.readout")
                if ctx.has("$qubit.module.readout")
                else Template("readout"),
            },
            "sweep": {
                "gain": Range.between(0.001, 1.0, expts=101),
                "freq": Range.around("$qubit.freq", "1.5 * $qubit.kappa", expts=201),
            },
            "relax_delay": 1.0,
            "reps": 1000,
            "rounds": 100,
            "initial_delay": Hidden(),
        },
    )
