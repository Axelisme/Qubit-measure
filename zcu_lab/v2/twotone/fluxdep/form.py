"""Flux-dependent spectroscopy starting values and named device selection."""

from zcu_tools.cfg_model import AliasName, Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from .core import FreqFluxCfg


def make_form(ctx: FormContext) -> Form[FreqFluxCfg]:
    """Build starting values and shape from the captured source availability."""
    return Form(
        FreqFluxCfg,
        tree={
            "modules": {
                "reset": Slot("$qubit.module.reset"),
                "init_pulse": Hidden(),
                "qub_pulse": Template("qub_probe", bindings={"qubit": "$qubit"}),
                "readout": Slot("$qubit.module.readout")
                if ctx.has("$qubit.module.readout")
                else Template("readout", bindings={"resonator": "$resonator"}),
            },
            "flux_dev": AliasName("flux_dev"),
            "flux_unit": "A",
            "sweep": {
                "freq": Range.around("$qubit.freq", "$qubit.kappa", expts=1001),
                "flux": Range.between(
                    "1.1 * $qubit.flux_int - 0.1 * $qubit.flux_half",
                    "1.1 * $qubit.flux_half - 0.1 * $qubit.flux_int",
                    expts=101,
                ),
            },
            "relax_delay": 1.0,
            "reps": 1000,
            "rounds": 100,
            "fail_retry": Hidden(),
            "initial_delay": Hidden(),
        },
    )
