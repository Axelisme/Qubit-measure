"""Core-derived starting values for reset/bath/freq."""

from zcu_tools.cfg_model import Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2._support.measure.ctx_helpers import t1_initial_time
from zcu_lab.v2.twotone.reset.bath.freq.core import FreqGainCfg


def make_form(ctx: FormContext) -> Form[FreqGainCfg]:
    """Build starting values and shape from the captured source availability."""
    return Form(
        FreqGainCfg,
        tree={
            "modules": {
                "reset": Slot("$qubit.module.reset"),
                "init_pulse": Slot("$qubit.module.x180"),
                "tested_reset": Template(
                    "bath_reset",
                    bindings={"qubit": "$qubit", "resonator": "$resonator"},
                    set={
                        "pi2_cfg": Slot(
                            "$qubit.module.x90", copy=True, set={"phase": 90.0}
                        )
                    },
                ),
                "readout": Slot("$qubit.module.readout")
                if ctx.has("$qubit.module.readout")
                else Template("readout", bindings={"resonator": "$resonator"}),
            },
            "sweep": {
                "freq": Range.between(
                    "$resonator.freq - 1.2 * $qubit.ext.rabi_freq",
                    "$resonator.freq - 0.8 * $qubit.ext.rabi_freq",
                    expts=51,
                ),
                "gain": Range.between(0.4, 1.0, expts=51),
            },
            "relax_delay": t1_initial_time(ctx.parameters, multiplier=5),
            "reps": 1000,
            "rounds": 100,
            "initial_delay": Hidden(),
        },
    )
