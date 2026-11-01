"""Experiment starting values from fixed captures and injected templates."""

from zcu_tools.cfg_model import Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.support.measure.ctx_helpers import source, t1_initial_time

from .core import ZigZagScanCfg


def make_gain_form(ctx: FormContext) -> Form[ZigZagScanCfg]:
    """Build starting values and shape from the captured source availability."""
    return Form(
        ZigZagScanCfg,
        tree={
            "reset": Slot("$qubit.module.reset"),
            "X90_pulse": source(
                ctx,
                "$qubit.module.x90",
                Template("pi2_pulse", bindings={"qubit": "$qubit"}),
            ),
            "X180_pulse": source(
                ctx,
                "$qubit.module.x180",
                Template("pi_pulse", bindings={"qubit": "$qubit"}),
            ),
            "readout": source(
                ctx,
                "$resonator.module.readout",
                Template("readout", bindings={"resonator": "$resonator"}),
            ),
            "relax_delay": t1_initial_time(ctx.parameters, multiplier=5),
            "n_times": 6,
            "repeat_on": "X180_pulse",
            "reps": 1000,
            "rounds": 10,
            "gain": Range(
                "0.8 * $qubit.module.x180.gain",
                "1.2 * $qubit.module.x180.gain",
                expts=101,
            ),
            "freq": Hidden(),
        },
    )


def make_freq_form(ctx: FormContext) -> Form[ZigZagScanCfg]:
    """Build starting values and shape from the captured source availability."""
    return Form(
        ZigZagScanCfg,
        tree={
            "reset": Slot("$qubit.module.reset"),
            "X90_pulse": source(
                ctx,
                "$qubit.module.x90",
                Template("pi2_pulse", bindings={"qubit": "$qubit"}),
            ),
            "X180_pulse": source(
                ctx,
                "$qubit.module.x180",
                Template("pi_pulse", bindings={"qubit": "$qubit"}),
            ),
            "readout": source(
                ctx,
                "$resonator.module.readout",
                Template("readout", bindings={"resonator": "$resonator"}),
            ),
            "relax_delay": t1_initial_time(ctx.parameters, multiplier=5),
            "n_times": 6,
            "repeat_on": "X180_pulse",
            "reps": 1000,
            "rounds": 10,
            "freq": Range.around("$qubit.freq", "2.0", expts=101),
            "gain": Hidden(),
        },
    )
