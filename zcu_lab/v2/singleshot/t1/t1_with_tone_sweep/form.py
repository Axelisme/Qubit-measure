"""Each form exposes length and exactly one outer sweep; the other stays None."""

from zcu_tools.cfg_model import Expr, Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2._support.measure.ctx_helpers import t1_initial_time
from zcu_lab.v2.singleshot.t1.t1_with_tone_sweep.core import T1WithToneSweepCfg


def make_form_gain(ctx: FormContext) -> Form[T1WithToneSweepCfg]:
    """Build starting values and shape from the captured source availability."""
    relax_delay = t1_initial_time(ctx.parameters, multiplier=5.0)
    sweep_stop = t1_initial_time(ctx.parameters, multiplier=5.0, with_tone=True)
    return Form(
        T1WithToneSweepCfg,
        tree={
            "modules": {
                "reset": Slot("$qubit.module.reset"),
                "pi_pulse": Slot("$qubit.module.x180")
                if ctx.has("$qubit.module.x180")
                else Template("pi_pulse", bindings={"qubit": "$qubit"}),
                "probe_pulse": Template(
                    "res_probe", bindings={"resonator": "$resonator"}
                ),
                "readout": Slot("$qubit.module.readout")
                if ctx.has("$qubit.module.readout")
                else Template("readout", bindings={"resonator": "$resonator"}),
            },
            "sweep": {
                "length": Range.between(0.0, sweep_stop, expts=51),
                "gain": Range.between(0.0, 1.0, expts=21),
                "freq": Hidden(),
            },
            "g_center": Expr("$qubit.ext.readout_cal.g_center"),
            "e_center": Expr("$qubit.ext.readout_cal.e_center"),
            "radius": Expr("$qubit.ext.readout_cal.radius"),
            "relax_delay": relax_delay,
            "uniform": True,
            "reps": 1000,
            "rounds": 10,
            "initial_delay": Hidden(),
        },
    )


def make_form_freq(ctx: FormContext) -> Form[T1WithToneSweepCfg]:
    """Build starting values and shape from the captured source availability."""
    relax_delay = t1_initial_time(ctx.parameters, multiplier=5.0)
    sweep_stop = t1_initial_time(ctx.parameters, multiplier=5.0, with_tone=True)
    return Form(
        T1WithToneSweepCfg,
        tree={
            "modules": {
                "reset": Slot("$qubit.module.reset"),
                "pi_pulse": Slot("$qubit.module.x180")
                if ctx.has("$qubit.module.x180")
                else Template("pi_pulse", bindings={"qubit": "$qubit"}),
                "probe_pulse": Template(
                    "res_probe", bindings={"resonator": "$resonator"}
                ),
                "readout": Slot("$qubit.module.readout")
                if ctx.has("$qubit.module.readout")
                else Template("readout", bindings={"resonator": "$resonator"}),
            },
            "sweep": {
                "length": Range.between(0.0, sweep_stop, expts=51),
                "freq": Range.around(
                    "$resonator.freq", "1.5 * $resonator.kappa", expts=21
                ),
                "gain": Hidden(),
            },
            "g_center": Expr("$qubit.ext.readout_cal.g_center"),
            "e_center": Expr("$qubit.ext.readout_cal.e_center"),
            "radius": Expr("$qubit.ext.readout_cal.radius"),
            "relax_delay": relax_delay,
            "uniform": True,
            "reps": 1000,
            "rounds": 10,
            "initial_delay": Hidden(),
        },
    )
