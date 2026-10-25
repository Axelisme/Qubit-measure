"""Starting values for singleshot/ac_stark; sources are captured at Open/Reset."""

from zcu_tools.cfg_model import Expr, Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.singleshot.ac_stark.core import AcStarkCfg
from zcu_lab.v2.support.measure.ctx_helpers import t1_initial_time


def make_form(ctx: FormContext) -> Form[AcStarkCfg]:
    """Build starting values and shape from the captured source availability."""
    return Form(
        AcStarkCfg,
        tree={
            "reset": None,
            "init_pulse": None,
            "stark_pulse1": Template(
                "res_probe",
                bindings={"resonator": "$resonator"},
                set={
                    # Run supplies the cavity gain scan.
                    "gain": Hidden(0.0),
                    "waveform.length": Expr(
                        "$qubit.module.x180.waveform.length + 1.0 / $resonator.kappa"
                    ),
                },
            ),
            "stark_pulse2": Template(
                "pi_pulse",
                bindings={"qubit": "$qubit"},
                set={
                    # Run supplies the qubit frequency scan.
                    "freq": Hidden(0.0),
                    "waveform.length": Expr("$qubit.module.x180.waveform.length"),
                    "mixer_freq": Expr("$qubit.module.x180.mixer_freq"),
                    "pre_delay": Expr("1.0 / $resonator.kappa"),
                    "post_delay": Expr("0.5 / $resonator.kappa"),
                },
            ),
            "readout": Slot("$resonator.module.readout")
            if ctx.has("$resonator.module.readout")
            else Template("readout", bindings={"resonator": "$resonator"}),
            "gain": Range(0.0, 0.22, expts=301),
            "freq": Range("$qubit.freq - 700.0", "$qubit.freq + 100.0", expts=801),
            "g_center": Expr("$qubit.ext.readout_cal.g_center"),
            "e_center": Expr("$qubit.ext.readout_cal.e_center"),
            "radius": Expr("$qubit.ext.readout_cal.radius"),
            "relax_delay": t1_initial_time(ctx.parameters, multiplier=5),
            "reps": 1000,
            "rounds": 10,
        },
    )
