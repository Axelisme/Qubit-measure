"""AC Stark Ramsey starting values from fixed captures."""

from zcu_tools.cfg_model import Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.support.measure.ctx_helpers import source

from .core import AcStarkRamseyCfg


def make_form(ctx: FormContext) -> Form[AcStarkRamseyCfg]:
    """Build a fixed Stark tone and a swept Ramsey delay, with detune in MHz."""
    return Form(
        AcStarkRamseyCfg,
        tree={
            "reset": Slot("$qubit.module.reset"),
            "stark_pulse": Template(
                "res_probe",
                bindings={"resonator": "$resonator"},
                # Run supplies the Stark gain scan; duration remains fixed.
                set={"gain": Hidden(0.0), "waveform.length": 10.0},
            ),
            "pi2_pulse": source(
                ctx,
                "$qubit.module.x90",
                Template("pi2_pulse", bindings={"qubit": "$qubit"}),
            ),
            "readout": source(
                ctx,
                "$resonator.module.readout",
                Template("readout", bindings={"resonator": "$resonator"}),
            ),
            "gain": Range(0.0, 0.1, expts=31),
            "delay": Range(0.0, 1.0, expts=101),
            "wait_delay": 0.1,
            "detune": 1.0,
            "relax_delay": 1.0,
            "reps": 1000,
            "rounds": 10,
        },
    )
