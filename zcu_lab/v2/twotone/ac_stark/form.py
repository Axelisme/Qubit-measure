"""AC Stark spectroscopy starting values from fixed captures."""

from zcu_tools.cfg_model import Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.support.measure.ctx_helpers import source

from .core import AcStarkCfg


def make_form(ctx: FormContext) -> Form[AcStarkCfg]:
    """Build gain/frequency spectroscopy with independently editable tone timing."""
    return Form(
        AcStarkCfg,
        tree={
            "reset": Slot("$qubit.module.reset"),
            "stark_tone": Template(
                "res_probe",
                bindings={"resonator": "$resonator"},
                # Run supplies the Stark gain scan.
                set={"gain": Hidden(0.0), "waveform.length": 1.0},
            ),
            "probe": Template(
                "qub_probe",
                bindings={"qubit": "$qubit"},
                set={"freq": Hidden(0.0)},
            ),
            "readout": source(
                ctx,
                "$resonator.module.readout",
                Template("readout", bindings={"resonator": "$resonator"}),
            ),
            "gain": Range(0.0, 0.1, expts=31),
            "freq": Range.around("$qubit.freq", "1.5 * $qubit.kappa", expts=201),
            "relax_delay": 1.0,
            "reps": 1000,
            "rounds": 10,
        },
    )
