"""AC Stark spectroscopy and Ramsey starting values from fixed captures."""

from zcu_tools.cfg_model import Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from .core import AcStarkCfg, AcStarkRamseyCfg


def make_form(ctx: FormContext) -> Form[AcStarkCfg]:
    """Build gain/frequency spectroscopy with independently editable tone timing."""
    return Form(
        AcStarkCfg,
        tree={
            "modules": {
                "reset": Slot("$qubit.module.reset"),
                "stark_pulse1": Template(
                    "res_probe",
                    bindings={"resonator": "$resonator"},
                    # Run supplies the Stark gain scan.
                    set={"gain": Hidden(0.0), "waveform.length": 1.0},
                ),
                # Run supplies the spectroscopy frequency scan.
                "stark_pulse2": Template(
                    "qub_probe",
                    bindings={"qubit": "$qubit"},
                    set={"freq": Hidden(0.0)},
                ),
                "readout": Slot("$qubit.module.readout")
                if ctx.has("$qubit.module.readout")
                else Template("readout", bindings={"resonator": "$resonator"}),
            },
            "sweep": {
                "gain": Range(0.0, 0.1, expts=21),
                "freq": Range.around("$qubit.freq", "1.5 * $qubit.kappa", expts=201),
            },
            "relax_delay": 1.0,
            "reps": 1000,
            "rounds": 10,
            "initial_delay": Hidden(),
        },
    )


def make_ramsey_form(ctx: FormContext) -> Form[AcStarkRamseyCfg]:
    """Build a fixed Stark tone and a swept Ramsey delay, with detune in MHz."""
    return Form(
        AcStarkRamseyCfg,
        tree={
            "modules": {
                "reset": Slot("$qubit.module.reset"),
                "stark_pulse": Template(
                    "res_probe",
                    bindings={"resonator": "$resonator"},
                    # Run supplies the Stark gain scan; duration remains fixed.
                    set={"gain": Hidden(0.0), "waveform.length": 10.0},
                ),
                "pi2_pulse": Slot("$qubit.module.x90")
                if ctx.has("$qubit.module.x90")
                else Template("pi2_pulse", bindings={"qubit": "$qubit"}),
                "readout": Slot("$qubit.module.readout")
                if ctx.has("$qubit.module.readout")
                else Template("readout", bindings={"resonator": "$resonator"}),
            },
            "sweep": {
                "gain": Range(0.0, 0.1, expts=21),
                "length": Range(0.0, 1.0, expts=101),
            },
            "wait_delay": 0.1,
            "detune": 1.0,
            "relax_delay": 1.0,
            "reps": 1000,
            "rounds": 10,
            "initial_delay": Hidden(),
        },
    )
