"""Captured starting values for flux-pulsed qubit spectroscopy."""

from zcu_tools.cfg_model import Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from .core import TwotoneCfg


def make_form(ctx: FormContext) -> Form[TwotoneCfg]:
    """Capture pulse sources and start with a one-point zero-gain sweep."""
    return Form(
        TwotoneCfg,
        tree={
            "reset": Slot("$qubit.module.reset"),
            "flux_pulse": Template(
                "flux_pulse",
                bindings={"qubit": "$qubit"},
                # Run supplies flux gain, independent of probe frequency.
                set={"gain": Hidden(0.0), "waveform.length": 1.0},
            ),
            "qub_pulse": Template(
                "qub_probe", bindings={"qubit": "$qubit"}, set={"freq": Hidden(0.0)}
            ),
            "readout": Slot("$resonator.module.readout")
            if ctx.has("$resonator.module.readout")
            else Template("readout", bindings={"resonator": "$resonator"}),
            "gain": Range(0.0, 0.0, expts=1),
            "freq": Range.around("$qubit.freq", "1.5 * $qubit.kappa", expts=301),
            "relax_delay": 1.0,
            "reps": 1000,
            "rounds": 10,
        },
    )
