"""Captured starting values for the MIST flux/gain map."""

from zcu_tools.cfg_model import AliasName, Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2._support.measure.ctx_helpers import t1_initial_time

from .core import FluxDepCfg


def make_form(ctx: FormContext) -> Form[FluxDepCfg]:
    """Seed optional preparation, resonator probe, readout and native flux sweep.

    Missing flux aliases/calibration leave required fields invalid. Open/Reset
    captures sources; editing and Run do not refresh them or connect devices.
    """
    return Form(
        FluxDepCfg,
        tree={
            "modules": {
                "reset": Slot("$qubit.module.reset"),
                "init_pulse": Slot("$qubit.module.x180"),
                "probe_pulse": Template(
                    "res_probe",
                    bindings={"resonator": "$resonator"},
                    # Run supplies these probe scan coordinates.
                    set={"gain": Hidden(0.0)},
                ),
                "readout": Slot("$qubit.module.readout")
                if ctx.has("$qubit.module.readout")
                else Template("readout", bindings={"resonator": "$resonator"}),
            },
            "flux_dev": AliasName("flux_dev"),
            "flux_unit": "A",
            "sweep": {
                "flux": Range.between("$qubit.flux_int", "$qubit.flux_half", expts=101),
                "gain": Range.between(0.0, 0.5, expts=101),
            },
            "relax_delay": t1_initial_time(ctx.parameters, multiplier=5),
            "reps": 1000,
            "rounds": 1,
            "initial_delay": Hidden(),
        },
    )
