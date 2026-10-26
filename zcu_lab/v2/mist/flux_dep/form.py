"""Captured starting values for the MIST flux/gain map."""

from functools import partial

from zcu_tools.cfg_model import Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.support.measure.ctx_helpers import t1_initial_time
from zcu_lab.v2.support.measure.flux_defaults import qubit_flux_initial_values

from .core import FluxDepCfg


def make_form(ctx: FormContext) -> Form[FluxDepCfg]:
    """Seed optional preparation, resonator probe, readout and native flux sweep.

    Missing flux_source/calibration leave required fields invalid. Open/Reset
    captures sources; editing and Run do not refresh them or connect devices.
    """
    return Form(
        FluxDepCfg,
        initial_values=partial(qubit_flux_initial_values, margin=0.0),
        tree={
            "reset": Slot("$qubit.module.reset"),
            "init_pulse": Slot("$qubit.module.x180"),
            "probe_pulse": Template(
                "res_probe",
                bindings={"resonator": "$resonator"},
                # Run supplies these probe scan coordinates.
                set={"gain": Hidden(0.0)},
            ),
            "readout": Slot("$resonator.module.readout")
            if ctx.has("$resonator.module.readout")
            else Template("readout", bindings={"resonator": "$resonator"}),
            "flux": Range(None, None, expts=101),
            "gain": Range(0.0, 0.5, expts=101),
            "relax_delay": t1_initial_time(ctx.parameters, multiplier=5),
            "reps": 1000,
            "rounds": 10,
        },
    )
