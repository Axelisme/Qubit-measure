"""Captured starting values for the MIST flux/gain map."""

from zcu_tools.cfg_model import Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.support.measure.ctx_helpers import t1_initial_time

from .core import FluxDepCfg


def make_form(ctx: FormContext) -> Form[FluxDepCfg]:
    """Seed optional preparation, resonator probe, readout and native flux sweep.

    Missing flux_source/calibration leave required fields invalid. Open/Reset
    captures sources; editing and Run do not refresh them or connect devices.
    """
    flux_source = (
        ctx.parameters.lookup_point("$qubit.flux_source").value
        if ctx.parameters is not None
        else None
    )
    if flux_source is not None and not isinstance(flux_source, str):
        raise ValueError("$qubit.flux_source must name a device")
    return Form(
        FluxDepCfg,
        tree={
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
            "flux_dev": flux_source,
            "flux_unit": "A",
            "flux": Range("$qubit.flux_int", "$qubit.flux_half", expts=101),
            "gain": Range(0.0, 0.5, expts=101),
            "relax_delay": t1_initial_time(ctx.parameters, multiplier=5),
            "reps": 1000,
            "rounds": 10,
        },
    )
