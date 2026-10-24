"""Flux-dependent spectroscopy starting values and named device selection."""

from zcu_tools.cfg_model import FieldMeta, Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2._support.measure.pulse_readout_metadata import (
    pulse_readout_seed_metadata,
)

from .core import FreqFluxCfg


def make_form(ctx: FormContext) -> Form[FreqFluxCfg]:
    """Build starting values and shape from the captured source availability."""
    flux_source = (
        ctx.parameters.lookup_point("$qubit.flux_source").value
        if ctx.parameters is not None
        else None
    )
    if flux_source is not None and not isinstance(flux_source, str):
        raise ValueError("$qubit.flux_source must name a device")
    return Form(
        FreqFluxCfg,
        field_order=(
            "reset",
            "init_pulse",
            "qub_pulse",
            "readout",
            "flux_dev",
            "flux_unit",
            "flux",
            "freq",
            "reps",
            "rounds",
            "relax_delay",
        ),
        tree={
            "reset": Slot("$qubit.module.reset"),
            "init_pulse": Hidden(),
            "qub_pulse": Template(
                "qub_probe",
                bindings={"qubit": "$qubit"},
                set={"freq": Hidden(0.0)},
            ),
            "readout": Slot("$qubit.module.readout")
            if ctx.has("$qubit.module.readout")
            else Template("readout", bindings={"resonator": "$resonator"}),
            "flux_dev": flux_source,
            "flux_unit": "A",
            "freq": Range.between(None, None, expts=1001),
            "flux": Range.between(
                "1.1 * $qubit.flux_int - 0.1 * $qubit.flux_half",
                "1.1 * $qubit.flux_half - 0.1 * $qubit.flux_int",
                expts=101,
            ),
            "relax_delay": 1.0,
            "reps": 1000,
            "rounds": 10,
            "fail_retry": Hidden(),
        },
    ).with_seed_metadata(
        {
            **(
                {}
                if ctx.has("$qubit.module.readout")
                else pulse_readout_seed_metadata(gain=0.1)
            ),
            **{
                ("qub_pulse", "module", "gain"): FieldMeta(
                    description="未校準的起始值，來源：固定常數。qub_probe Template gain 0.1 normalized; operator confirmation required",
                    calibration="uncalibrated",
                )
            },
        }
    )
