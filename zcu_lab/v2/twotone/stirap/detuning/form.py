"""Detuning controls with measured carriers and independently selected envelopes."""

from zcu_tools.cfg_model import Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.support.measure.ctx_helpers import source, t1_initial_time
from zcu_lab.v2.support.measure.seed_metadata import pulse_readout_seed_metadata

from .core import StirapDetuningCfg


def make_form(ctx: FormContext) -> Form[StirapDetuningCfg]:
    """Seed both Gaussians at sigma 0.05 us and length 0.3 us, not carrier guesses.

    Pump/Stokes gains, measured carriers and detuning ranges remain required.
    Readout comes from the resonator slot when available; centroid3 calibration
    is required for Run and must be obtained separately.
    """
    return Form(
        StirapDetuningCfg,
        tree={
            "pump_waveform": Template(
                "waveform:gauss", set={"sigma": 0.05, "length": 0.3}
            ),
            "stokes_waveform": Template(
                "waveform:gauss", set={"sigma": 0.05, "length": 0.3}
            ),
            "readout": source(
                ctx,
                "$resonator.module.readout",
                Template(
                    "readout",
                    bindings={"resonator": "$resonator"},
                    seed_metadata=pulse_readout_seed_metadata(gain=0.1),
                ),
            ),
            "relax_delay": t1_initial_time(ctx.parameters, multiplier=5),
        },
    )
