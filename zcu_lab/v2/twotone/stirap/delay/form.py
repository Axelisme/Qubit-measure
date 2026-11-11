"""Captured-source Form seeds for the STIRAP delay experiment."""

from zcu_tools.cfg_model import Range, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.support.measure.ctx_helpers import source, t1_initial_time
from zcu_lab.v2.support.measure.seed_metadata import pulse_readout_seed_metadata

from .core import StirapDelayCfg


def make_form(ctx: FormContext) -> Form[StirapDelayCfg]:
    """Seed independent Gaussian envelopes and leave measured carriers required."""
    return Form(
        StirapDelayCfg,
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
            "stokes_lead": Range(-0.1, 0.1, expts=41),
            "relax_delay": t1_initial_time(ctx.parameters, multiplier=5),
        },
    )
