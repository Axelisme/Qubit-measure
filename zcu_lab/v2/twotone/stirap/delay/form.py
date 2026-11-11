"""Captured-source Form seeds for the STIRAP delay experiment."""

from zcu_tools.cfg_model import Range, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.support.measure.ctx_helpers import source

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
                Template("readout", bindings={"resonator": "$resonator"}),
            ),
            "stokes_lead": Range(-0.1, 0.1, expts=41),
        },
    )
