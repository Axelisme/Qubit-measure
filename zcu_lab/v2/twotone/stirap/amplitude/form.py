"""Amplitude controls from captured sources and explicit envelope seeds."""

from zcu_tools.cfg_model import Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.support.measure.ctx_helpers import source, t1_initial_time
from zcu_lab.v2.support.measure.seed_metadata import pulse_readout_seed_metadata

from .core import StirapAmplitudeCfg


def make_form(ctx: FormContext) -> Form[StirapAmplitudeCfg]:
    """Seed Gaussian envelopes, not measured carriers, gains or scale bounds.

    Both Gaussians start at sigma 0.05 us and length 0.3 us. Use the captured
    resonator readout slot when present; otherwise seed a readout template.
    A centroid3 calibration is required separately before Run.
    """
    return Form(
        StirapAmplitudeCfg,
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
