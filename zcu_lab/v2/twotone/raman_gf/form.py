"""Operator carrier and sampled-envelope inputs; no simulator spectrum lookup."""

from zcu_tools.cfg_model import Expr, Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.support.measure.ctx_helpers import t1_initial_time
from zcu_lab.v2.support.measure.seed_metadata import pulse_readout_seed_metadata

from .core import RamanGFCfg


def make_form(ctx: FormContext) -> Form[RamanGFCfg]:
    """Leave the measured carrier blank and require a sampled waveform asset.

    Frequency mode holds the gain sweep's start; area mode holds the detuning
    sweep's start. No calibration is inferred from simulator models.
    """
    return Form(
        RamanGFCfg,
        tree={
            "scan_mode": "frequency_area",
            "detuning": Range(-50, 50, expts=11),
            "gain": Range(0, 0.9, expts=10),
            "drive": Template(
                "pulse:blank",
                set={
                    "ch": Expr("$qubit.wiring.ch"),
                    "freq": Hidden(0.0),
                    "gain": Hidden(0.0),
                    "waveform.style": "arb",
                },
            ),
            "readout": Slot("$resonator.module.readout", fallback=Template("readout", bindings={"resonator": "$resonator"}, seed_metadata=pulse_readout_seed_metadata(gain=0.1))),
            "shots": 1024,
            "relax_delay": t1_initial_time(ctx.parameters, multiplier=5),
            "initial_delay": Hidden(),
        },
    )
