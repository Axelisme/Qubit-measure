"""Experiment starting values from fixed captures and injected templates."""

from zcu_tools.cfg_model import FieldMeta, Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.support.measure.ctx_helpers import t1_initial_time
from zcu_lab.v2.support.measure.rabi_initial_values import len_rabi_initial_values
from zcu_lab.v2.support.measure.seed_metadata import (
    pulse_readout_seed_metadata,
)

from .core import LenRabiCfg


def make_form(ctx: FormContext) -> Form[LenRabiCfg]:
    """Build starting values and shape from the captured source availability."""
    qub_pulse, length_stop = len_rabi_initial_values(ctx.parameters)
    return Form(
        LenRabiCfg,
        field_order=(
            "reset",
            "init_pulse",
            "qub_pulse",
            "readout",
            "length",
            "reps",
            "rounds",
            "acquire_retry",
            "earlystop_snr",
            "relax_delay",
        ),
        tree={
            "reset": Slot("$qubit.module.reset"),
            "qub_pulse": qub_pulse,
            "readout": Slot("$resonator.module.readout")
            if ctx.has("$resonator.module.readout")
            else Template("readout", bindings={"resonator": "$resonator"}),
            "init_pulse": Hidden(),
            "length": Range(0.03, length_stop, expts=101),
            "relax_delay": t1_initial_time(ctx.parameters, multiplier=5),
            "reps": 1000,
            "rounds": 10,
        },
    ).with_seed_metadata(
        {
            **(
                {}
                if ctx.has("$resonator.module.readout")
                else pulse_readout_seed_metadata(gain=0.1)
            ),
            **(
                {
                    ("qub_pulse", "module", "gain"): FieldMeta(
                        description="未校準的起始值，來源：固定常數。qub_probe Template gain 1.0 a.u. within [-1, 1]; operator confirmation required",
                        calibration="uncalibrated",
                    )
                }
                if isinstance(qub_pulse, Template)
                else {}
            ),
        }
    )
