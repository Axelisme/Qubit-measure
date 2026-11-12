"""Starting values for chain/len_rabi; Qt-free data only."""

from zcu_tools.cfg_model import Range, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.support.measure.ctx_helpers import source, t1_initial_time
from zcu_lab.v2.support.measure.rabi_initial_values import len_rabi_length_stop
from zcu_lab.v2.support.measure.seed_metadata import (
    pulse_readout_seed_metadata,
    qub_probe_gain_seed_metadata,
)

from .core import ChainLenRabiCfg


def make_form(ctx: FormContext) -> Form[ChainLenRabiCfg]:
    """Use a fresh const qub_probe pulse and the twotone/rabi/len_rabi range.

    The pulse starts from the qub_probe template rather than x180, because the
    FPGA length sweep accepts only const or flat_top waveforms. K is 5 T1;
    reps and rounds start at 100.
    """
    return Form(
        ChainLenRabiCfg,
        tree={
            "reps": 100,
            "rounds": 100,
            "qub_pulse": Template(
                "qub_probe",
                bindings={"qubit": "$qubit"},
                set={"gain": 1.0},
                seed_metadata=qub_probe_gain_seed_metadata(
                    ("qub_pulse", "module", "gain")
                ),
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
            "length": Range(0.03, len_rabi_length_stop(ctx.parameters), expts=101),
            "relax_delay": t1_initial_time(ctx.parameters, multiplier=5),
        },
    )
