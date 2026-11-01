"""Starting values for singleshot/len_rabi; sources are captured at Open/Reset."""

from dataclasses import replace

from zcu_tools.cfg_model import Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.singleshot.len_rabi.core import LenRabiCfg
from zcu_lab.v2.support.measure.ctx_helpers import source, t1_initial_time
from zcu_lab.v2.support.measure.rabi_initial_values import len_rabi_length_stop

from zcu_lab.v2.support.measure.seed_metadata import qub_probe_gain_seed_metadata


def make_form(ctx: FormContext) -> Form[LenRabiCfg]:
    """Build starting values and shape from the captured source availability."""
    length_stop = len_rabi_length_stop(ctx.parameters)
    qub_pulse = source(
        ctx,
        "$qubit.module.x180",
        Template(
            "qub_probe",
            bindings={"qubit": "$qubit"},
            set={"gain": 1.0},
            seed_metadata=qub_probe_gain_seed_metadata(
                ("qub_pulse", "module", "gain")
            ),
        ),
    )
    # This Run accepts only const/flat_top and replaces total pulse duration.
    qub_pulse = replace(
        qub_pulse,
        set={**qub_pulse.set, "waveform.length": Hidden(0.0)},
    )
    return Form(
        LenRabiCfg,
        tree={
            "reset": Slot("$qubit.module.reset"),
            "qub_pulse": qub_pulse,
            "readout": source(
                ctx,
                "$resonator.module.readout",
                Template("readout", bindings={"resonator": "$resonator"}),
            ),
            "length": Range(0.03, length_stop, expts=51),
            "relax_delay": t1_initial_time(ctx.parameters, multiplier=5),
            "shots": 1000,
            "initial_delay": Hidden(),
        },
    )
