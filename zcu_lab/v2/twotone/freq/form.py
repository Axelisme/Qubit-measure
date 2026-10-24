"""Starting values and form deviations for twotone/freq; Qt-free data only."""

from zcu_tools.cfg_model import FieldMeta, Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.support.measure.seed_metadata import (
    pulse_readout_seed_metadata,
)
from zcu_lab.v2.twotone.freq.core import FreqCfg


def make_form(ctx: FormContext) -> Form[FreqCfg]:
    """Build starting values and shape from the captured source availability."""
    return Form(
        FreqCfg,
        field_order=(
            "reset",
            "init_pulse",
            "qub_pulse",
            "readout",
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
                # Run replaces probe frequency with the scan.
                set={"freq": Hidden(0.0), "gain": 0.1, "waveform.length": 1.0},
            ),
            "readout": Slot("$qubit.module.readout")
            if ctx.has("$qubit.module.readout")
            else Template("readout", bindings={"resonator": "$resonator"}),
            "freq": Range.around("$qubit.freq", "20 * $qubit.kappa", expts=301),
            "relax_delay": 1.0,
            "reps": 1000,
            "rounds": 10,
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
