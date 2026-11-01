"""Captured calibration sources for the three reference preparations."""

from zcu_tools.cfg_model import Hidden, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.support.measure.ctx_helpers import t1_initial_time
from zcu_lab.v2.support.measure.seed_metadata import pulse_readout_seed_metadata

from .core import QutritPopulationCfg


def make_form(ctx: FormContext) -> Form[QutritPopulationCfg]:
    """Open required pulse editors without model-derived frequency defaults.

    Pi 01 may use a saved x180 slot. Pi 03/Pi 32 start from an uncalibrated
    pulse template using captured qubit sources, never a model spectrum.
    The integer-fluxonium entry has no frequency calibration, so these are blank.
    """
    return Form(
        QutritPopulationCfg,
        field_order=(
            "pi01",
            "pi03",
            "pi32",
            "readout",
            "shots",
            "relax_delay",
            "initial_delay",
        ),
        tree={
            "pi01": Slot(
                "$qubit.module.x180",
                fallback=Template("pi_pulse", bindings={"qubit": "$qubit"}),
            ),
            "pi03": Template("pi_pulse", bindings={"qubit": "$qubit"}),
            "pi32": Template("pi_pulse", bindings={"qubit": "$qubit"}),
            "readout": Slot(
                "$resonator.module.readout",
                fallback=Template(
                    "readout",
                    bindings={"resonator": "$resonator"},
                    seed_metadata=pulse_readout_seed_metadata(gain=0.1),
                ),
            ),
            "shots": 4096,
            "relax_delay": t1_initial_time(ctx.parameters, multiplier=5),
            "initial_delay": Hidden(),
        },
    )
