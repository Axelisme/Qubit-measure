"""STIRAP scan controls with operator-supplied spectroscopy carriers."""

from zcu_tools.cfg_model import Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.support.measure.ctx_helpers import t1_initial_time
from zcu_lab.v2.support.measure.seed_metadata import pulse_readout_seed_metadata

from .core import StirapCfg


def make_form(ctx: FormContext) -> Form[StirapCfg]:
    """Capture readout and expose three scan modes without seeding carriers.

    Pump/Stokes gains and carriers remain required. Select a scan mode and
    enter its endpoints and point counts; delay scans should span both signs.
    """
    return Form(
        StirapCfg,
        tree={
            "readout": Slot(
                "$resonator.module.readout",
                fallback=Template(
                    "readout",
                    bindings={"resonator": "$resonator"},
                    seed_metadata=pulse_readout_seed_metadata(gain=0.1),
                ),
            ),
            "relax_delay": t1_initial_time(ctx.parameters, multiplier=5),
        },
    )
