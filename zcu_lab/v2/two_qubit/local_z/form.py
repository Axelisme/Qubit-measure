"""Fixed-capture inputs for a two-target local-Z experiment."""

from zcu_tools.cfg_model import Slot
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.support.measure.mux_forms import (
    mux_readout_initial,
    mux_readout_seed_metadata,
)

from .._measure import captured_roles
from .core import LocalZCfg


def make_form(ctx: FormContext) -> Form[LocalZCfg]:
    """Capture identities and copy calibrated pulses; absent CZ slots stay required.

    Missing readout may be explicitly calibrated in the complete mux template.
    Centers derive only from saved role-tone circle classifiers. Missing or linear
    calibration permits raw acquisition; analysis returns a calibration reason.
    No other experiment Form or simulator truth supplies inputs.
    """
    roles = captured_roles(ctx)
    readout = mux_readout_initial(ctx)
    return Form(
        LocalZCfg,
        field_order=(
            "roles",
            "x90_f1",
            "x90_f2",
            "readout",
            "drive",
            "gate_counts",
            "reps",
            "rounds",
            "relax_delay",
            "initial_delay",
        ),
        tree={
            "reps": 1000,
            "rounds": 10,
            "roles": roles,
            "x90_f1": Slot("$f1.module.x90", copy=True),
            "x90_f2": Slot("$f2.module.x90", copy=True),
            "readout": readout,
            "drive": {
                "f1": Slot("$coupler.module.cz_f1", copy=True),
                "f2": Slot("$coupler.module.cz_f2", copy=True),
            },
        },
    ).with_seed_metadata(mux_readout_seed_metadata(readout))
