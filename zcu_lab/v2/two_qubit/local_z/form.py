"""Fixed-capture inputs for a two-target local-Z experiment."""

from zcu_tools.cfg_model import Slot
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.support.measure.mux_forms import (
    mux_readout_initial,
)

from .._measure import captured_roles
from .core import LocalZCfg


def make_form(ctx: FormContext) -> Form[LocalZCfg]:
    """Capture role identities and seed calibrated pulses.

    Missing CZ slots stay required. Both role tones need circle + radius from
    GE Post before Run. No other experiment Form or simulator truth supplies inputs.
    reset starts as None and can be authored explicitly.
    """
    roles = captured_roles(ctx)
    readout = mux_readout_initial(ctx)
    return Form(
        LocalZCfg,
        tree={
            "reps": 1000,
            "rounds": 10,
            "relax_delay": 1.0,
            "roles": roles,
            "reset": None,
            "x90_f1": Slot("$f1.module.x90", copy=True),
            "x90_f2": Slot("$f2.module.x90", copy=True),
            "readout": readout,
            "drive": {
                "f1": Slot("$coupler.module.cz_f1", copy=True),
                "f2": Slot("$coupler.module.cz_f2", copy=True),
            },
        },
    )
