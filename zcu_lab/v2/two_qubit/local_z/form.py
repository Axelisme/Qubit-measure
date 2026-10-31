"""Fixed-capture inputs for a two-target local-Z experiment."""

from typing import TypeVar

from zcu_tools.cfg_model import Slot
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.support.measure.mux_forms import (
    mux_readout_initial,
    mux_readout_seed_metadata,
)

from .._measure import captured_roles
from .core import LocalZCfg


CfgT = TypeVar("CfgT", bound=LocalZCfg)


def make_form(ctx: FormContext) -> Form[LocalZCfg]:
    """Capture roles and seed the ordinary classified phase experiment."""
    return phase_form(ctx, LocalZCfg)


def phase_form(ctx: FormContext, cfg_type: type[CfgT]) -> Form[CfgT]:
    """Capture role identities and seed calibrated pulses for either variant.

    Missing CZ slots stay required. Both role tones need circle + radius from
    GE Post before Run. No other experiment Form or simulator truth supplies inputs.
    reset starts as None and can be authored explicitly.
    """
    roles = captured_roles(ctx)
    readout = mux_readout_initial(ctx)
    return Form(
        cfg_type,
        field_order=(
            "roles",
            "x90_f1",
            "x90_f2",
            "readout",
            "reset",
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
            "reset": None,
            "x90_f1": Slot("$f1.module.x90", copy=True),
            "x90_f2": Slot("$f2.module.x90", copy=True),
            "readout": readout,
            "drive": {
                "f1": Slot("$coupler.module.cz_f1", copy=True),
                "f2": Slot("$coupler.module.cz_f2", copy=True),
            },
        },
    ).with_seed_metadata(mux_readout_seed_metadata(readout))
