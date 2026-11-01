"""Fixed-capture inputs for conditional phase calibration."""

from typing import TypeVar

from zcu_tools.cfg_model import Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.support.measure.mux_forms import mux_readout_seed_metadata

from .._measure import captured_roles
from .core import ConditionalPhaseCfg

CfgT = TypeVar("CfgT", bound=ConditionalPhaseCfg)


def make_form(ctx: FormContext) -> Form[ConditionalPhaseCfg]:
    """Capture roles and seed the ordinary classified phase experiment."""
    return phase_form(ctx, ConditionalPhaseCfg)


def phase_form(ctx: FormContext, cfg_type: type[CfgT]) -> Form[CfgT]:
    """Capture role identities and seed calibrated pulses for either variant.

    Missing CZ slots stay required. Both role tones need circle + radius from
    GE Post before Run. No other experiment Form or simulator truth supplies inputs.
    reset starts as None and can be authored explicitly.
    """
    roles = captured_roles(ctx)
    readout = Slot("$mux.module.readout", fallback=Template("readout/mux:blank"))
    return Form(
        cfg_type,
        field_order=(
            "roles",
            "x180_f1",
            "x180_f2",
            "x90_f1",
            "x90_f2",
            "readout",
            "reset",
            "drive",
            "target",
            "frequency",
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
            "x180_f1": Slot("$f1.module.x180", copy=True),
            "x180_f2": Slot("$f2.module.x180", copy=True),
            "x90_f1": Slot("$f1.module.x90", copy=True),
            "x90_f2": Slot("$f2.module.x90", copy=True),
            "readout": readout,
            "drive": {
                # Run overrides both CZ frequencies with the common hardware sweep.
                "f1": Slot(
                    "$coupler.module.cz_f1", copy=True, set={"freq": Hidden(0.0)}
                ),
                "f2": Slot(
                    "$coupler.module.cz_f2", copy=True, set={"freq": Hidden(0.0)}
                ),
            },
            "frequency": Range(None, None, expts=101),
        },
    ).with_seed_metadata(mux_readout_seed_metadata(readout))
