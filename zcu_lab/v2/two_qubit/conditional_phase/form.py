"""Fixed-capture inputs for conditional phase calibration."""

from zcu_tools.cfg_model import Expr, Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.support.measure.mux_forms import mux_readout_seed_metadata

from .._measure import captured_roles
from .core import ConditionalPhaseCfg


def make_form(ctx: FormContext) -> Form[ConditionalPhaseCfg]:
    """Capture identities and copy calibrated pulses; absent CZ slots stay required.

    Missing readout may be explicitly calibrated in the complete mux template.
    Missing GE centers permit raw acquisition but analysis explains missing source
    calibration. No other experiment Form or simulator truth supplies inputs.
    """
    roles = captured_roles(ctx)
    readout = (
        Slot("$mux.module.readout")
        if ctx.has("$mux.module.readout")
        else Template("readout/mux:blank")
    )
    return Form(
        ConditionalPhaseCfg,
        field_order=(
            "roles",
            "x180_f1",
            "x180_f2",
            "x90_f1",
            "x90_f2",
            "readout",
            "drive",
            "target",
            "f1_centers",
            "f2_centers",
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
            "x180_f1": Slot("$f1.module.x180", copy=True),
            "x180_f2": Slot("$f2.module.x180", copy=True),
            "x90_f1": Slot("$f1.module.x90", copy=True),
            "x90_f2": Slot("$f2.module.x90", copy=True),
            "readout": readout,
            "drive": {
                # dual_line_drive replaces both frequencies, including repeated gates.
                "f1": Slot(
                    "$coupler.module.cz_f1", copy=True, set={"freq": Hidden(0.0)}
                ),
                "f2": Slot(
                    "$coupler.module.cz_f2", copy=True, set={"freq": Hidden(0.0)}
                ),
            },
            "frequency": Range.between(None, None, expts=101),
            **{
                f"{role}_centers": {
                    "g": Expr(f"${role}.ext.readout_cal.g_center"),
                    "e": Expr(f"${role}.ext.readout_cal.e_center"),
                }
                if ctx.parameters is not None
                and ctx.parameters.lookup(f"${role}.ext.readout_cal.g_center").present
                and ctx.parameters.lookup(f"${role}.ext.readout_cal.e_center").present
                else None
                for role in ("f1", "f2")
            },
        },
    ).with_seed_metadata(mux_readout_seed_metadata(readout))
