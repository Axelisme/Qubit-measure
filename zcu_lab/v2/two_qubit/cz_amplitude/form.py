"""Fixed-capture two-line CZ amplitude inputs, never borrowed from another Form."""

from zcu_tools.cfg_model import Expr, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2._support.measure.mux_forms import (
    mux_readout_initial,
    mux_readout_seed_metadata,
)

from .._measure import captured_roles
from .core import CZAmplitudeCfg


def make_form(ctx: FormContext) -> Form[CZAmplitudeCfg]:
    """Capture roles and editable calibrated pulses/readout/GE centers once.

    Coupler CZ slots supply ordinary pulse templates when present. Otherwise
    explicit cosine inputs require operator frequency/gain/duration/Nyquist zone;
    only generator wiring comes from capture. No simulator truth or drive preset
    supplies missing calibrations. A common scale range is required explicitly.
    """
    roles = captured_roles(ctx)
    drive = {}
    for role in ("f1", "f2"):
        slot = f"$coupler.module.cz_{role}"
        drive[role] = (
            Slot(slot, copy=True)
            if ctx.has(slot)
            else Template(
                "pulse:blank",
                set={
                    "ch": Expr(f"${role}.wiring.ch"),
                    "nqz": None,
                    "freq": None,
                    "gain": None,
                    "waveform.style": "cosine",
                    "waveform.length": None,
                },
            )
        )
    readout = mux_readout_initial(ctx)
    return Form(
        CZAmplitudeCfg,
        field_order=(
            "roles",
            "modules",
            "drive",
            "f1_centers",
            "f2_centers",
            "scales",
            "reps",
            "rounds",
            "relax_delay",
            "initial_delay",
        ),
        tree={
            "reps": 1000,
            "rounds": 10,
            "roles": roles,
            "drive": drive,
            "modules": {
                "x180_f1": Slot("$f1.module.x180", copy=True),
                "x180_f2": Slot("$f2.module.x180", copy=True),
                "readout": readout,
            },
            "f1_centers": {
                "g": Expr("$f1.ext.readout_cal.g_center"),
                "e": Expr("$f1.ext.readout_cal.e_center"),
            },
            "f2_centers": {
                "g": Expr("$f2.ext.readout_cal.g_center"),
                "e": Expr("$f2.ext.readout_cal.e_center"),
            },
            "scales": Range.between(None, None, expts=101),
        },
    ).with_seed_metadata(mux_readout_seed_metadata(readout))
