"""Fixed-capture duration scan with explicit two-line drive and GE calibration inputs."""

from zcu_tools.cfg_model import Expr, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.support.measure.mux_forms import mux_readout_seed_metadata

from .._measure import captured_roles
from .core import ConstructivePhaseCfg


def make_form(ctx: FormContext) -> Form[ConstructivePhaseCfg]:
    """Capture roles, pulses and GE centers; scan endpoints require operator input.

    Existing CZ slots are editable source pairs, not certified gate settings.
    Missing slots offer explicit cosine templates on captured role wiring.
    Missing calibration stays Invalid. Open/Reset never performs acquisition.
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
    readout = (
        Slot("$mux.module.readout")
        if ctx.has("$mux.module.readout")
        else Template("readout/mux:blank")
    )
    return Form(
        ConstructivePhaseCfg,
        field_order=(
            "roles",
            "x180_f1",
            "x180_f2",
            "readout",
            "drive",
            "f1_centers",
            "f2_centers",
            "duration",
            "relative_phase",
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
            "readout": readout,
            "drive": drive,
            "relative_phase": Range.between(None, None, expts=25),
            "duration": Range.between(None, None, expts=101),
            "f1_centers": {
                "g": Expr("$f1.ext.readout_cal.g_center"),
                "e": Expr("$f1.ext.readout_cal.e_center"),
            },
            "f2_centers": {
                "g": Expr("$f2.ext.readout_cal.g_center"),
                "e": Expr("$f2.ext.readout_cal.e_center"),
            },
        },
    ).with_seed_metadata(mux_readout_seed_metadata(readout))
