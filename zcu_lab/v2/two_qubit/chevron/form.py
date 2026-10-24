"""Fixed-capture duration scan with explicit two-line drive and GE calibration inputs."""

from zcu_tools.cfg_model import Expr, Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.support.measure.mux_forms import (
    mux_readout_initial,
    mux_readout_seed_metadata,
)

from .._measure import captured_roles
from .core import ChevronCfg


def make_form(ctx: FormContext) -> Form[ChevronCfg]:
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
            # Run replaces both line frequencies with the common scan coordinate.
            Slot(slot, copy=True, set={"freq": Hidden(0.0)})
            if ctx.has(slot)
            else Template(
                "pulse:blank",
                set={
                    "ch": Expr(f"${role}.wiring.ch"),
                    "nqz": None,
                    "freq": Hidden(0.0),
                    "gain": None,
                    "waveform.style": "cosine",
                    "waveform.length": None,
                },
            )
        )
    readout = mux_readout_initial(ctx)
    return Form(
        ChevronCfg,
        field_order=(
            "roles",
            "x180_f1",
            "x180_f2",
            "readout",
            "drive",
            "f1_centers",
            "f2_centers",
            "frequency",
            "duration",
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
            "frequency": Range(None, None, expts=25),
            "duration": Range(None, None, expts=101),
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
