"""Fixed-capture duration scan with explicit two-line drive and GE calibration inputs."""

from dataclasses import replace

from zcu_tools.cfg_model import Expr, Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.support.measure.ctx_helpers import source
from zcu_lab.v2.support.measure.mux_forms import (
    mux_readout_initial,
)

from .._measure import captured_roles
from .core import ChevronCfg


def make_form(ctx: FormContext) -> Form[ChevronCfg]:
    """Capture roles and pulses; scan endpoints require operator input.

    Existing CZ slots are editable source pairs, not certified gate settings.
    Missing slots offer explicit cosine templates on captured role wiring.
    Missing or linear role-tone classifiers, or missing radii, stay Invalid. Run GE Post first. Open/Reset never performs acquisition.
    """
    roles = captured_roles(ctx)
    drive = {}
    for role in ("f1", "f2"):
        slot = f"$coupler.module.cz_{role}"
        # Run replaces both line frequencies with the common scan coordinate.
        node = source(
            ctx,
            slot,
            Template(
                "pulse:blank",
                set={
                    "ch": Expr(f"${role}.wiring.ch"),
                    "nqz": None,
                    "freq": Hidden(0.0),
                    "gain": None,
                    "waveform.style": "cosine",
                    "waveform.length": None,
                },
            ),
        )
        if isinstance(node, Slot):
            node = replace(node, copy=True, set={"freq": Hidden(0.0)})
        drive[role] = node
    readout = mux_readout_initial(ctx)
    return Form(
        ChevronCfg,
        tree={
            "reps": 1000,
            "rounds": 10,
            "relax_delay": 1.0,
            "roles": roles,
            "reset": None,
            "x180_f1": Slot("$f1.module.x180", copy=True),
            "x180_f2": Slot("$f2.module.x180", copy=True),
            "readout": readout,
            "drive": drive,
            "frequency": Range(None, None, expts=25),
            "duration": Range(None, None, expts=101),
        },
    )
