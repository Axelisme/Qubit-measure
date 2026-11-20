"""Fixed-capture two-line CZ amplitude inputs, never borrowed from another Form."""

from dataclasses import replace

from zcu_tools.cfg_model import Expr, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.support.measure.ctx_helpers import source
from zcu_lab.v2.support.measure.mux_forms import (
    mux_readout_initial,
)

from .._measure import captured_roles
from .core import CZAmplitudeCfg


def make_form(ctx: FormContext) -> Form[CZAmplitudeCfg]:
    """Capture roles and editable calibrated pulses/readout once.

    F1/F2 require circle classifiers with radius from GE Post. Reset starts None.

    Coupler CZ slots supply ordinary pulse templates when present. Otherwise
    explicit cosine inputs require operator frequency/gain/duration/Nyquist zone;
    only generator wiring comes from capture. No simulator truth or drive preset
    supplies missing calibrations. A common scale range is required explicitly.
    """
    roles = captured_roles(ctx)
    drive = {}
    for role in ("f1", "f2"):
        slot = f"$coupler.module.cz_{role}"
        node = source(
            ctx,
            slot,
            Template(
                "pulse:blank",
                set={
                    "ch": Expr(f"${role}.wiring.ch"),
                    "nqz": None,
                    "freq": None,
                    "gain": None,
                    "waveform.style": "cosine",
                    "waveform.length": None,
                },
            ),
        )
        if isinstance(node, Slot):
            node = replace(node, copy=True)
        drive[role] = node
    readout = mux_readout_initial(ctx)
    return Form(
        CZAmplitudeCfg,
        tree={
            "reps": 1000,
            "rounds": 10,
            "relax_delay": 1.0,
            "roles": roles,
            "reset": None,
            "drive": drive,
            "x180_f1": Slot("$f1.module.x180", copy=True),
            "x180_f2": Slot("$f2.module.x180", copy=True),
            "readout": readout,
            "scale": Range(None, None, expts=101),
        },
    )
