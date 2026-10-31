"""Fixed-capture duration scan with explicit two-line drive and GE calibration inputs."""

from typing import TypeVar

from zcu_tools.cfg_model import Expr, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.support.measure.mux_forms import mux_readout_seed_metadata

from .._measure import captured_roles
from .core import ConstructivePhaseCfg

T_Cfg = TypeVar("T_Cfg", bound=ConstructivePhaseCfg)


def scan_form(ctx: FormContext, cfg_type: type[T_Cfg]) -> Form[T_Cfg]:
    """Capture roles and pulses; scan endpoints require operator input.

    Existing CZ slots are editable source pairs, not certified gate settings.
    Missing slots offer explicit cosine templates on captured role wiring.
    Missing or linear role-tone classifiers, or missing radii, stay Invalid. Run GE Post first. Open/Reset never performs acquisition.
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
        cfg_type,
        field_order=(
            "roles",
            "reset",
            "x180_f1",
            "x180_f2",
            "readout",
            "drive",
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
            "reset": None,
            "x180_f1": Slot("$f1.module.x180", copy=True),
            "x180_f2": Slot("$f2.module.x180", copy=True),
            "readout": readout,
            "drive": drive,
            "relative_phase": Range(None, None, expts=25),
            "duration": Range(None, None, expts=101),
        },
    ).with_seed_metadata(mux_readout_seed_metadata(readout))


def make_form(ctx: FormContext) -> Form[ConstructivePhaseCfg]:
    """Capture roles and seed ordinary scan inputs; optional reset starts None."""
    return scan_form(ctx, ConstructivePhaseCfg)
