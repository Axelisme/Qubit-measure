"""Capture-owned native mux Form; the picker never selects participants."""

from zcu_tools.cfg_model import FormNode, Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import FixedElements, Form, FormContext

from zcu_lab.v2._support.measure.mux_forms import (
    READOUT_ELEMENTS,
    captured_mux_names,
    mux_readout_initial,
    mux_readout_seed_metadata,
    mux_relax_delay,
)

from .core import MuxT1Cfg


def make_form(ctx: FormContext) -> Form[MuxT1Cfg]:
    """Build the complete fixed name list from this Open/Reset capture.

    All positions share the native element declaration and remain required,
    including nonparticipants. Missing pulse/calibration leaves stay invalid.
    Binding/source faults propagate; Reset alone recaptures. Shared controls
    apply to every branch. No native conversion or device access occurs.
    """
    names = captured_mux_names(ctx)
    tones: list[FormNode] = [
        {
            "name": name,
            "drive": Slot(f"{name}.module.x180")
            if ctx.has(f"{name}.module.x180")
            else Template("pi_pulse", bindings={"qubit": name}),
        }
        for name in names
    ]
    recovery = mux_relax_delay(ctx, names)
    readout = mux_readout_initial(ctx)
    return Form(
        MuxT1Cfg,
        tree={
            "modules": {
                "reset": Slot("$mux.module.reset"),
                "readout": readout,
            },
            "tones": tones,
            "participants": list(names),
            "delays": Range.between(0.0, recovery, expts=101),
            "relax_delay": recovery,
            "initial_delay": Hidden(),
            "reps": 1000,
            "rounds": 10,
        },
        fixed_elements=(
            FixedElements(("tones",), "name", ("participants",)),
            READOUT_ELEMENTS,
        ),
        field_order=(
            "participants",
            "tones",
            "delays",
            "modules",
            "relax_delay",
            "reps",
            "rounds",
        ),
    ).with_seed_metadata(mux_readout_seed_metadata(readout))
