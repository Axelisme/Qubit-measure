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

from .core import MuxT2EchoCfg


def make_form(ctx: FormContext) -> Form[MuxT2EchoCfg]:
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
            "drive": Slot(f"{name}.module.x90")
            if ctx.has(f"{name}.module.x90")
            else Template("pi2_pulse", bindings={"qubit": name}),
            "pi_pulse": Slot(f"{name}.module.x180")
            if ctx.has(f"{name}.module.x180")
            else Template("pi_pulse", bindings={"qubit": name}),
            "detune": 0.0,
        }
        for name in names
    ]
    readout = mux_readout_initial(ctx)
    return Form(
        MuxT2EchoCfg,
        tree={
            "modules": {
                "reset": Slot("$mux.module.reset"),
                "readout": readout,
            },
            "tones": tones,
            "participants": list(names),
            "delay": Range.between(0.0, 30.0, expts=101),
            "relax_delay": mux_relax_delay(ctx, names),
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
            "modules",
            "delay",
            "reps",
            "rounds",
            "relax_delay",
            "initial_delay",
        ),
    ).with_seed_metadata(mux_readout_seed_metadata(readout))
