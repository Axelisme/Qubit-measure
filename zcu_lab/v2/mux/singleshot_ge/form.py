"""Capture-owned native mux Form; the picker never selects participants."""

from zcu_tools.cfg_model import FormNode, Hidden, Locked, Slot, Template
from zcu_tools.gui.form_adapter import FixedElements, Form, FormContext

from zcu_lab.v2._support.measure.mux_forms import captured_mux_names, mux_relax_delay

from .core import MuxGECfg


def make_form(ctx: FormContext) -> Form[MuxGECfg]:
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
            else Template("pi_pulse", qubit=name),
        }
        for name in names
    ]
    return Form(
        MuxGECfg,
        tree={
            "modules": {
                "reset": Slot("$mux.module.reset"),
                "readout": Slot("$mux.module.readout")
                if ctx.has("$mux.module.readout")
                else Template("readout/mux:blank"),
            },
            "tones": tones,
            "participants": list(names),
            "shots": 100000,
            "relax_delay": mux_relax_delay(ctx, names),
            "initial_delay": Hidden(),
            "reps": Locked(1),
            "rounds": Locked(1),
        },
        fixed_elements=(FixedElements(("tones",), "name", ("participants",)),),
    )
