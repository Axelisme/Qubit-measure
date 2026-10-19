"""Capture-owned native mux Form; the picker never selects participants."""

from zcu_tools.cfg_model import FormNode, Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import FixedElements, Form, FormContext

from zcu_lab.v2._support.measure.mux_forms import captured_mux_names, mux_relax_delay
from zcu_lab.v2._support.measure.rabi_initial_values import amp_rabi_initial_pulse

from .core import MuxAmpRabiCfg


def make_form(ctx: FormContext) -> Form[MuxAmpRabiCfg]:
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
            "drive": amp_rabi_initial_pulse(ctx.parameters, qubit=name),
            "sweep": Range.between(-0.3, 1.0, expts=51),
        }
        for name in names
    ]
    return Form(
        MuxAmpRabiCfg,
        tree={
            "modules": {
                "reset": Slot("$mux.module.reset"),
                "readout": Slot("$mux.module.readout")
                if ctx.has("$mux.module.readout")
                else Template("readout/mux:blank"),
            },
            "tones": tones,
            "participants": list(names),
            "relax_delay": mux_relax_delay(ctx, names),
            "initial_delay": Hidden(),
            "reps": 1000,
            "rounds": 100,
        },
        fixed_elements=(FixedElements(("tones",), "name", ("participants",)),),
    )
