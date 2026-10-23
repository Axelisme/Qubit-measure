"""Capture-owned native mux Form; the picker never selects participants."""

from dataclasses import replace

from zcu_tools.cfg_model import FormNode, Hidden, Slot
from zcu_tools.gui.form_adapter import FixedElements, Form, FormContext

from zcu_lab.v2._support.measure.mux_forms import (
    READOUT_ELEMENTS,
    captured_mux_names,
    mux_readout_initial,
    mux_readout_seed_metadata,
    mux_relax_delay,
)
from zcu_lab.v2._support.measure.rabi_initial_values import len_rabi_initial_values

from .core import MuxLenRabiCfg


def make_form(ctx: FormContext) -> Form[MuxLenRabiCfg]:
    """Build the complete fixed name list from this Open/Reset capture.

    All positions share the native element declaration and remain required,
    including nonparticipants. Missing pulse/calibration leaves stay invalid.
    Binding/source faults propagate; Reset alone recaptures. Shared controls
    apply to every branch. No native conversion or device access occurs.
    """
    names = captured_mux_names(ctx)
    tones: list[FormNode] = []
    for name in names:
        drive, stop = len_rabi_initial_values(ctx.parameters, qubit=name)
        # Core requires const/flat_top and replaces each total duration.
        drive = replace(drive, set={**drive.set, "waveform.length": Hidden(0.0)})
        tones.append(
            {
                "name": name,
                "drive": drive,
                "sweep": {"start": 0.03, "stop": stop},
            }
        )
    readout = mux_readout_initial(ctx)
    return Form(
        MuxLenRabiCfg,
        tree={
            "modules": {
                "reset": Slot("$mux.module.reset"),
                "readout": readout,
            },
            "tones": tones,
            "expts": 101,
            "participants": list(names),
            "relax_delay": mux_relax_delay(ctx, names),
            "initial_delay": Hidden(),
            "reps": 1000,
            "rounds": 100,
        },
        fixed_elements=(
            FixedElements(("tones",), "name", ("participants",)),
            READOUT_ELEMENTS,
        ),
        field_order=(
            "participants",
            "tones",
            "expts",
            "modules",
            "relax_delay",
            "reps",
            "rounds",
        ),
    ).with_seed_metadata(mux_readout_seed_metadata(readout))
