"""Capture-owned native mux Form; the picker never selects participants."""

from dataclasses import replace

from zcu_tools.cfg_model import FormNode, Hidden, Slot
from zcu_tools.gui.form_adapter import FixedElements, Form, FormContext

from zcu_lab.v2._support.measure.mux_forms import (
    READOUT_ELEMENTS,
    captured_mux_names,
    captured_mux_participants,
    mux_readout_initial,
    mux_readout_seed_metadata,
    mux_relax_delay,
    mux_tone_initials,
)
from zcu_lab.v2._support.measure.rabi_initial_values import amp_rabi_initial_pulse

from .core import MuxAmpRabiCfg


def make_form(ctx: FormContext) -> Form[MuxAmpRabiCfg]:
    """Build the complete fixed name list from this Open/Reset capture.

    Readout identity stays required for every position. Only participant drive
    and sweep fields are required. Other tones start with None in those fields;
    all nullable fields remain editable and ordinary edits retain their values.
    Binding/source faults propagate; Reset alone recaptures. Shared controls
    apply to every branch. No native conversion or device access occurs.
    """
    names = captured_mux_names(ctx)
    participants = captured_mux_participants(ctx, names)
    tones: list[dict[str, FormNode]] = []
    for name in names:
        drive = amp_rabi_initial_pulse(ctx.parameters, qubit=name)
        # Run supplies each gain; retain independent shape admission.
        drive = replace(drive, set={**drive.set, "gain": Hidden(0.0)})
        tones.append(
            {
                "name": name,
                "drive": drive,
                "sweep": {"start": -0.3, "stop": 1.0},
            }
        )
    readout = mux_readout_initial(ctx)
    return Form(
        MuxAmpRabiCfg,
        tree={
            "modules": {
                "reset": Slot("$mux.module.reset"),
                "readout": readout,
            },
            "tones": mux_tone_initials(
                tones,
                participants,
                fields=(
                    "drive",
                    "sweep",
                ),
            ),
            "expts": 51,
            "participants": [index for index in participants],
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
            "expts",
            "reps",
            "rounds",
            "relax_delay",
            "initial_delay",
        ),
    ).with_seed_metadata(mux_readout_seed_metadata(readout))
