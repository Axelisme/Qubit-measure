"""Capture-owned native mux Form; the picker never selects participants."""

from zcu_tools.cfg_model import FormNode, Hidden, Slot, Template
from zcu_tools.gui.form_adapter import FixedElements, Form, FormContext

from zcu_lab.v2.support.measure.mux_forms import (
    READOUT_ELEMENTS,
    captured_mux_names,
    captured_mux_participants,
    mux_readout_initial,
    mux_readout_seed_metadata,
    mux_relax_delay,
    mux_tone_initials,
)

from .core import MuxGECfg


def make_form(ctx: FormContext) -> Form[MuxGECfg]:
    """Build the complete fixed name list from this Open/Reset capture.

    Readout identity stays required for every position. Only participant drive
    and sweep fields are required. Other tones start with None in those fields;
    all nullable fields remain editable and ordinary edits retain their values.
    Binding/source faults propagate; Reset alone recaptures. Shared controls
    apply to every branch. No native conversion or device access occurs.
    """
    names = captured_mux_names(ctx)
    participants = captured_mux_participants(ctx, names)
    tones: list[dict[str, FormNode]] = [
        {
            "name": name,
            "drive": Slot(f"{name}.module.x180")
            if ctx.has(f"{name}.module.x180")
            else Template("pi_pulse", bindings={"qubit": name}),
        }
        for name in names
    ]
    readout = mux_readout_initial(ctx)
    return Form(
        MuxGECfg,
        tree={
            "reset": Slot("$mux.module.reset"),
            "readout": readout,
            "tones": mux_tone_initials(tones, participants, fields=("drive",)),
            "participants": [index for index in participants],
            "shots": 100000,
            "relax_delay": mux_relax_delay(ctx, names),
            "initial_delay": Hidden(),
        },
        fixed_elements=(
            FixedElements(("tones",), "name", ("participants",)),
            READOUT_ELEMENTS,
        ),
        field_order=(
            "participants",
            "tones",
            "reset",
            "readout",
            "shots",
            "relax_delay",
            "initial_delay",
        ),
    ).with_seed_metadata(mux_readout_seed_metadata(readout))
