"""Mux readout defaults from the fixed Open/Reset capture."""

from zcu_tools.cfg_model import Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2._support.measure.mux_bindings import validate_mux_bindings
from zcu_lab.v2.mux.onetone.core import MuxOnetoneCfg


def make_form(ctx: FormContext) -> Form[MuxOnetoneCfg]:
    """Prepare a common-offset scan from the mux group's readout in ctx.

    Use the captured readout slot when present, otherwise an editable blank
    readout without tones. Start with all tones selected and 101 offsets from
    -1 to 1 MHz. If the group names qubits, invalid qubit/resonator pairings
    raise at Open/Reset. Groups without qubits allow manually entered readout.
    """
    # Check declared qubit/resonator pairs before choosing readout defaults.
    validate_mux_bindings(ctx.parameters)

    # Retain a group slot reference when available; otherwise leave tones to fill.
    return Form(
        MuxOnetoneCfg,
        tree={
            "modules": {
                "reset": Hidden(),
                "readout": Slot("$mux.module.readout")
                if ctx.has("$mux.module.readout")
                else Template("readout/mux:blank"),
            },
            "sweep": {"df": Range.between(-1.0, 1.0, expts=101)},
            "initial_delay": Hidden(),
        },
        field_order=(
            "active_tones",
            "sweep",
            "modules",
            "relax_delay",
            "reps",
            "rounds",
        ),
    )
