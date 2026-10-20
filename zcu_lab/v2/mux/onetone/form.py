"""Mux readout defaults from the fixed Open/Reset capture."""

from zcu_tools.cfg_model import Hidden, Range
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2._support.measure.mux_bindings import validate_mux_bindings
from zcu_lab.v2._support.measure.mux_forms import READOUT_ELEMENTS, mux_readout_initial
from zcu_lab.v2.mux.onetone.core import MuxOnetoneCfg


def make_form(ctx: FormContext) -> Form[MuxOnetoneCfg]:
    """Prepare a common-offset scan from the mux group's readout in ctx.

    Generate one editable readout slot per group resonator. Preserve captured
    readout scalar settings when present, otherwise use the blank template.
    Start with all tones selected and 101 offsets from -1 to 1 MHz. If the group
    names qubits, invalid pairings raise at Open/Reset. Groups without qubits
    still support resonator bring-up. Add five ringup times using the smallest
    group kappa to the original scalar trigger offset and extend the pulse to
    cover the unchanged ADC window. Invalid member kappa rejects Open/Reset.
    """
    # Check declared qubit/resonator pairs before choosing readout defaults.
    validate_mux_bindings(ctx.parameters)

    # The group owns readout slot count; individual values remain editable.
    return Form(
        MuxOnetoneCfg,
        tree={
            "modules": {
                "reset": Hidden(),
                "readout": mux_readout_initial(ctx, wait_for_ringup=True),
            },
            "sweep": {"df": Range.between(-1.0, 1.0, expts=101)},
            "initial_delay": Hidden(),
        },
        fixed_elements=(READOUT_ELEMENTS,),
        field_order=(
            "active_tones",
            "sweep",
            "modules",
            "relax_delay",
            "reps",
            "rounds",
        ),
    )
