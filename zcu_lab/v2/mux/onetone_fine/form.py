"""Mux readout defaults from the fixed Open/Reset capture."""

from zcu_tools.cfg_model import Hidden, Range
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2._support.measure.mux_bindings import validate_mux_bindings
from zcu_lab.v2._support.measure.mux_forms import (
    READOUT_ELEMENTS,
    mux_readout_initial,
    mux_readout_seed_metadata,
)
from zcu_lab.v2.mux.onetone_fine.core import MuxOnetoneCfg


def make_form(ctx: FormContext) -> Form[MuxOnetoneCfg]:
    """Prepare a common-offset scan from the mux group's readout in ctx.

    Generate one editable readout slot per group resonator. Preserve captured
    readout scalar settings when present, otherwise use the blank template.
    Start with all tones selected and 101 offsets from -1 to 1 MHz. If the group
    names qubits, invalid pairings raise at Open/Reset. Groups without qubits
    still support resonator bring-up. With a slot, add five ringup times from
    the smallest set group kappa to its trigger and extend pulse coverage,
    preserving its ADC window; all kappas unset adds no wait. Without a slot,
    always seed the ADC window at 1.0 us, independent of kappa. Require all
    kappas for mixer, trigger and pulse length candidates. Use the extreme-
    frequency midpoint as mixer and the largest group resonator.ext.time_of_fly
    plus five ringup times as trigger. Pulse length is trigger plus ADC window.
    Missing flight times leave trigger and pulse length blank. Missing any kappa
    leaves mixer, trigger and pulse length blank. Invalid set sources reject
    Open/Reset. Candidates remain editable and require device confirmation at Run.
    """
    # Check declared qubit/resonator pairs before choosing readout defaults.
    validate_mux_bindings(ctx.parameters)

    # The group owns readout slot count; individual values remain editable.
    readout = mux_readout_initial(ctx, wait_for_ringup=True)
    return Form(
        MuxOnetoneCfg,
        tree={
            "modules": {
                "reset": Hidden(),
                "readout": readout,
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
    ).with_seed_metadata(mux_readout_seed_metadata(readout))
