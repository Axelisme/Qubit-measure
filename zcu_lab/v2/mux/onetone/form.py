"""Mux readout defaults from the fixed Open/Reset capture."""

from zcu_tools.cfg_model import Hidden, Range
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2._support.measure.mux_bindings import (
    require_mux_group,
    validate_mux_bindings,
)
from zcu_lab.v2._support.measure.mux_forms import (
    READOUT_ELEMENTS,
    captured_mux_participants,
    mux_readout_initial,
    mux_readout_seed_metadata,
)
from zcu_lab.v2.mux.onetone.core import MuxOnetoneCfg


def make_form(ctx: FormContext) -> Form[MuxOnetoneCfg]:
    """Prepare a common-offset scan from the mux group's readout in ctx.

    Generate one editable readout slot per group resonator. Preserve captured
    readout scalar settings when present, otherwise use the blank template.
    Capture $participants in resonator order and 101 offsets from -1 to 1 MHz.
    Missing/unbound aliases select all original indices. If the group
    names qubits, invalid pairings raise at Open/Reset. Groups without qubits
    still support resonator bring-up. With a slot, add five ringup times from
    the smallest set group kappa to its trigger and extend pulse coverage,
    preserving its ADC window; all kappas unset adds no wait. Without a slot,
    always seed the ADC window at 1.0 us, independent of kappa. Mixer reads
    $mux.mixer_freq independently of kappa and tone frequencies; null stays
    required and blank. Trigger uses $mux.time_of_fly plus five ringup times
    from the smallest group kappa. Pulse length is trigger plus ADC window.
    Missing group flight time or any kappa leaves trigger and pulse length
    required and blank. Invalid set sources reject
    Open/Reset. Candidates remain editable and require device confirmation at Run.
    """
    # Check declared qubit/resonator pairs before choosing readout defaults.
    validate_mux_bindings(ctx.parameters)

    # The group owns readout slot count; individual values remain editable.
    participants = captured_mux_participants(
        ctx, require_mux_group(ctx.parameters).resonators
    )
    readout = mux_readout_initial(ctx, wait_for_ringup=True)
    return Form(
        MuxOnetoneCfg,
        tree={
            "participants": [index for index in participants],
            "reps": 1000,
            "rounds": 10,
            "reset": Hidden(),
            "readout": readout,
            "df": Range.between(-1.0, 1.0, expts=101),
            "initial_delay": Hidden(),
        },
        fixed_elements=(READOUT_ELEMENTS,),
        field_order=(
            "participants",
            "reset",
            "readout",
            "df",
            "reps",
            "rounds",
            "relax_delay",
            "initial_delay",
        ),
    ).with_seed_metadata(mux_readout_seed_metadata(readout))
