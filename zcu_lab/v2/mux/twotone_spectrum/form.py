"""Frequency sweep inputs for every qubit in the selected mux group."""

from zcu_tools.cfg_model import Expr, FieldMeta, FormNode, Hidden, Slot, Template
from zcu_tools.gui.form_adapter import FixedElements, Form, FormContext

from zcu_lab.v2._support.measure.mux_forms import (
    READOUT_ELEMENTS,
    captured_mux_names,
    mux_readout_initial,
    mux_readout_seed_metadata,
    mux_tone_initials,
)

from .core import MuxFreqCfg


def make_form(ctx: FormContext) -> Form[MuxFreqCfg]:
    """Build per-qubit drive sweeps and shared averaging/readout controls.

    ctx supplies the mux group and calibration values at Open/Reset. Every qubit
    needs valid readout identity. Selected drive/sweep fields are required;
    others start with None and stay editable. Missing
    calibration leaves required inputs blank; invalid group mappings raise.
    The tone picker chooses which qubit to edit, not which qubits to measure.
    """
    # Start each sweep around its qubit calibration, keeping all group positions.
    names = captured_mux_names(ctx)
    participants = list(range(len(names)))
    tones: list[dict[str, FormNode]] = [
        {
            "name": name,
            "drive": Template(
                "qub_probe",
                bindings={"qubit": name},
                # Run supplies this independent frequency scan.
                set={"freq": Hidden(0.0), "gain": 0.1, "waveform.length": 1.0},
            ),
            "sweep": {
                "start": Expr(f"{name}.freq - 1.5 * {name}.kappa"),
                "stop": Expr(f"{name}.freq + 1.5 * {name}.kappa"),
            },
        }
        for name in names
    ]

    # Readout and recovery are shared; participants controls measurement order.
    readout = mux_readout_initial(ctx)
    return Form(
        MuxFreqCfg,
        tree={
            "modules": {
                "reset": Slot("$mux.module.reset"),
                "readout": readout,
            },
            "tones": mux_tone_initials(tones, participants, fields=("drive", "sweep")),
            "expts": 301,
            "participants": [index for index in participants],
            "relax_delay": 1.0,
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
    ).with_seed_metadata(
        {
            **mux_readout_seed_metadata(readout),
            ("relax_delay",): FieldMeta(
                description="Fixed 1.0 us initial recovery delay"
            ),
        }
    )
