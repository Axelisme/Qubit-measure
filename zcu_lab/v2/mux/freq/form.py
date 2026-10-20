"""Frequency sweep inputs for every qubit in the selected mux group."""

from zcu_tools.cfg_model import FormNode, Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import FixedElements, Form, FormContext

from zcu_lab.v2._support.measure.mux_forms import captured_mux_names, mux_relax_delay

from .core import MuxFreqCfg


def make_form(ctx: FormContext) -> Form[MuxFreqCfg]:
    """Build per-qubit drive sweeps and shared averaging/readout controls.

    ctx supplies the mux group and calibration values at Open/Reset. Every qubit
    needs valid settings, including those omitted from participants. Missing
    calibration leaves required inputs blank; invalid group mappings raise.
    The tone picker chooses which qubit to edit, not which qubits to measure.
    """
    # Start each sweep around its qubit calibration, keeping all group positions.
    names = captured_mux_names(ctx)
    tones: list[FormNode] = [
        {
            "name": name,
            "drive": Template(
                "qub_probe",
                bindings={"qubit": name},
                # Run supplies this independent frequency scan.
                set={"freq": Hidden(0.0), "gain": 0.1, "waveform.length": 1.0},
            ),
            "sweep": Range.around(f"{name}.freq", f"1.5 * {name}.kappa", expts=301),
        }
        for name in names
    ]

    # Readout and recovery are shared; participants controls measurement order.
    return Form(
        MuxFreqCfg,
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
        field_order=(
            "participants",
            "tones",
            "modules",
            "relax_delay",
            "reps",
            "rounds",
        ),
    )
