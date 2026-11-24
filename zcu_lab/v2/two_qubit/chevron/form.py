"""Fixed-capture duration scan with explicit two-line drive and GE calibration inputs."""

from dataclasses import replace

from zcu_tools.cfg_model import Expr, Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.support.measure.ctx_helpers import source
from zcu_lab.v2.support.measure.mux_forms import (
    mux_readout_initial,
)
from zcu_lab.v2.support.measure.paired_qubits import TwoQubitForm, captured_qubits

from .core import ChevronCfg


def make_form(ctx: FormContext) -> Form[ChevronCfg]:
    """Seed the chevron from GUI aliases $qubit1, $qubit2, $mux and $coupler.

    Missing or inconsistent bindings raise ValueError on Open/Reset. Notebook
    bindings of the same names are separate. Tone indices are hidden. Reference
    the coupler's cz_qubit1/cz_qubit2 slots when present, otherwise offer
    cosine templates on captured qubit wiring. Scan endpoints and missing
    calibration must be supplied before Run.
    """
    qubits = captured_qubits(ctx, require_coupler=True)
    drive = {}
    for position in ("qubit1", "qubit2"):
        # CZ slots: $coupler.module.cz_qubit1 and $coupler.module.cz_qubit2.
        slot = f"$coupler.module.cz_{position}"
        pulse = source(
            ctx,
            slot,
            Template(
                "pulse:blank",
                set={
                    "ch": Expr(f"${position}.wiring.ch"),
                    "nqz": None,
                    "gain": None,
                    "waveform.style": "cosine",
                    "waveform.length": None,
                },
            ),
        )
        # Hide the swept frequency in inline without changing a captured ref.
        drive[position] = replace(pulse, set={**pulse.set, "freq": Hidden(0.0)})
    readout = mux_readout_initial(ctx)
    return TwoQubitForm(
        ChevronCfg,
        readout_of=lambda cfg: cfg.readout,
        tree={
            "reps": 1000,
            "rounds": 10,
            "relax_delay": 1.0,
            "qubit1_tone": Hidden(qubits.qubit1_tone),
            "qubit2_tone": Hidden(qubits.qubit2_tone),
            "reset": None,
            "x180_qubit1": Slot("$qubit1.module.x180"),
            "x180_qubit2": Slot("$qubit2.module.x180"),
            "readout": readout,
            "drive": drive,
            "frequency": Range(None, None, expts=31),
            "duration": Range(None, None, expts=101),
        },
    )
