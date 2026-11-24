"""Fixed-capture inputs for the two-line heralded chevron."""

from dataclasses import replace

from zcu_tools.cfg_model import Expr, Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.support.measure.ctx_helpers import source
from zcu_lab.v2.support.measure.mux_forms import mux_readout_initial
from zcu_lab.v2.support.measure.paired_qubits import TwoQubitForm, captured_qubits

from .core import HeraldedChevronCfg


def make_form(ctx: FormContext) -> Form[HeraldedChevronCfg]:
    """Seed the chevron from the captured qubit, mux and coupler aliases.

    Missing or inconsistent bindings raise ValueError. Ordered mux tone
    positions stay hidden. Reference the captured CZ slots when present;
    otherwise offer cosine envelopes on the captured qubit wiring. Supply
    missing calibrations and scan endpoints before Run.
    """
    qubits = captured_qubits(ctx, require_coupler=True)
    drive = {}
    for position in ("qubit1", "qubit2"):
        # CZ slots: $coupler.module.cz_qubit1 and $coupler.module.cz_qubit2.
        pulse = source(
            ctx,
            f"$coupler.module.cz_{position}",
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
        # Both drive carriers come from the scan, not from these Form inputs.
        drive[position] = replace(pulse, set={**pulse.set, "freq": Hidden(0.0)})
    return TwoQubitForm(
        HeraldedChevronCfg,
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
            "readout": mux_readout_initial(ctx),
            "drive": drive,
            "frequency": Range(None, None, expts=31),
            "duration": Range(None, None, expts=101),
        },
    )
