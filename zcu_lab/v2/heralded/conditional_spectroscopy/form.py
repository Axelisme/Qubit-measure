"""Fixed-capture thermal spectroscopy Form with explicit two-line probe inputs."""

from dataclasses import replace

from zcu_tools.cfg_model import Expr, Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.support.measure.ctx_helpers import source
from zcu_lab.v2.support.measure.mux_forms import (
    mux_readout_initial,
)
from zcu_lab.v2.support.measure.paired_qubits import TwoQubitForm, captured_qubits

from .core import HeraldedSpectroscopyCfg


def make_form(ctx: FormContext) -> Form[HeraldedSpectroscopyCfg]:
    """Seed the spectroscopy form from GUI aliases $qubit1, $qubit2 and $mux.

    The fixed capture seeds hidden tone indices; no $coupler binding is needed.
    Notebook bindings of the same names are separate. Missing or inconsistent
    qubit/mux bindings raise ValueError. Available coupler CZ slots start as
    references, not assumed weak probes. Without a slot, offer a cosine
    on the qubit's captured channel; gain, duration and Nyquist zone remain
    required. Run supplies both drive frequencies from the common scan.
    No model truth or safe-drive defaults are supplied. The frequency scan must
    be entered. Optional DC starts disabled.
    """
    qubits = captured_qubits(ctx)
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
        HeraldedSpectroscopyCfg,
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
            "frequency": Range(None, None, expts=101),
            "dc_device": None,
            "dc": Range(0, 0, expts=1),
        },
    )
