"""Fixed-capture inputs for conditional phase calibration."""

from typing import TypeVar

from zcu_tools.cfg_model import Hidden, Range, Slot, Template
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.support.measure.ctx_helpers import source
from zcu_lab.v2.support.measure.mux_forms import mux_readout_seed_metadata

from .._measure import TwoQubitForm, captured_qubits
from .core import ConditionalPhaseCfg

CfgT = TypeVar("CfgT", bound=ConditionalPhaseCfg)


def make_form(ctx: FormContext) -> Form[ConditionalPhaseCfg]:
    """Use fixed qubit and coupler bindings to seed the ordinary classified phase experiment."""
    return phase_form(ctx, ConditionalPhaseCfg)


def phase_form(ctx: FormContext, cfg_type: type[CfgT]) -> Form[CfgT]:
    """Seed calibrated pulses and hidden tone positions for either variant.

    Missing CZ slots stay required. Both qubit tones need circle + radius from
    GE Post before Run. No other experiment Form or simulator truth supplies inputs.
    reset starts as None and can be authored explicitly.
    """
    qubits = captured_qubits(ctx, require_coupler=True)
    readout = source(
        ctx,
        "$mux.module.readout",
        Template("readout/mux:blank"),
    )
    return TwoQubitForm(
        cfg_type,
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
            "x90_qubit1": Slot("$qubit1.module.x90"),
            "x90_qubit2": Slot("$qubit2.module.x90"),
            "readout": readout,
            "drive": {
                # Run overrides both CZ frequencies with the common hardware sweep.
                "qubit1": Slot("$coupler.module.cz_qubit1"),
                "qubit2": Slot("$coupler.module.cz_qubit2"),
            },
            "frequency": Range(None, None, expts=101),
        },
    ).with_seed_metadata(mux_readout_seed_metadata(readout))
