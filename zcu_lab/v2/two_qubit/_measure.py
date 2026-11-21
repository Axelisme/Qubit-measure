"""Fixed capture binding for two-qubit Forms, never used by core."""

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from zcu_tools.cfg_model import ConfigBase, FormNode, MuxReadoutCfg, MuxReadoutSpec
from zcu_tools.gui.form import FormEnv, FormErrors, FormModel
from zcu_tools.gui.form_adapter import Form, FormContext

from zcu_lab.v2.support.measure.mux_bindings import (
    require_mux_group,
    require_mux_qubits,
)


class TwoQubitForm[C: ConfigBase](Form[C]):
    """Lower a two-qubit draft against the complete captured mux group."""

    def __init__(
        self,
        core_cfg: type[C],
        *,
        tree: Mapping[str, FormNode],
        readout_of: Callable[[C], MuxReadoutCfg],
    ) -> None:
        """Use the usual Form tree and a typed accessor for the cfg's terminal readout."""
        super().__init__(core_cfg, tree)
        self._readout_of = readout_of

    def to_cfg(self, value: FormModel, env: FormEnv) -> C | FormErrors:
        """Lower a result in its fixed Form environment, without modifying its draft.

        Preserve ordinary lowering errors. If the validated terminal readout
        is not the size of the captured group, return an error at readout.
        """
        cfg = super().to_cfg(value, env)
        if isinstance(cfg, FormErrors):
            return cfg
        # Open/Reset (or Refresh) has already checked the same group's ordered links.
        expected = len(require_mux_group(env.parameters).resonators)
        actual = len(self._readout_of(cfg).tones)
        if actual != expected:
            return FormErrors(
                {("readout",): f"$mux group: expected {expected} tones, got {actual}"}
            )
        return cfg


@dataclass(frozen=True)
class CapturedQubits:
    """Component names and zero-based positions in the full mux readout.

    qubit1, qubit2 and mux are the captured component names. coupler is the
    captured component name only when the caller requires it, otherwise None.
    qubit1_tone and qubit2_tone follow $mux.qubits, not ADC numbers or labels.
    """

    qubit1: str
    qubit2: str
    mux: str
    coupler: str | None
    qubit1_tone: int
    qubit2_tone: int


def captured_qubits(
    ctx: FormContext, *, require_coupler: bool = False
) -> CapturedQubits:
    """Resolve positions and component names from a fixed Open/Reset capture.

    Require scalar $qubit1, $qubit2 and $mux aliases; only resolve $coupler
    when require_coupler is true. Both distinct qubits must appear in the
    ordered $mux.qubits list. Require a group/mux with complete, ordered
    qubit/resonator links (ADR-0072), without requiring a qubit or coupler
    subtype. If $mux.module.readout is bound, require a mux readout with one
    tone per listed qubit. Missing or inconsistent bindings raise ValueError.
    The capture is neither refreshed nor changed.
    """
    parameters = ctx.parameters
    if parameters is None:
        raise ValueError("Bind $qubit1, $qubit2 and $mux in a fixed parameter capture")
    qubit1 = parameters.resolve_path("$qubit1")
    qubit2 = parameters.resolve_path("$qubit2")
    mux = parameters.resolve_path("$mux")
    coupler = parameters.resolve_path("$coupler") if require_coupler else None

    if qubit1 == qubit2:
        raise ValueError(
            f"$qubit1 and $qubit2 both bind {qubit1!r}; use distinct components"
        )
    qubits = require_mux_qubits(parameters)
    for alias, name in (("$qubit1", qubit1), ("$qubit2", qubit2)):
        if name not in qubits:
            raise ValueError(f"$mux.qubits does not contain {alias}={name!r}")

    slot = "$mux.module.readout"
    if ctx.has(slot):
        readout = parameters.library.module_spec(parameters.library_key(slot))
        if not isinstance(readout, MuxReadoutSpec):
            raise ValueError(f"{slot} must provide a mux readout")
        if len(readout.tones) != len(qubits):
            raise ValueError(
                f"{slot} has {len(readout.tones)} tones; $mux.qubits has {len(qubits)} names"
            )

    return CapturedQubits(
        qubit1=qubit1,
        qubit2=qubit2,
        mux=mux,
        coupler=coupler,
        qubit1_tone=qubits.index(qubit1),
        qubit2_tone=qubits.index(qubit2),
    )
