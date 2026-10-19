"""Fixed mux identities and shared recovery-time initial values for seven Forms."""

from zcu_tools.cfg_model import MuxReadoutSpec
from zcu_tools.gui.form_adapter import FormContext

from .ctx_helpers import t1_initial_time
from .mux_bindings import require_mux_qubits


def captured_mux_names(ctx: FormContext) -> tuple[str, ...]:
    """Require complete qubit bindings and, when present, matching mux readout.

    Read only the Open/Reset capture. A missing readout slot may be filled in
    the Form; an existing slot must be a MuxReadoutSpec with the full binding
    length. Wrong family/count, missing group/qubits or invalid links raise.
    No assets, native conversion, live files or devices are consulted.
    """
    names = require_mux_qubits(ctx.parameters)
    if ctx.has("$mux.module.readout"):
        parameters = ctx.parameters
        if parameters is None:
            raise ValueError("Mux qubit Forms require a parameter capture")
        readout = parameters.library.module_spec(
            parameters.library_key("$mux.module.readout")
        )
        if not isinstance(readout, MuxReadoutSpec):
            raise ValueError("$mux.module.readout must reference readout/mux")
        require_mux_qubits(parameters, len(readout.tones))
    return names


def mux_relax_delay(ctx: FormContext, names: tuple[str, ...]) -> float | None:
    """Return five times the largest captured T1, in us, for shared recovery.

    names is the complete validated capture-ordered qubit tuple. Each missing
    T1 uses the common 30 us initial scale; any stored invalid T1 leaves the
    required delay blank. Uses the same T1 policy as single-qubit Forms,
    without refreshing sources or writing calibration.
    """
    delays = [
        t1_initial_time(ctx.parameters, multiplier=5.0, qubit=name) for name in names
    ]
    if any(delay is None for delay in delays):
        return None
    return max(delay for delay in delays if delay is not None)
