"""Qt-free wire-method contract table — the single source of truth for every
dispersive remote method's parameter schema, timeout and description.

Free of Qt and of any handler/Controller code so the lightweight ``mcp_server``
bridge can import it (to generate MCP tool schemas) without pulling in the
Qt-bound service layer. ``dispatch`` binds a synchronous handler to each spec here.

The agent is READ-ONLY: it observes a dispersive-fit-gui that the user drives, it
does not perform the analysis. Every method is a pure query — there is no load /
preprocess / tune / fit / export method (those need the human's eye on the preview
and the slider tuning). The agent reads the current state and reports it.
"""

from __future__ import annotations

from dataclasses import dataclass

from zcu_tools.gui.remote.param_spec import ParamSpec


@dataclass(frozen=True)
class MethodSpec:
    """Contract for one wire method, independent of its handler.

    ``timeout_seconds`` is the main-thread handler budget. ``params`` is the
    parameter contract (validation + MCP ``inputSchema``). ``tool_name`` overrides
    the derived ``dispersive_<method>`` MCP tool name when non-empty. The dispersive
    method set is entirely read-only, so no method takes parameters.
    """

    timeout_seconds: float
    description: str
    params: tuple[ParamSpec, ...] = ()
    tool_name: str = ""
    off_main_thread: bool = False


METHOD_SPECS: dict[str, MethodSpec] = {
    # Project
    "project.info": MethodSpec(
        5.0,
        "Read the current project info (chip_name, qub_name, result_dir, "
        "database_path).",
    ),
    # Fluxonium fit inputs (the fluxdep_fit handoff dispersive reads)
    "fit_inputs.info": MethodSpec(
        5.0,
        "Read the fluxonium fit inputs loaded from params.json: {has_inputs, "
        "params:{EJ,EC,EL} or null, flux_half, flux_int, flux_period, bare_rf_seed}.",
    ),
    # Preprocessing status
    "preprocess.status": MethodSpec(
        5.0,
        "Read the preprocessing status: {has_preprocess, n_flux, n_freq, edelay}.",
    ),
    # Dispersive result (the user's accepted g / bare_rf tuning) — read only
    "fit.result": MethodSpec(
        5.0,
        "Read the accepted dispersive result: {has_result, g, bare_rf, res_dim, step}.",
    ),
    # Resource version table (mcp/RPC bookkeeping; never surfaced to the agent).
    "resources.versions": MethodSpec(5.0, "Snapshot of all resource versions"),
    # State readiness (fan-out at MCP into one dispersive_state_check reply).
    "state.check": MethodSpec(
        5.0,
        "Read readiness flags at once: {has_project, has_fit_inputs, has_onetone, "
        "has_preprocess, has_result}.",
    ),
}
