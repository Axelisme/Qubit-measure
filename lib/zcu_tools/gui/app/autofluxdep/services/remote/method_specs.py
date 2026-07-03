"""Qt-free wire-method contract table — the single source of truth for every
autofluxdep remote method's parameter schema, timeout and description.

This module is intentionally free of Qt and of any handler/Controller code so
that the lightweight ``mcp_server`` bridge can import it (to generate MCP tool
schemas) without pulling in the Qt-bound service layer. ``dispatch`` binds a
synchronous handler to each spec here to form its runtime registry.

The autofluxdep method set is entirely READ-ONLY: the agent observes a workflow
the user drives in the GUI, so there is no setup / edit-node / set-flux / run /
stop method. The only parametrised method is ``node.cfg`` (it needs the placed
node's ``name``); every other method is a no-arg pure query.
"""

from __future__ import annotations

from zcu_tools.gui.remote.method_spec import MethodSpec
from zcu_tools.gui.remote.param_spec import JsonType, ParamSpec

# ---------------------------------------------------------------------------
# The contract table. Keys are dotted wire-method names.
# ---------------------------------------------------------------------------


# The agent is READ-ONLY: it observes an autofluxdep-gui that the user drives, it
# does not assemble the workflow nor run the sweep. Every method here is a pure
# query — there is no add-node / set-flux / run / stop method. Those are user
# actions in the GUI (building the node graph and judging the live fits need the
# user's eye on the plot). The agent's job is to read the current state and report
# it.
METHOD_SPECS: dict[str, MethodSpec] = {
    # Project
    "project.info": MethodSpec(
        5.0,
        "Read the current project info (chip_name, qub_name, result_dir, "
        "database_path, params_path); fields are null when no real project is set.",
    ),
    # Workflow definition (the ordered node placements)
    "workflow.list": MethodSpec(
        5.0,
        "List the placed workflow nodes in order: each {name, type, enabled, "
        "provides, provides_modules, requires, has_result}. Excludes the "
        "predictor service (prepended only while a run is in progress, never a "
        "list row). Disabled nodes remain listed but are omitted from future "
        "runs until re-enabled.",
    ),
    # One placed node's user knobs
    "node.cfg": MethodSpec(
        5.0,
        "Read one placed node's user knobs by name: {name, type, knobs:{...}} — "
        "the un-lowered values the user set (a scalar reads to its value, a sweep "
        "to {start, stop, expts}). Errors if no node has that name.",
        params=(
            ParamSpec(
                "name",
                JsonType.STRING,
                required=True,
                description="The placed node's instance name (from workflow.list).",
            ),
        ),
    ),
    # Per-node run results (progress summary, never the raw 2D arrays)
    "result.summary": MethodSpec(
        5.0,
        "Summarise each node-with-a-result: {name, kind, n_flux, n_measured, "
        "fit_summary} — how far the sweep has progressed and a tiny fit summary, "
        "NOT the raw 2D signal data.",
    ),
    # Resource version table (optimistic-concurrency guard baseline). Full
    # snapshot the mcp layer reads to track last-seen versions; the version
    # integers are mcp/RPC bookkeeping and are never surfaced to the agent.
    "resources.versions": MethodSpec(5.0, "Snapshot of all resource versions"),
    # State readiness (fan-out at MCP into one state_check reply).
    "state.check": MethodSpec(
        5.0,
        "Read readiness flags at once: {has_project, has_soc, node_count, "
        "flux_count, has_flux_device, is_running, has_results, "
        "has_loaded_predictor, has_run_predictor}.",
    ),
}
