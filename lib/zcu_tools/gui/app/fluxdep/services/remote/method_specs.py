"""Qt-free wire-method contract table — the single source of truth for every
fluxdep remote method's parameter schema, timeout and description.

This module is intentionally free of Qt and of any handler/Controller code so
that the lightweight ``mcp_server`` bridge can import it (to generate MCP tool
schemas) without pulling in the Qt-bound service layer. ``dispatch`` binds a
synchronous handler to each spec here to form its runtime registry.

The fluxdep method set is entirely read-only (see ``METHOD_SPECS`` below), so no
method here takes parameters — every ``MethodSpec.params`` keeps its empty default.
"""

from __future__ import annotations

from zcu_tools.gui.remote.method_spec import MethodSpec

# ---------------------------------------------------------------------------
# The contract table. Keys are dotted wire-method names.
# ---------------------------------------------------------------------------


# The agent is READ-ONLY: it observes a fluxdep-gui that the user drives, it does
# not perform the analysis. Every method here is a pure query — there is no
# load / align / point-pick / select / fit / export method. Those are user
# actions in the GUI (point-picking and axis judgement need the human's eye on the
# preview, which the agent does not have). The agent's job is to read the current
# state and report it.
METHOD_SPECS: dict[str, MethodSpec] = {
    # Project
    "project.info": MethodSpec(
        5.0,
        "Read the current project info (chip_name, qub_name, result_dir, "
        "database_path).",
    ),
    # Spectrum collection
    "spectrum.list": MethodSpec(
        5.0,
        "List the loaded spectra: each {name, spec_type, aligned, points_selected}.",
    ),
    # Cross-spectrum selection
    "selection.pointcloud": MethodSpec(
        5.0,
        "Derive the joint (flux, freq) point cloud assembled from every "
        "spectrum's selected points. Returns {fluxs:[...], freqs:[...]}.",
    ),
    # Database-search fit (v2) — read only
    "fit.result": MethodSpec(
        5.0,
        "Read the current fit inputs and result: {has_result, params:{EJ,EC,EL} "
        "or null, database_path, EJb, ECb, ELb, transitions, r_f, sample_f}.",
    ),
    # Resource version table (optimistic-concurrency guard baseline). Full
    # snapshot the mcp layer reads to track last-seen versions; the version
    # integers are mcp/RPC bookkeeping and are never surfaced to the agent.
    "resources.versions": MethodSpec(5.0, "Snapshot of all resource versions"),
    # State readiness (fan-out at MCP into one gui_state_check reply).
    "state.check": MethodSpec(
        5.0,
        "Read readiness flags at once: {has_project, spectrum_count, has_active}.",
    ),
}
