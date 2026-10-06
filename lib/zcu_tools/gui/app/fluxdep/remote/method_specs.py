"""Qt-free wire-method contract table — the single source of truth for every
fluxdep remote method's parameter schema, timeout and description.

This module is intentionally free of Qt and of any handler/Controller code so
that the lightweight ``mcp_server`` bridge can import it (to generate MCP tool
schemas) without pulling in the Qt-bound service layer. ``dispatch`` binds a
synchronous handler to each spec here to form its runtime registry.

Read methods establish only their declared resource observations. Mutating
methods use the existing Controller owners; shared dispatch enforces their
per-connection guards before invoking the handler.
"""

from __future__ import annotations

from zcu_tools.gui.remote.method_spec import MethodSpec
from zcu_tools.gui.remote.observation import ResourceObservationPolicy
from zcu_tools.gui.remote.param_spec import JsonType, ParamSpec

# ---------------------------------------------------------------------------
# The contract table. Keys are dotted wire-method names.
# ---------------------------------------------------------------------------


METHOD_SPECS: dict[str, MethodSpec] = {
    # Project
    "project.info": MethodSpec(
        5.0,
        "Read the current project info (chip_name, qub_name, result_dir, "
        "database_path).",
    ),
    "project.setup": MethodSpec(
        5.0,
        "Apply project identity and paths after reading project.info. Empty or "
        "omitted paths use the native project defaults; database_path is the raw "
        "data root, not the fit search database file.",
        params=(
            ParamSpec("chip_name", JsonType.STRING),
            ParamSpec("qub_name", JsonType.STRING),
            ParamSpec("result_dir", JsonType.STRING, required=False, default=""),
            ParamSpec("database_path", JsonType.STRING, required=False, default=""),
        ),
    ),
    # Spectrum collection
    "spectrum.list": MethodSpec(
        5.0,
        "List spectra: each {name, spec_type, aligned, points_completed, point_count}. "
        "Completion includes zero points; point_count reports available data.",
    ),
    # Cross-spectrum selection
    "selection.pointcloud": MethodSpec(
        5.0,
        "Derive the joint (flux, freq) point cloud assembled from every "
        "spectrum's selected points. Returns {fluxs:[...], freqs:[...]}.",
    ),
    # Database-search fit (v2)
    "fit.result": MethodSpec(
        5.0,
        "Read the current fit inputs and result: {has_result, params:{EJ,EC,EL} "
        "or null, database_path, EJb, ECb, ELb, transitions, r_f, sample_f}.",
    ),
    # Resource version table (optimistic-concurrency guard baseline). Full
    # snapshot is bookkeeping, not a full editable read. Only GUI shared
    # dispatch owns per-connection seen; MCP must not infer observations here.
    "resources.versions": MethodSpec(5.0, "Snapshot of all resource versions"),
    # State readiness (fan-out at MCP into one fluxdep_state_check reply).
    "state.check": MethodSpec(
        5.0,
        "Read readiness flags at once: {has_project, spectrum_count, has_active}.",
    ),
}

# App resource semantics only; shared RemoteControlServiceBase owns seen maps,
# guard comparison, full-read observations and successful self-write tracking.
OBSERVATION_POLICIES: dict[str, ResourceObservationPolicy] = {
    "project.info": ResourceObservationPolicy(reveals=("project",)),
    "project.setup": ResourceObservationPolicy(
        guard_deps=("project",), refresh_after_write=True
    ),
    "spectrum.list": ResourceObservationPolicy(reveals=("spectrums:__set__",)),
    "selection.pointcloud": ResourceObservationPolicy(),
    "fit.result": ResourceObservationPolicy(reveals=("fit",)),
    "resources.versions": ResourceObservationPolicy(),
    "state.check": ResourceObservationPolicy(),
}
