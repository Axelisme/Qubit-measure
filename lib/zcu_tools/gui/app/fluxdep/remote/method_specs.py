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
    "spectrum.snapshot": MethodSpec(
        5.0,
        "Read the complete editable spectrum projection for a literal name. "
        "Absent names return {name, exists: false}; no raw complex matrix is sent.",
        params=(ParamSpec("name", JsonType.STRING),),
    ),
    "spectrum.load": MethodSpec(
        30.0,
        "Load native raw data after reading project.info, spectrum.list and "
        "every current spectrum snapshot. The basename replaces an existing "
        "entry. New spectra still need a full snapshot before editing.",
        params=(
            ParamSpec("filepath", JsonType.STRING),
            ParamSpec("spec_type", JsonType.STRING, enum=("OneTone", "TwoTone")),
            ParamSpec("inherit_from", JsonType.STRING, required=False),
            ParamSpec(
                "transpose_axes", JsonType.BOOLEAN, required=False, default=False
            ),
        ),
    ),
    "spectrum.load_processed": MethodSpec(
        30.0,
        "Restore processed spectra after reading project.info, spectrum.list "
        "and every current spectrum snapshot. Includes empty completed spectra; "
        "publication is incremental, not rolled back on failure.",
        params=(ParamSpec("filepath", JsonType.STRING),),
    ),
    "spectrum.remove": MethodSpec(
        5.0,
        "Remove a literal spectrum after reading its snapshot and spectrum.list.",
        params=(ParamSpec("name", JsonType.STRING),),
    ),
    "spectrum.set_active": MethodSpec(
        5.0,
        "Select a spectrum for display after reading spectrum.list. Null or "
        "omitted name clears display selection without changing source versions.",
        params=(ParamSpec("name", JsonType.STRING, required=False),),
    ),
    "spectrum.reset_alignment": MethodSpec(
        5.0,
        "Reopen alignment after reading the spectrum snapshot. Preserve "
        "native points, completion and the last calibration.",
        params=(ParamSpec("name", JsonType.STRING),),
    ),
    "spectrum.reset_points": MethodSpec(
        5.0,
        "Clear committed points and completion on an aligned spectrum after "
        "reading its snapshot. Preserve alignment.",
        params=(ParamSpec("name", JsonType.STRING),),
    ),
    # Cross-spectrum selection
    "selection.snapshot": MethodSpec(
        5.0,
        "Read the complete published joint-cloud selection mask and native "
        "normalized min_distance. Null mask means all points, not a live Session.",
    ),
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
    "fit.set_params": MethodSpec(
        5.0,
        "Replace all fit inputs after reading fit.result; clear the old result "
        "without starting a search. database_path is the search database file. "
        "EJb, ECb and ELb are finite [lower, upper] bounds in GHz. transitions "
        "maps category names to integer pairs; reserved r_f/sample_f are finite "
        "numbers. Optional separate r_f/sample_f default to null.",
        params=(
            ParamSpec("database_path", JsonType.STRING),
            ParamSpec("EJb", JsonType.ARRAY),
            ParamSpec("ECb", JsonType.ARRAY),
            ParamSpec("ELb", JsonType.ARRAY),
            ParamSpec("transitions", JsonType.OBJECT),
            ParamSpec("r_f", JsonType.NUMBER, required=False),
            ParamSpec("sample_f", JsonType.NUMBER, required=False),
        ),
    ),
    "fit.search": MethodSpec(
        5.0,
        "Start the app's single-flight database search after reading project, "
        "fit, selection, collection and every spectrum snapshot. Returns the "
        "app token; after completion explicitly reread fit.result.",
    ),
    "operation.status": MethodSpec(
        5.0,
        "Read a retained search token's pending or terminal activity. Null or "
        "omitted token reads the latest GUI/agent activity, or null before "
        "any search. Unknown tokens are rejected; no observations are updated.",
        params=(ParamSpec("token", JsonType.INTEGER, required=False),),
    ),
    "operation.cancel": MethodSpec(
        5.0,
        "Request cooperative cancellation of a known search token. The receipt "
        "is not a terminal outcome; terminal tokens are a legal no-op.",
        params=(ParamSpec("token", JsonType.INTEGER),),
    ),
    "operation.await": MethodSpec(
        35.0,
        "Wait off-owner for a known search token, returning completed, timeout "
        "or user_feedback with native outcome/feedback. timeout is finite "
        "seconds from 0 to 30 (default 10). Timeout does not cancel; this read "
        "does not refresh fit or other observations.",
        params=(
            ParamSpec("token", JsonType.INTEGER),
            ParamSpec("timeout", JsonType.NUMBER, required=False, default=10.0),
        ),
        off_main_thread=True,
    ),
    "export.spectrums": MethodSpec(
        30.0,
        "Export the native spectrum collection after reading project.info, "
        "spectrum.list and every current spectrum snapshot. Optional filepath "
        "defaults to the project result directory. Existing files are rejected "
        "unless overwrite is true; I/O failure may leave partial output.",
        params=(
            ParamSpec("filepath", JsonType.STRING, required=False),
            ParamSpec("overwrite", JsonType.BOOLEAN, required=False, default=False),
        ),
    ),
    "fit.export_params": MethodSpec(
        30.0,
        "Export the current fit result after reading project.info, fit.result, "
        "spectrum.list and every current spectrum snapshot. Optional savepath "
        "defaults to params.json in the project result directory. Merge with "
        "independent sections and use the first aligned spectrum's calibration.",
        params=(ParamSpec("savepath", JsonType.STRING, required=False),),
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
    "spectrum.snapshot": ResourceObservationPolicy(reveals=("spectrum:{name}",)),
    "spectrum.load": ResourceObservationPolicy(
        guard_deps=("project", "spectrums:__set__", "spectrum:*"),
        refresh_after_write=True,
    ),
    "spectrum.load_processed": ResourceObservationPolicy(
        guard_deps=("project", "spectrums:__set__", "spectrum:*"),
        refresh_after_write=True,
    ),
    "spectrum.remove": ResourceObservationPolicy(
        guard_deps=("spectrums:__set__", "spectrum:{name}"), refresh_after_write=True
    ),
    "spectrum.set_active": ResourceObservationPolicy(guard_deps=("spectrums:__set__",)),
    "spectrum.reset_alignment": ResourceObservationPolicy(
        guard_deps=("spectrum:{name}",), refresh_after_write=True
    ),
    "spectrum.reset_points": ResourceObservationPolicy(
        guard_deps=("spectrum:{name}",), refresh_after_write=True
    ),
    "selection.snapshot": ResourceObservationPolicy(reveals=("selection",)),
    "selection.pointcloud": ResourceObservationPolicy(),
    "fit.result": ResourceObservationPolicy(reveals=("fit",)),
    "fit.set_params": ResourceObservationPolicy(
        guard_deps=("fit",), refresh_after_write=True
    ),
    "fit.search": ResourceObservationPolicy(
        guard_deps=("project", "fit", "selection", "spectrums:__set__", "spectrum:*")
    ),
    "operation.status": ResourceObservationPolicy(),
    "operation.cancel": ResourceObservationPolicy(),
    "operation.await": ResourceObservationPolicy(),
    "export.spectrums": ResourceObservationPolicy(
        guard_deps=("project", "spectrums:__set__", "spectrum:*")
    ),
    "fit.export_params": ResourceObservationPolicy(
        guard_deps=("project", "fit", "spectrums:__set__", "spectrum:*")
    ),
    "resources.versions": ResourceObservationPolicy(),
    "state.check": ResourceObservationPolicy(),
}
