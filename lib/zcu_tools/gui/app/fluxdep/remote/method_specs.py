"""Qt-free wire-method contract table — the single source of truth for every
fluxdep remote method's parameter schema, timeout and description.

This module is intentionally free of Qt and of any handler/Controller code so
that the lightweight ``mcp_server`` bridge can import it (to generate MCP tool
schemas) without pulling in the Qt-bound service layer. ``dispatch`` binds a
synchronous handler to each spec here to form its runtime registry.

Read methods are explicitly readonly. Mutating methods use the existing
Controller owners; shared dispatch checks control authority before invoking
the handler.
"""

from __future__ import annotations

from zcu_tools.gui.remote.method_spec import MethodSpec
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
        readonly=True,
    ),
    "project.setup": MethodSpec(
        5.0,
        "Apply project identity and paths. Empty or "
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
        "List spectra: each {name, spec_type, aligned, points_completed, point_count, pending}. "
        "Completion includes zero points; point_count reports available data.",
        readonly=True,
    ),
    "spectrum.snapshot": MethodSpec(
        5.0,
        "Read the complete editable spectrum projection for a literal name. "
        "Absent names return {name, exists: false}; no raw complex matrix is sent.",
        params=(ParamSpec("name", JsonType.STRING),),
        readonly=True,
    ),
    "spectrum.render": MethodSpec(
        30.0,
        "Render published raw spectrum, committed alignment lines and sampled points "
        "as {png_b64, bytes}. Does not open input or change display.",
        params=(ParamSpec("name", JsonType.STRING),),
        readonly=True,
    ),
    "spectrum.load": MethodSpec(
        30.0,
        "Load native raw data. The basename replaces an existing entry.",
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
        "Restore processed spectra, including empty completed spectra; "
        "publication is incremental, not rolled back on failure.",
        params=(ParamSpec("filepath", JsonType.STRING),),
    ),
    "spectrum.remove": MethodSpec(
        5.0,
        "Remove a literal spectrum.",
        params=(ParamSpec("name", JsonType.STRING),),
    ),
    "spectrum.set_active": MethodSpec(
        5.0,
        "Select a spectrum for display. Null or "
        "omitted name clears display selection without changing source versions.",
        params=(ParamSpec("name", JsonType.STRING, required=False),),
    ),
    "spectrum.reset_alignment": MethodSpec(
        5.0,
        "Reopen alignment. Preserve "
        "native points, completion and the last calibration.",
        params=(ParamSpec("name", JsonType.STRING),),
    ),
    "spectrum.reset_points": MethodSpec(
        5.0,
        "Clear committed points and completion on an aligned spectrum. Preserve alignment.",
        params=(ParamSpec("name", JsonType.STRING),),
    ),
    # Cross-spectrum selection
    "selection.snapshot": MethodSpec(
        5.0,
        "Read the complete published joint-cloud selection mask and native "
        "normalized min_distance, plus pending for an unfinished Filter draft. "
        "Null mask means all points, not a live Session.",
        readonly=True,
    ),
    "selection.pointcloud": MethodSpec(
        5.0,
        "Derive the joint (flux, freq) point cloud assembled from every "
        "spectrum's selected points. Returns {fluxs:[...], freqs:[...]}.",
        readonly=True,
    ),
    "interactive.read": MethodSpec(
        30.0,
        "Inspect the live context without opening input or changing display. "
        "Returns state and native PNG, or null when inactive.",
        readonly=True,
    ),
    "spectrum.interactive.open": MethodSpec(
        30.0,
        "Open/reuse a picker for an already active literal spectrum. "
        "kind is line, onetone or twotone. Returns the "
        "owner identity, commands, state and native PNG.",
        params=(
            ParamSpec("name", JsonType.STRING),
            ParamSpec("kind", JsonType.STRING, enum=("line", "onetone", "twotone")),
        ),
    ),
    "spectrum.interactive.command": MethodSpec(
        30.0,
        "Command the current picker identity. "
        "Use plugin commands or parameterless undo, finish, cancel. params "
        "defaults to an empty object. Terminal receipt retains the old identity.",
        params=(
            ParamSpec("name", JsonType.STRING),
            ParamSpec("context_id", JsonType.INTEGER),
            ParamSpec("command", JsonType.STRING),
            ParamSpec("params", JsonType.OBJECT, required=False, default={}),
        ),
    ),
    "selection.interactive.open": MethodSpec(
        30.0,
        "Open/reuse the joint-cloud selection. "
        "Returns identity, commands, input state and native PNG.",
    ),
    "selection.interactive.command": MethodSpec(
        30.0,
        "Command the current joint identity. "
        "Use plugin commands or parameterless undo, apply, cancel. "
        "Apply publishes without closing.",
        params=(
            ParamSpec("context_id", JsonType.INTEGER),
            ParamSpec("command", JsonType.STRING),
            ParamSpec("params", JsonType.OBJECT, required=False, default={}),
        ),
    ),
    # Database-search fit (v2)
    "fit.result": MethodSpec(
        5.0,
        "Read the current fit inputs and result: {has_result, params:{EJ,EC,EL} "
        "or null, database_path, EJb, ECb, ELb, transitions, r_f, sample_f}.",
        readonly=True,
    ),
    "fit.set_params": MethodSpec(
        5.0,
        "Replace all fit inputs; clear the old result "
        "without starting a search. database_path is the search database file. "
        "EJb, ECb and ELb are finite [lower, upper] bounds in GHz. transitions "
        "maps category names to integer pairs; reserved r_f/sample_f are finite "
        "numbers. Optional separate r_f/sample_f default to null.",
        params=(
            ParamSpec("database_path", JsonType.STRING),
            ParamSpec("EJb", JsonType.ARRAY, array_item_type="number"),
            ParamSpec("ECb", JsonType.ARRAY, array_item_type="number"),
            ParamSpec("ELb", JsonType.ARRAY, array_item_type="number"),
            ParamSpec("transitions", JsonType.OBJECT),
            ParamSpec("r_f", JsonType.NUMBER, required=False),
            ParamSpec("sample_f", JsonType.NUMBER, required=False),
        ),
    ),
    "fit.search": MethodSpec(
        5.0,
        "Start the app's single-flight database search. Returns the "
        "app token; after completion explicitly reread fit.result.",
    ),
    "operation.status": MethodSpec(
        5.0,
        "Read a retained search token's pending or terminal activity. Null or "
        "omitted token reads the latest GUI/agent activity, or null before "
        "any search. Unknown tokens are rejected.",
        params=(ParamSpec("token", JsonType.INTEGER, required=False),),
        readonly=True,
    ),
    "operation.cancel": MethodSpec(
        5.0,
        "Request cooperative cancellation of a known search token. The receipt "
        "is not a terminal outcome; terminal tokens are a legal no-op.",
        params=(ParamSpec("token", JsonType.INTEGER),),
    ),
    "operation.await": MethodSpec(
        35.0,
        "Wait off-owner for a known search token, returning completed or timeout "
        "with native outcome and cancellation reasons. timeout is finite "
        "seconds from 0 to 30 (default 10). Timeout does not cancel.",
        params=(
            ParamSpec("token", JsonType.INTEGER),
            ParamSpec("timeout", JsonType.NUMBER, required=False, default=10.0),
        ),
        off_main_thread=True,
        readonly=True,
    ),
    "export.spectrums": MethodSpec(
        30.0,
        "Export the native spectrum collection. Optional filepath "
        "defaults to the project result directory. Existing files are rejected "
        "unless overwrite is true; I/O failure may leave partial output.",
        params=(
            ParamSpec("filepath", JsonType.STRING, required=False),
            ParamSpec("overwrite", JsonType.BOOLEAN, required=False, default=False),
        ),
    ),
    "fit.export_params": MethodSpec(
        30.0,
        "Export the current fit result. Optional savepath "
        "defaults to params.json in the project result directory. Merge with "
        "independent sections and use the first aligned spectrum's calibration.",
        params=(ParamSpec("savepath", JsonType.STRING, required=False),),
    ),
    # Domain resource versions remain available as read-only bookkeeping.
    "resources.versions": MethodSpec(
        5.0, "Snapshot of all resource versions", readonly=True
    ),
    # State readiness (fan-out at MCP into one fluxdep_state_check reply).
    "state.check": MethodSpec(
        5.0,
        "Read readiness flags at once: {has_project, spectrum_count, has_active}.",
        readonly=True,
    ),
}
