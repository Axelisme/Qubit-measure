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
        "Read the opened entry, component and derived Result paths; null before opening.",
        readonly=True,
    ),
    "project.setup": MethodSpec(
        5.0,
        "Open an existing entry under project_root/result and select its fluxonium "
        "component. Omitted component uses the setup qubit alias.",
        params=(
            ParamSpec("project_root", JsonType.STRING),
            ParamSpec("entry", JsonType.STRING),
            ParamSpec("component", JsonType.STRING, required=False),
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
        "Load raw data. Native containers require explicit variable/device_axis; "
        "device selects the injected component instrument (unnamed_device without target). "
        "The basename replaces an existing entry.",
        params=(
            ParamSpec("filepath", JsonType.STRING),
            ParamSpec("spec_type", JsonType.STRING, enum=("OneTone", "TwoTone")),
            ParamSpec("inherit_from", JsonType.STRING, required=False),
            ParamSpec("variable", JsonType.STRING, required=False),
            ParamSpec("device_axis", JsonType.STRING, required=False),
            ParamSpec(
                "device", JsonType.STRING, required=False, default="unnamed_device"
            ),
            ParamSpec(
                "transpose_axes", JsonType.BOOLEAN, required=False, default=False
            ),
        ),
    ),
    "spectrum.load_processed": MethodSpec(
        30.0,
        "Restore sampled checkpoints, including empty completed spectra. "
        "Reject all names before publication if any already exist; Filter masks are not restored.",
        params=(ParamSpec("filepath", JsonType.STRING),),
    ),
    "spectrum.remove": MethodSpec(
        5.0,
        "Remove a literal spectrum.",
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
    "selection.render": MethodSpec(
        30.0,
        "Render the published joint cloud and per-spectrum Filter masks as PNG. "
        "Never opens a Session or reads a draft.",
        readonly=True,
    ),
    "interactive.read": MethodSpec(
        30.0,
        "Inspect the live context without opening input or changing display. "
        "Returns committed state, or null when inactive. target is required: "
        "spectrum requires a literal nonempty name; selection forbids name. "
        "Optional context_id validates the addressed draft lifetime.",
        params=(
            ParamSpec("target", JsonType.STRING, enum=("spectrum", "selection")),
            ParamSpec("name", JsonType.STRING, required=False),
            ParamSpec("context_id", JsonType.INTEGER, required=False),
        ),
        readonly=True,
    ),
    "interactive.render": MethodSpec(
        30.0,
        "Render a current context as PNG without editing or switching display. "
        "target is required: spectrum requires a literal nonempty name; "
        "selection forbids name. Missing drafts are rejected.",
        readonly=True,
        params=(
            ParamSpec("target", JsonType.STRING, enum=("spectrum", "selection")),
            ParamSpec("name", JsonType.STRING, required=False),
        ),
    ),
    "spectrum.interactive.open": MethodSpec(
        30.0,
        "Open/reuse a picker by literal spectrum name. "
        "kind is line, sample, onetone or twotone. sample uses the published "
        "spectrum type; native admission rejects unknown or unaligned sources. "
        "Returns owner identity, actual native kind and committed state.",
        params=(
            ParamSpec("name", JsonType.STRING),
            ParamSpec(
                "kind", JsonType.STRING, enum=("line", "sample", "onetone", "twotone")
            ),
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
        "Open/reuse the joint-cloud selection. Returns identity and input state.",
    ),
    "selection.interactive.command": MethodSpec(
        30.0,
        "Command the current joint identity. "
        "Use plugin commands or parameterless undo, finish, apply, cancel. "
        "Finish publishes and closes; Apply publishes without closing.",
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
        "or null, search_db, EJb, ECb, ELb, transitions, r_f, sample_f}.",
        readonly=True,
    ),
    "fit.render": MethodSpec(
        30.0,
        "Render the native Show overlay as PNG. Requires a fit result; "
        "does not change display or save an artifact.",
        readonly=True,
    ),
    "fit.database_info": MethodSpec(
        30.0,
        "Read search DB path, entry/level counts and inclusive EJ/EC/EL GHz and "
        "normalized flux ranges without loading energy values.",
        params=(ParamSpec("path", JsonType.STRING),),
        readonly=True,
    ),
    "fit.set_params": MethodSpec(
        5.0,
        "Replace all fit inputs; clear the old result "
        "without starting a search. search_db is the search database file. "
        "EJb, ECb and ELb are finite [lower, upper] bounds in GHz. transitions "
        "maps category names to integer pairs; reserved r_f/sample_f are finite "
        "numbers. Optional separate r_f/sample_f default to null.",
        params=(
            ParamSpec("search_db", JsonType.STRING),
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
        "Write predictors/<component>.json, update declared setup energy/alignment "
        "fields and record analysis in the opened entry. Uses the first aligned "
        "spectrum. Writes can partially succeed; no rollback or retry.",
    ),
    # Domain resource versions remain available as read-only bookkeeping.
    "resources.versions": MethodSpec(
        5.0, "Snapshot of all resource versions", readonly=True
    ),
    # State readiness (fan-out at MCP into one fluxdep_state_check reply).
}
