"""Qt-free wire-method contract table — the single source of truth for every
fluxdep remote method's parameter schema, timeout and description.

This module is intentionally free of Qt and of any handler/Controller code so
that the lightweight ``mcp_server`` bridge can import it (to generate MCP tool
schemas) without pulling in the Qt-bound service layer. ``dispatch`` binds a
synchronous handler to each spec here to form its runtime registry.
"""

from __future__ import annotations

from dataclasses import dataclass

from .param_spec import JsonType, ParamSpec


@dataclass(frozen=True)
class MethodSpec:
    """Contract for one wire method, independent of its handler.

    ``timeout_seconds`` is the main-thread handler budget. ``params`` is the
    parameter contract used both for runtime validation (dispatch/service) and
    MCP ``inputSchema`` generation (mcp_server). ``tool_name`` overrides the
    derived ``fluxdep_<method>`` MCP tool name when non-empty.

    ``off_main_thread`` marks a blocking handler that must NOT be marshalled
    onto the Qt main thread. The fluxdep method set has no such handler (every
    action is a fast main-thread state mutation), but the flag is kept for
    mechanism parity with the dispatcher.
    """

    timeout_seconds: float
    description: str
    params: tuple[ParamSpec, ...] = ()
    tool_name: str = ""
    off_main_thread: bool = False


# ---------------------------------------------------------------------------
# ParamSpec factory shorthands — keep the table readable.
# ---------------------------------------------------------------------------


def _str(name: str, desc: str = "") -> ParamSpec:
    return ParamSpec(name, JsonType.STRING, required=True, description=desc)


def _str_opt(name: str, desc: str = "") -> ParamSpec:
    return ParamSpec(name, JsonType.STRING, required=False, description=desc)


def _num(name: str, desc: str = "") -> ParamSpec:
    return ParamSpec(name, JsonType.NUMBER, required=True, description=desc)


def _json(name: str, desc: str = "") -> ParamSpec:
    return ParamSpec(name, JsonType.JSON, required=True, description=desc)


def _bool_opt(name: str, desc: str = "") -> ParamSpec:
    return ParamSpec(
        name, JsonType.BOOLEAN, required=False, default=False, description=desc
    )


# ---------------------------------------------------------------------------
# The contract table. Keys are dotted wire-method names.
# ---------------------------------------------------------------------------


METHOD_SPECS: dict[str, MethodSpec] = {
    # Project
    "project.setup": MethodSpec(
        10.0,
        "Set the analysis project: chip / qubit names plus optional result_dir "
        "(root for processed output) and database_path (root for raw spectrum "
        "hdf5 files). fluxdep never touches hardware, so this only locates files.",
        (
            _str("chip_name"),
            _str("qub_name"),
            _str_opt("result_dir", "Root for processed output"),
            _str_opt("database_path", "Root for raw spectrum hdf5 files"),
        ),
    ),
    "project.info": MethodSpec(
        5.0,
        "Read the current project info (chip_name, qub_name, result_dir, "
        "database_path).",
    ),
    # Spectrum collection
    "spectrum.load": MethodSpec(
        30.0,
        "Load a raw spectrum hdf5 into the collection and return its assigned "
        "name. spec_type is 'OneTone' or 'TwoTone'. inherit_from optionally "
        "names an already-loaded spectrum whose flux alignment is copied as an "
        "initial guess. transpose_axes=true swaps the device-value and frequency "
        "axes at load (for legacy files stored as x=frequency / y=flux).",
        (
            _str("filepath", "Path to the raw spectrum hdf5 file"),
            _str("spec_type", "'OneTone' or 'TwoTone'"),
            _str_opt("inherit_from", "Loaded spectrum to inherit alignment from"),
            _bool_opt(
                "transpose_axes",
                "Swap dev-value/frequency axes for legacy x=freq/y=flux files",
            ),
        ),
    ),
    "spectrum.load_processed": MethodSpec(
        30.0,
        "Restore a processed spectrums.hdf5 (aligned spectra with selected "
        "points) into the collection; returns the loaded names. NOTE: spec_type "
        "is not persisted, so a missing type defaults to TwoTone.",
        (_str("filepath", "Path to a processed spectrums.hdf5 file"),),
    ),
    "spectrum.list": MethodSpec(
        5.0,
        "List the loaded spectra: each {name, spec_type, aligned, points_selected}.",
    ),
    "spectrum.remove": MethodSpec(
        5.0, "Remove a loaded spectrum by name", (_str("name", "Spectrum name"),)
    ),
    "spectrum.set_active": MethodSpec(
        5.0,
        "Set the active spectrum (the one the editor view operates on)",
        (_str("name", "Spectrum name to activate"),),
    ),
    # Alignment / points
    "alignment.set": MethodSpec(
        5.0,
        "Set a spectrum's flux alignment (flux_half / flux_int) and mark it "
        "aligned. Both are flux numbers.",
        (
            _str("name", "Spectrum name"),
            _num("flux_half", "Flux value of the half-flux-quantum point"),
            _num("flux_int", "Flux value of the integer-flux-quantum point"),
        ),
    ),
    "points.set": MethodSpec(
        10.0,
        "Set a spectrum's selected points and mark points selected. dev_values "
        "and freqs are JSON arrays of equal length (converted to numpy arrays "
        "server-side).",
        (
            _str("name", "Spectrum name"),
            _json("dev_values", "JSON array of device values"),
            _json("freqs", "JSON array of frequencies (MHz)"),
        ),
    ),
    # Cross-spectrum selection
    "selection.pointcloud": MethodSpec(
        5.0,
        "Derive the joint (flux, freq) point cloud assembled from every "
        "spectrum's selected points. Returns {fluxs:[...], freqs:[...]}.",
    ),
    "selection.set": MethodSpec(
        5.0,
        "Set the cross-spectrum selection mask over the joint point cloud. "
        "'selected' is a JSON array of booleans whose length must equal the "
        "joint point-cloud size (selection.pointcloud).",
        (_json("selected", "JSON array of booleans over the joint point cloud"),),
    ),
    # Export
    "export.spectrums": MethodSpec(
        30.0,
        "Write the whole spectrum collection to a spectrums.hdf5 file and return "
        "its path. Omit filepath to use the notebook-layout default "
        "(result_dir/data/fluxdep/spectrums.hdf5).",
        (_str_opt("filepath", "Override the export path"),),
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
