"""Saved-result methods exposed through the live catalog, not fixed MCP tools."""

from zcu_tools.gui.remote.method_spec import MethodSpec

from ._params import required_string
from ._registry import RemoteMethodEntry, method_entry

METHODS: tuple[RemoteMethodEntry, ...] = (
    method_entry(
        "results.list",
        "results:h_results_list",
        MethodSpec(
            30.0,
            "Read all saved Database runs under the current project root as {runs: [...]}. "
            "No opened entry or selected point required. Each row includes opaque key, "
            "entry, nullable run_id/experiment/point/at, captured roles, registered, "
            "exact run_path/data_path/figure_paths/analysis_paths, figures (filename "
            "selectors), search_text including tone names, and issues. Unknown history "
            "stays null. No full cfg/analysis documents or arrays. Fresh disk snapshot "
            "on every call; no filtering, files, locks or session changes.",
            readonly=True,
        ),
    ),
    method_entry(
        "results.figure",
        "results:h_results_figure",
        MethodSpec(
            30.0,
            "Read a saved PNG as {mime_type: 'image/png', png_b64: '...'}. "
            "Use results.list selectors, never arbitrary paths. Fresh disk snapshot "
            "on every call; files may change or disappear between calls. "
            "Unknown run/figure returns invalid_params with unknown_result_run/"
            "unknown_result_figure. I/O and escaping-path failures retain structured "
            "dispatch errors. No rendering, analysis, writes or retry.",
            (
                required_string("run_key", "Opaque key returned by results.list."),
                required_string("figure", "Exact member of the run's figures list."),
            ),
            readonly=True,
        ),
    ),
)
