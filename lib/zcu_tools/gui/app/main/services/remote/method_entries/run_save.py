"""Run Save remote method entries."""

from __future__ import annotations

from zcu_tools.gui.remote.method_spec import McpMethodPolicy, MethodSpec

from ._params import (
    _comment,
    _expected_versions,
    _str,
    _str_opt,
)
from ._registry import RemoteMethodEntry, method_entry

METHODS: tuple[RemoteMethodEntry, ...] = (
    method_entry(
        "tab.run_start",
        "run_save:_h_tab_run_start",
        MethodSpec(
            5.0,
            "Start a run (fire-and-forget)",
            (_str("tab_id"), _expected_versions()),
            mcp=McpMethodPolicy.override(
                "gui_tab_run_start",
                reason="manual MCP tool adds short-wait handle and figure folding",
            ),
        ),
    ),
    method_entry(
        "tab.load_data",
        "run_save:_h_tab_load_data",
        MethodSpec(
            30.0,
            "Load a canonical result file into an already-open adapter tab. The tab "
            "then has a run result and can be analyzed without a SoC connection. "
            "First release does not backfill Config from cfg_snapshot.",
            (
                _str("tab_id"),
                _str("data_path", "Canonical HDF5 result file to load"),
                _expected_versions(),
            ),
        ),
    ),
    method_entry(
        "tab.run_cancel",
        "run_save:_h_tab_run_cancel",
        MethodSpec(
            5.0,
            "Request cancellation of the current run (op-specific cancel; there is no "
            "generic cancel — see ADR-0026 §8). Returns {ok, cancelled}: ok is always "
            "true (the call succeeded); cancelled is BEST-EFFORT — true when a live run "
            "was signalled to stop, false (a graceful no-op) when no run was in flight. "
            "It does NOT mean the worker has stopped: the run's true terminal "
            "('cancelled') is observed by gui_op_wait/gui_op_poll on the run handle.",
        ),
    ),
    method_entry(
        "run.running_tab",
        "run_save:_h_run_running_tab",
        MethodSpec(
            5.0,
            "Current running tab",
            mcp=McpMethodPolicy.internal(
                "folded into gui_overview and tab listing surfaces"
            ),
        ),
    ),
    method_entry(
        "tab.save_data",
        "run_save:_h_tab_save_data",
        MethodSpec(
            30.0,
            "Save data file",
            (
                _str("tab_id"),
                _str_opt("data_path", "Override data path"),
                _comment(),
                _expected_versions(),
            ),
            mcp=McpMethodPolicy.override(
                "gui_tab_save",
                reason="manual MCP tool merges data/image save selectors",
            ),
        ),
    ),
    method_entry(
        "tab.save_image",
        "run_save:_h_tab_save_image",
        MethodSpec(
            30.0,
            "Save image file",
            (
                _str("tab_id"),
                _str_opt("image_path", "Override image path"),
                _expected_versions(),
            ),
            mcp=McpMethodPolicy.override(
                "gui_tab_save",
                reason="manual MCP tool merges data/image save selectors",
            ),
        ),
    ),
    method_entry(
        "tab.save_post_image",
        "run_save:_h_tab_save_post_image",
        MethodSpec(
            30.0,
            "Save the post-analysis figure image file (the post sub-tab's own Save "
            "Image). Mirrors tab.save_image but targets the tab's post-analysis figure; "
            "requires a post-analysis result.",
            (
                _str("tab_id"),
                _str_opt("image_path", "Override image path"),
                _expected_versions(),
            ),
            mcp=McpMethodPolicy.override(
                "gui_tab_save",
                reason="manual MCP tool merges data/image save selectors",
            ),
        ),
    ),
    method_entry(
        "tab.save_result",
        "run_save:_h_tab_save_result",
        MethodSpec(
            30.0,
            "Save the result's data and image",
            (
                _str("tab_id"),
                _str_opt("data_path", "Override data path"),
                _str_opt("image_path", "Override image path"),
                _comment(),
                _expected_versions(),
            ),
            mcp=McpMethodPolicy.override(
                "gui_tab_save",
                reason="manual MCP tool merges data/image save selectors",
            ),
        ),
    ),
    method_entry(
        "tab.save_set_paths",
        "run_save:_h_tab_save_set_paths",
        MethodSpec(
            5.0,
            "Set the tab's default save destinations (data + image). Echoes the "
            "applied {data_path, image_path}. Version-guarded on the tab's save_path: "
            "rejects with precondition_failed if a concurrent edit moved it.",
            (
                _str("tab_id"),
                _str("data_path"),
                _str("image_path"),
                _expected_versions(),
            ),
            tool_name="gui_tab_set_save_paths",
        ),
    ),
)
