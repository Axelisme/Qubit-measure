"""Run Save remote method entries."""

from __future__ import annotations

from zcu_tools.gui.remote.method_spec import MethodSpec
from zcu_tools.gui.remote.param_spec import JsonType, ParamSpec

from ._params import (
    optional_integer,
    optional_string,
    required_object,
    required_string,
    save_comment,
)
from ._registry import AgentMethodPolicy, RemoteMethodEntry, method_entry

METHODS: tuple[RemoteMethodEntry, ...] = (
    method_entry(
        "tab.run_start",
        "run_save:h_tab_run_start",
        MethodSpec(
            5.0,
            "Start a tab run with the explicitly observed cfg ref, without "
            "refresh or substitution; use wait(op=handle) for terminal "
            "status, failure/cancellation and Send & Stop feedback. The GUI "
            "returns an operation_id, which MCP exposes as {handle}; starting "
            "is not completion. After completion, read result state with "
            "rpc_call on tab.snapshot, or the run figure with rpc_call on "
            "tab.get_figure using subtab_id=run.",
            (
                required_string("tab_id"),
                required_object("expected", "Observed cfg_id and string revision"),
            ),
        ),
        agent=AgentMethodPolicy(
            guard_deps=(
                "tab:{tab_id}",
                "soc",
                "device:*",
                "devices:__set__",
            ),
            operation_key="tab:{tab_id}",
            refresh_after_write=True,
        ),
    ),
    method_entry(
        "tab.load_data",
        "run_save:h_tab_load_data",
        MethodSpec(
            30.0,
            "Load a canonical result file into an already-open adapter tab. The tab "
            "then has a run result and can be analyzed without a SoC connection. "
            "Compatible snapshot values backfill Config automatically, replacing "
            "unsubmitted edits. cfg_backfill reports applied or not_applied; a "
            "backfill failure does not undo the loaded result. analysis_error "
            "reports analysis preparation failure without undoing the load; "
            "has_analyze_params is false until preparation succeeds.",
            (
                required_string("tab_id"),
                required_string("data_path", "Canonical HDF5 result file to load"),
            ),
        ),
        agent=AgentMethodPolicy(
            guard_deps=(
                "tab:{tab_id}",
                "tab:{tab_id}:result",
                "tab:{tab_id}:analyze",
                "context",
            ),
            refresh_after_write=True,
        ),
    ),
    method_entry(
        "tab.run_cancel",
        "run_save:h_tab_run_cancel",
        MethodSpec(
            5.0,
            "Request cancellation of the current run. Returns {ok, cancelled}: ok is always "
            "true (the call succeeded); cancelled is BEST-EFFORT — true when a live run "
            "was signalled to stop, false (a graceful no-op) when no run was in flight. "
            "It does NOT mean the worker has stopped: the run's true terminal "
            "('cancelled') is observed by wait(op) on the run handle.",
        ),
        agent=AgentMethodPolicy(exposure="internal"),
    ),
    method_entry(
        "run.running_tab",
        "run_save:h_run_running_tab",
        MethodSpec(
            5.0,
            "Current running tab",
        ),
        agent=AgentMethodPolicy(exposure="internal"),
    ),
    method_entry(
        "tab.save_data",
        "run_save:h_tab_save_data",
        MethodSpec(
            30.0,
            "Start non-cancellable data saving without a hardware lease. Explicit "
            "data_path/comment update the GUI draft; omitted values keep it. "
            "Returns an operation handle and reserved path, not proof of success. "
            "Wait for completion and read artifacts for the last successful path.",
            (
                required_string("tab_id"),
                optional_string("data_path", "Override data path"),
                save_comment(),
                optional_integer("run_operation_id"),
            ),
        ),
        agent=AgentMethodPolicy(
            guard_deps=(
                "tab:{tab_id}:result",
                "tab:{tab_id}:path:data",
            ),
            operation_key="tab:{tab_id}",
            refresh_after_write=True,
        ),
    ),
    method_entry(
        "tab.save_artifacts",
        "run_save:h_tab_save_artifacts",
        MethodSpec(
            30.0,
            "Start one non-cancellable save operation over selected artifacts. "
            "Keys are data, analysis:<name> and post:<name>; all selects unsaved artifacts. "
            "Explicit paths/comment update the shared drafts. Returns operation_id "
            "and reserved destinations, not proof of completion. Read artifacts "
            "after terminal failure for partial successes.",
            (
                required_string("tab_id"),
                ParamSpec("artifacts", JsonType.JSON, required=False, default="all"),
                ParamSpec(
                    "paths",
                    JsonType.OBJECT,
                    required=False,
                    default={},
                    description="Artifact key to destination path",
                ),
                save_comment(),
            ),
        ),
        agent=AgentMethodPolicy(
            exposure="tool",
            tool_names=("tab_save",),
            guard_deps=(
                "tab:{tab_id}",
                "tab:{tab_id}:result",
                "tab:{tab_id}:analyze",
                "tab:{tab_id}:post_analyze",
                "tab:{tab_id}:path:data",
                "tab:{tab_id}:path:analysis_image",
                "tab:{tab_id}:path:post_analysis_image",
            ),
            operation_key="tab:{tab_id}",
            refresh_after_write=True,
        ),
    ),
    method_entry(
        "tab.save_image",
        "run_save:h_tab_save_image",
        MethodSpec(
            30.0,
            "Save one named canonical image (analysis|post_analysis only; run "
            "has no canonical image). Requires tab_id, subtab_id and figure_name. "
            "The pane is analysis|post_analysis. Explicit image_path updates the GUI "
            "draft before saving; omission keeps the draft, and an empty path is rejected. "
            "Optional operation_id rejects a replaced result before changing paths or exporting.",
            (
                required_string("tab_id"),
                required_string("subtab_id", "Pane: analysis|post_analysis"),
                required_string("figure_name"),
                optional_string("image_path", "Override image path"),
                optional_integer(
                    "operation_id", "Require this analysis operation's current result"
                ),
            ),
        ),
        agent=AgentMethodPolicy(
            guard_deps=(
                "tab:{tab_id}:result",
                "tab:{tab_id}:post_analyze",
                "tab:{tab_id}:path:analysis_image",
                "tab:{tab_id}:path:post_analysis_image",
            ),
            refresh_after_write=True,
        ),
    ),
)
