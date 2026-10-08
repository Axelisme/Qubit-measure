"""Run Save remote method entries."""

from __future__ import annotations

from zcu_tools.gui.remote.method_spec import MethodSpec
from zcu_tools.gui.remote.param_spec import JsonType, ParamSpec

from ._params import (
    optional_integer,
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
            "Save the current measurement_data artifact to every native/Labber output. "
            "Non-cancellable and lease-free. Returns operation_id and complete reserved "
            "artifacts; await the operation, then read tab.snapshot(operation_id=save ID) "
            "for the same complete report. Reserved outputs do not prove success. "
            "call_id is the initiating native MCP ID, preserved verbatim. Optional "
            "run_operation_id requires that Run result; it does not replace seen guards.",
            (
                required_string("tab_id"),
                required_string("call_id", "Initiating native MCP call ID"),
                save_comment(),
                optional_integer("run_operation_id"),
            ),
        ),
        agent=AgentMethodPolicy(
            guard_deps=("tab:{tab_id}:result", "tab:{tab_id}:save"),
            operation_key="tab:{tab_id}",
            refresh_after_write=True,
        ),
    ),
    method_entry(
        "tab.save_artifacts",
        "run_save:h_tab_save_artifacts",
        MethodSpec(
            30.0,
            "Save whole artifacts, including every figure, JSON member and destination. "
            "artifacts is all (including Saved) or unique {section,name} refs. "
            "No caller paths. Returns operation_id and complete reservations; await, then "
            "read tab.snapshot(operation_id=save ID) for that report. Failure can contain "
            "saved and failed outputs. call_id is the initiating native MCP ID. "
            "run_operation_id binds the Run source. operation_id optionally binds one "
            "explicitly selected analysis/post_analysis artifact to its analysis source.",
            (
                required_string("tab_id"),
                required_string("call_id", "Initiating native MCP call ID"),
                ParamSpec("artifacts", JsonType.JSON, required=False, default="all"),
                save_comment(),
                optional_integer("run_operation_id"),
                optional_integer(
                    "operation_id", "Require the selected analysis source"
                ),
            ),
        ),
        agent=AgentMethodPolicy(
            guard_deps=(
                "tab:{tab_id}",
                "tab:{tab_id}:result",
                "tab:{tab_id}:analyze",
                "tab:{tab_id}:post_analyze",
                "tab:{tab_id}:save",
            ),
            operation_key="tab:{tab_id}",
            refresh_after_write=True,
        ),
    ),
)
