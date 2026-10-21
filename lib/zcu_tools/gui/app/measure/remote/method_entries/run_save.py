"""Run Save remote method entries."""

from __future__ import annotations

from zcu_tools.gui.remote.method_spec import MethodSpec
from zcu_tools.gui.remote.param_spec import JsonType, ParamSpec

from ._params import optional_integer, required_string, save_comment
from ._registry import AgentMethodPolicy, RemoteMethodEntry, method_entry

METHODS: tuple[RemoteMethodEntry, ...] = (
    method_entry(
        "tab.run_start",
        "run_save:h_tab_run_start",
        MethodSpec(
            5.0,
            "Start the current cfg inputs with fixed execution values. "
            "Requires an explicitly selected or newly created working point. "
            "Nonempty aliases are rejected before focus or operation effects. "
            "Reset with aliases first. None and {} use displayed Form values and "
            "the fixed Open/Reset parameter capture. Invalid Form errors reject "
            "admission. Use wait(op=handle) for terminal "
            "status, failure/cancellation and Send & Stop feedback. The GUI "
            "returns an operation_id, which MCP exposes as {handle}; starting "
            "is not completion. Run completion does not save raw data; call "
            "tab.save_data explicitly to save it. tab.snapshot.raw_data reports "
            "the acquisition run_id and whether measurement_data is fully saved. "
            "After completion, read result state with "
            "rpc_call on tab.snapshot, or the run figure with rpc_call on "
            "tab.get_figure using subtab_id=run.",
            (
                required_string("tab_id"),
                ParamSpec(
                    "aliases",
                    JsonType.OBJECT,
                    required=False,
                    description="Nonempty overrides are rejected; Reset with aliases first. None and {} use the fixed tab capture.",
                ),
            ),
        ),
        agent=AgentMethodPolicy(
            operation_key="tab:{tab_id}",
        ),
    ),
    method_entry(
        "tab.load_data",
        "run_save:h_tab_load_data",
        MethodSpec(
            30.0,
            "Load an exact native result file and its original history into an "
            "already-open adapter tab. The tab "
            "then has a run result and can be analyzed without a SoC connection. "
            "Form input stays unchanged. cfg_issue is null when typed cfg is "
            "available, or a readable reason when valid data loads without cfg. "
            "Read the file's original raw cfg offline as StoredRun through the "
            "public datafile.load_run_data API; the GUI has no raw cfg inspection "
            "entry point. analysis_error "
            "reports analysis preparation failure without undoing the load; "
            "has_analyze_params is false until preparation succeeds.",
            (
                required_string("tab_id"),
                required_string("data_path", "Exact native data.h5 file to load"),
            ),
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
            readonly=True,
        ),
        agent=AgentMethodPolicy(exposure="internal"),
    ),
    method_entry(
        "tab.save_data",
        "run_save:h_tab_save_data",
        MethodSpec(
            30.0,
            "Synchronously save measurement_data to every native/Labber output on "
            "the GUI owner thread. Returns the original Run ID, even if all outputs "
            "fail, and the complete "
            "current artifacts with paths/status/error; no Save handle or await. "
            "Partial failures stay in the report. run_operation_id optionally "
            "binds the Run source.",
            (
                required_string("tab_id"),
                save_comment(),
                optional_integer("run_operation_id"),
            ),
        ),
    ),
    method_entry(
        "tab.save_artifacts",
        "run_save:h_tab_save_artifacts",
        MethodSpec(
            30.0,
            "Synchronously save complete artifacts, including all figures, JSON "
            "and destinations. artifacts is all (including Saved) or unique "
            "{section,name} refs. No caller paths. Returns the original Run ID, "
            "even if all outputs fail, and "
            "this attempt's complete paths/status/error; no Save handle or await. "
            "run_operation_id binds the Run source. operation_id optionally binds "
            "one explicitly selected analysis/post_analysis artifact to its "
            "analysis source, not to a Save operation.",
            (
                required_string("tab_id"),
                ParamSpec("artifacts", JsonType.JSON, required=False, default="all"),
                save_comment(),
                optional_integer("run_operation_id"),
                optional_integer(
                    "operation_id", "Require the selected analysis source"
                ),
            ),
        ),
    ),
)
