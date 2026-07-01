"""Run Save remote method specs."""

from __future__ import annotations

from zcu_tools.gui.remote.method_spec import MethodSpec

from ._params import (
    _comment,
    _expected_versions,
    _str,
    _str_opt,
)

SPECS: dict[str, MethodSpec] = {
    "tab.run_start": MethodSpec(
        5.0, "Start a run (fire-and-forget)", (_str("tab_id"), _expected_versions())
    ),
    "tab.load_data": MethodSpec(
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
    "tab.run_cancel": MethodSpec(
        5.0,
        "Request cancellation of the current run (op-specific cancel; there is no "
        "generic cancel — see ADR-0026 §8). Returns {ok, cancelled}: ok is always "
        "true (the call succeeded); cancelled is BEST-EFFORT — true when a live run "
        "was signalled to stop, false (a graceful no-op) when no run was in flight. "
        "It does NOT mean the worker has stopped: the run's true terminal "
        "('cancelled') is observed by gui_op_wait/gui_op_poll on the run handle.",
    ),
    "run.running_tab": MethodSpec(5.0, "Current running tab"),
    "tab.save_data": MethodSpec(
        30.0,
        "Save data file",
        (
            _str("tab_id"),
            _str_opt("data_path", "Override data path"),
            _comment(),
            _expected_versions(),
        ),
    ),
    "tab.save_image": MethodSpec(
        30.0,
        "Save image file",
        (
            _str("tab_id"),
            _str_opt("image_path", "Override image path"),
            _expected_versions(),
        ),
    ),
    "tab.save_post_image": MethodSpec(
        30.0,
        "Save the post-analysis figure image file (the post sub-tab's own Save "
        "Image). Mirrors tab.save_image but targets the tab's post-analysis figure; "
        "requires a post-analysis result.",
        (
            _str("tab_id"),
            _str_opt("image_path", "Override image path"),
            _expected_versions(),
        ),
    ),
    "tab.save_result": MethodSpec(
        30.0,
        "Save the result's data and image",
        (
            _str("tab_id"),
            _str_opt("data_path", "Override data path"),
            _str_opt("image_path", "Override image path"),
            _comment(),
            _expected_versions(),
        ),
    ),
    "tab.save_set_paths": MethodSpec(
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
}
