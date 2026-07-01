"""State Project remote method specs."""

from __future__ import annotations

from zcu_tools.gui.remote.method_spec import MethodSpec

from ._params import (
    _bool_default,
)

SPECS: dict[str, MethodSpec] = {
    "state.has_project": MethodSpec(5.0, ""),
    "state.has_context": MethodSpec(5.0, ""),
    "state.has_active_context": MethodSpec(5.0, ""),
    "state.has_soc": MethodSpec(5.0, ""),
    "soc.info": MethodSpec(
        5.0,
        "Read the connected SoC's hardware summary (QICK soccfg): a compact "
        "human-readable 'description' table (per-channel generator/readout type, "
        "converter port, sample rate, max pulse/buffer length) plus 'is_mock'. "
        "The structured 'cfg' (the full ~2 KB QICK config) is included only when "
        "include_cfg=true (default false), so the common case pays nothing for it. "
        "Requires a connected SoC. The SoC has no teardown (Pyro4-backed): there "
        "is no disconnect / reconnect / health-check tool (deferred, E3).",
        (_bool_default("include_cfg", False, "Include the full ~2 KB QICK cfg"),),
    ),
    "project.info": MethodSpec(
        5.0,
        "Read the applied project identity: chip_name / qub_name / res_name plus "
        "the resolved result_dir and database_path. Fast-fails with "
        "precondition_failed (no_project) when no project is applied yet.",
    ),
    "result_scope.list": MethodSpec(
        10.0,
        "List discovered result scopes from result/**/params.json. Each scope "
        "carries {scope_id, chip_name, qub_name, result_dir, params_path, source}; "
        "startup.apply may pass a returned scope_id to use an existing non-generated "
        "scope. Existing params.json files are migrated in place to schema v1 "
        "project identity when needed; this read never creates new params.json files.",
        tool_name="gui_result_scope_list",
    ),
    "resources.versions": MethodSpec(5.0, "Snapshot of all resource versions"),
}
