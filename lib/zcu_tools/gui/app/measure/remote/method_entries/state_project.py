"""State Project remote method entries."""

from __future__ import annotations

from zcu_tools.gui.remote.method_spec import MethodSpec

from ._params import (
    _bool_default,
)
from ._registry import AgentMethodPolicy, RemoteMethodEntry, method_entry

METHODS: tuple[RemoteMethodEntry, ...] = (
    method_entry(
        "state.has_project",
        "state_project:h_state_has_project",
        MethodSpec(
            5.0,
            "",
        ),
        agent=AgentMethodPolicy(exposure="internal"),
    ),
    method_entry(
        "state.has_context",
        "state_project:h_state_has_context",
        MethodSpec(
            5.0,
            "",
        ),
        agent=AgentMethodPolicy(exposure="internal"),
    ),
    method_entry(
        "state.has_active_context",
        "state_project:h_state_has_active_context",
        MethodSpec(
            5.0,
            "",
        ),
        agent=AgentMethodPolicy(exposure="internal"),
    ),
    method_entry(
        "state.has_soc",
        "state_project:h_state_has_soc",
        MethodSpec(
            5.0,
            "",
        ),
        agent=AgentMethodPolicy(exposure="internal"),
    ),
    method_entry(
        "state.hardware_gate",
        "state_project:h_state_hardware_gate",
        MethodSpec(
            5.0,
            "Read active hardware exclusion leases with kind, origin, note, and age.",
        ),
        agent=AgentMethodPolicy(exposure="internal"),
    ),
    method_entry(
        "soc.info",
        "state_project:h_soc_info",
        MethodSpec(
            5.0,
            "Read the connected SoC's hardware summary (QICK soccfg): a compact "
            "human-readable 'description' table (per-channel generator/readout type, "
            "converter port, sample rate, max pulse/buffer length), 'is_mock', "
            "and the successful GUI connection's address/port (null for mock). "
            "The structured 'cfg' (the full ~2 KB QICK config) is included only when "
            "include_cfg=true (default false), so the common case pays nothing for it. "
            "Requires a connected SoC. Only include_cfg=true fully reveals the SoC "
            "for a version guard; a summary does not. The SoC has no teardown "
            "(Pyro4-backed): there is no disconnect / reconnect / health-check "
            "tool (deferred, E3).",
            (_bool_default("include_cfg", False, "Include the full ~2 KB QICK cfg"),),
        ),
        agent=AgentMethodPolicy(
            reveals=("soc",), reveals_when_nonempty=("include_cfg",)
        ),
    ),
    method_entry(
        "project.info",
        "state_project:h_project_info",
        MethodSpec(
            5.0,
            "Read the applied project identity: chip_name / qub_name / res_name plus "
            "the resolved result_dir and database_path. Fast-fails with "
            "precondition_failed (no_project) when no project is applied yet.",
        ),
        agent=AgentMethodPolicy(exposure="internal"),
    ),
    method_entry(
        "result_scope.list",
        "state_project:h_result_scope_list",
        MethodSpec(
            10.0,
            "List discovered result scopes from result/**/params.json. Each scope "
            "carries {scope_id, chip_name, qub_name, result_dir, params_path, source}; "
            "startup.apply may pass a returned scope_id to use an existing non-generated "
            "scope. Existing params.json files are migrated in place to schema v1 "
            "project identity when needed; this read never creates new params.json files.",
        ),
    ),
    method_entry(
        "resources.versions",
        "state_project:h_resources_versions",
        MethodSpec(
            5.0,
            "Snapshot of all resource versions",
        ),
        agent=AgentMethodPolicy(exposure="internal"),
    ),
)
