"""Per-app wire / GUI code version constants for measure-gui.

These are the two hand-maintained version integers reported by the no-auth
``wire.version`` handshake. They are intentionally NOT in the shared
``zcu_tools.gui.remote.wire`` module — each GUI app evolves its own wire contract
independently, so the constants (and their per-app changelog) live beside the
app's own RemoteControlAdapter.
"""

from __future__ import annotations

# Two independent hand-maintained versions reported by the no-auth
# ``wire.version`` handshake (which also surfaces the MCP server's own
# MCP_VERSION). Only WIRE_VERSION is *compared*; the code revisions are
# *reported* so an agent can eyeball whether a reload took effect:
#
#   WIRE_VERSION — the mcp<->RPC *interface contract* (the RPC method set, their
#     params, and the event/serialization shape). The MCP server pins it and
#     compares it on the handshake; a mismatch means the two sides speak different
#     protocols → hard MISMATCH. Bump ONLY on a contract change (a new/removed/
#     renamed RPC method or param, or a change to a reply/event wire shape).
#   GUI_VERSION  — this GUI code's *revision*. Reported, never compared (the MCP
#     server does not pin it — a GUI revision is the GUI process's own property).
#     Bump on any meaningful GUI change you want to be able to spot a reload of,
#     INCLUDING pure-internal logic changes that DON'T touch the wire (that is the
#     point of the split: an internal change bumps GUI_VERSION, leaving the
#     contract version put). A wire-contract change bumps BOTH.
#
# (Git history holds the per-version evolution of both constants.)
# v39: removed redundant wire methods (tab.get_cfg_summary, adapter.cfg_spec,
# adapter.analyze_spec, tab.update_cfg, dialog.open/close/list_open).
# v40: Phase 170a tab cfg I/O normalization — removed old raw tab.get_cfg;
# renamed tab.list_paths -> tab.get_cfg (value tree); added tab.set_cfg
# (tab-keyed batch setter).
# v41: Phase 170b tab listing + run + analyze under tab.* — renamed tab.list ->
# tab.list_all (new 2-tuple shape [tabs, running_tab_id]); run.start ->
# tab.run_start, run.cancel -> tab.run_cancel; analyze.start -> tab.analyze;
# post_analyze.start -> tab.post_analyze; run.running_tab kept as internal-only.
# v42: Phase 170c save + writeback under tab.* — save.data/image/post_image/
# result/set_paths renamed to tab.save_*; writeback.preview/set/apply renamed to
# tab.writeback_*.
# v43: Phase 170d context md/ml prefix + editor open->new/save_as_module->save —
# context.get_md/get_md_attr/set_md_attr/del_md_attr renamed to context.md_get/
# md_get_attr/md_set_attr/md_del_attr; context.get_ml/del_ml_module/del_ml_waveform/
# rename_ml_module/rename_ml_waveform renamed to context.ml_get/ml_del_module/
# ml_del_waveform/ml_rename_module/ml_rename_waveform; ml.list_roles/create_from_role
# moved to context.ml_list_roles/ml_create_from_role; editor.open->editor.new.
# v44: Phase 171 measure-gui interface redesign — the MCP tool *aliases* were
# renamed via method_specs.tool_name / server overrides (those are NOT wire
# changes; the wire dotted method names are unchanged). This bump tracks the
# genuine wire-contract changes (reply shapes + params) made across P1-P4:
#   - startup.apply: returns the resolved project dict
#     {chip_name, qub_name, res_name, result_dir, database_path} instead of bool.
#   - tab.list_all: returns a named dict
#     {tabs:[{tab_id, adapter_name, is_running}], active_tab_id, running_tab_id}
#     instead of the positional 2-tuple [tabs, running_tab_id].
#   - soc.info: new optional param include_cfg (default false) — the ~2 KB QICK
#     cfg is omitted unless requested.
#   - device.list: each entry carries the fine-grained status enum
#     (DeviceStatus value) instead of is_connected:bool.
#   - device.reconnect: returns {operation_id} (now a tracked async op) instead
#     of {}.
#   - predictor.predict: params renamed value->device_value,
#     from_lvl->from_level, to_lvl->to_level.
#   - tab.run_start / tab.analyze / tab.post_analyze / device start ops: START
#     replies surface the operation handle (operation_id) instead of stripping it.
#   - tab.writeback_apply: enriched echo
#     {applied_ids, written, context_version}.
WIRE_VERSION = 44

# GUI code revision (see header). Bump on any meaningful GUI change you want a
# stale-process check to flag; independent of WIRE_VERSION (a wire-contract change
# bumps both; a pure-internal GUI change bumps only this). Git history holds the
# per-version evolution.
GUI_VERSION = 51
