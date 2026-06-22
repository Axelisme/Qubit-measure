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
WIRE_VERSION = 40

# GUI code revision (see header). Bump on any meaningful GUI change you want a
# stale-process check to flag; independent of WIRE_VERSION (a wire-contract change
# bumps both; a pure-internal GUI change bumps only this). Git history holds the
# per-version evolution.
GUI_VERSION = 47
