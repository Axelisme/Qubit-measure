"""Per-app wire / GUI code version constants for the fluxdep-gui.

These are the two hand-maintained version integers reported by the no-auth
``wire.version`` handshake. They are intentionally NOT in the shared
``zcu_tools.gui.remote.wire`` module — each GUI app evolves its own wire contract
independently, so the constants live beside the app's own RemoteControlAdapter.
"""

from __future__ import annotations

# Two independent hand-maintained versions reported by the no-auth
# ``wire.version`` handshake (which also surfaces the MCP server's own
# MCP_VERSION). Only WIRE_VERSION is *compared*; the code revisions are
# *reported* so an agent can eyeball whether a reload took effect:
#
#   WIRE_VERSION — the mcp<->RPC *interface contract* (RPC method set, their
#     params, event/serialization shape). The MCP server pins it; a mismatch
#     means the two sides speak different protocols → hard MISMATCH. Bump ONLY
#     on a contract change.
#   GUI_VERSION  — this GUI code's *revision*. Reported, never compared. Bump on
#     any meaningful GUI change you want to be able to spot a reload of,
#     INCLUDING pure-internal logic changes that don't touch the wire.
# This unpublished interface revision includes identity-only project setup,
# data_root, spectrum evidence/drafts and fit search_db/render/database_info.
# 9: a new Filter context starts from the published masks (all-selected only
# without one), so selection.interactive.open evidence reports that seed.
WIRE_VERSION = 9

# GUI code revision (see header). Bump on any meaningful GUI change you want a
# stale-process check to flag; independent of WIRE_VERSION.
# 12: window shell with Spectra/Fit pages, editing flow and picker columns,
# Fit page form/status, shared spectrum row query; two-line spectrum list
# rows, one dip-depth OneTone background in every view (OneTone PNGs too),
# constrained Search diagnostic layout; Search disabled while a needed
# frequency is blank, non-holder Filter tab switches detach only; literal
# spectrum targets and independent global Filter lifetime; inherited
# alignment preserved on presentation-only draft edits.
# 13: Spectra stage bar (1 Align -> 2 Sample) replaces the result preview's
# next-step buttons; a list pick shows the published preview first.
# 14: Show display controls follow control authority and a Show zoom toolbar;
# colour-blind-safe shared point palette, brush band along the whole stroke,
# picker titles refit on update, faded (unset) frequencies, inherited
# half/integer lines on the Add spectrum preview.
# 15: Fit stage bar (1 Filter -> 2 Search -> 3 Show) with locks; Filter shows
# the published masks without a Session, Edit/Discard/Finish own the Session,
# leaving Filter only detaches; a new Filter starts from the published masks.
GUI_VERSION = 15
