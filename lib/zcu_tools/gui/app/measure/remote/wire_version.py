"""Measure-gui versions reported by the no-auth ``wire.version`` handshake.

Each app owns its wire contract and GUI code revision. MCP pins and compares
WIRE_VERSION; it reports GUI_VERSION without comparing it.
"""

from __future__ import annotations

# Workflow Ask answers require the exact request ID; records expose commit directories.
# Writeback previews include adapter-authored initial selection and notes.
# Library/recipe diagnostics, explicit raw-save observations and predictor files.
# Async recipe declarations use request-scoped answers and script/run cancellation.
# Writeback acceptance checks all pane IDs, then writes current owner drafts.
# Scope status/cancel replies include live_paths and ambiguous candidates.
# Analysis work failures carry failure_kind; current-draft writeback remains unchanged.
# result_state.partial_result preserves inner cancellation independently of root Stop.
# Writeback receipts, validated recipe batches, all-tone diagnostics and analysis recovery.
# value.read identifies unregistered sources with a structured reason and path.
# Record writeback uses record IDs/kinds and written.records; no sample aliases.
WIRE_VERSION = 123

# Bump for meaningful GUI code changes that need a reload signal.
# Wire-contract changes bump both versions.
GUI_VERSION = 156
