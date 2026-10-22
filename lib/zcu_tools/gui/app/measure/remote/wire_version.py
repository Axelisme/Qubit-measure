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
# Scoped Run control and result_state.partial_result preserve inner cancellation.
WIRE_VERSION = 119

# Bump for meaningful GUI code changes that need a reload signal.
# Wire-contract changes bump both versions.
GUI_VERSION = 152
