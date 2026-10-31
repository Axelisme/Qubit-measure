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
# Analyze errors use the same failed/error completion as every other operation.
# result_state.stop_requested records frontend Stop and selector cancel requests.
# Writeback receipts, validated recipe batches, all-tone diagnostics and analysis recovery.
# value.read identifies unregistered sources with a structured reason and path.
# Record writeback uses record IDs/kinds and written.records; no sample aliases.
# Workflow records expose executor operations without program acquisition observations.
# Predictor queries require device_values and explicit v2 flux; calibration returns Phi0 correction.
# Tab admission shares domain-first reasons before wire source checks.
# Results list exposes named item issues for unenumerable Database paths.
# point.list includes parameter-only availability and per-point error diagnostics.
# Context snapshots no longer include library provenance.
# Entry info, point bindings and writeback targets use folder names, without UUIDs.
# Loaded Runs expose recorded parameter content independently of live storage.
# Point library revisions may be null; library writers accept an explicit null base.
# Point file paths use the shared parameters.yaml/library.yaml document names.
# Global/point readiness, active three files, installed-file reload and point creation.
# Writeback destinations report entry_name/nullable point without a kind field.
# Workflow Pause/Resume controls are removed; Ask uses waiting in workflow/recipe status.
WIRE_VERSION = 136

# Bump for meaningful GUI code changes that need a reload signal.
# Wire-contract changes bump both versions.
GUI_VERSION = 169
