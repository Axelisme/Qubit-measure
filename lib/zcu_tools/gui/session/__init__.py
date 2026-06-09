"""Shared measurement-session core for the GUI apps.

App-agnostic (no app import); the apps that drive a QICK measurement session —
``main`` (measure-gui) and ``autofluxdep`` — build on these. A "session" here is
the measurement substrate every such app needs *before* it layers its own
experiment surface: the active context (MetaDict + ModuleLibrary), the SoC
connection, the multi-device set, and the async-operation lifecycle.

This package is import-clean: importing its leaf modules pulls in neither Qt nor
matplotlib nor any ``zcu_tools.gui.app.*`` package, so it can sit *below* the
apps without a back-edge. Submodules are imported on demand:

- ``types`` — session value types (``ExpContext`` + readiness, the ``SocHandle`` /
  ``SocCfgHandle`` structural surfaces). No experiment cfg-tree coupling.
- ``events`` — the session event vocabulary: ``SessionEvent`` + the data/SoC/
  device/predictor payloads, on the shared ``BaseEventBus`` (payload-type keyed).
- ``operation_handles`` — the async-operation Handle/Cancel facet (ADR-0019),
  pure token mint/settle/await/poll/cancel with zero operation-kind knowledge.

Each app keeps its own experiment surface (tabs/run for measure, node-sweep for
autofluxdep) and its own ``OperationGate`` exclusion policy; only the
session-core mechanism lives here.
"""
