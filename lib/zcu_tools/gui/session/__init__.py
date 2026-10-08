"""Shared measurement-session core for the GUI apps.

App-agnostic (no app import); measure-gui builds its QICK measurement session on these. A "session" here is
the measurement substrate every such app needs *before* it layers its own
experiment surface: the active context (MetaDict + ModuleLibrary), the SoC
connection, the multi-device set, and the async-operation lifecycle.

This package is import-clean: importing its leaf modules pulls in neither Qt nor
matplotlib nor any ``zcu_tools.gui.app.*`` package, so it can sit *below* the
apps without a back-edge. Submodules are imported on demand:

- ``types`` — session value types (``SessionEnv`` + readiness, the ``SocHandle`` /
  ``SocCfgHandle`` structural surfaces). No experiment cfg-tree coupling.
- ``events`` — the session event vocabulary: ``SessionEvent`` + the data/SoC/
  device/predictor payloads, on the shared ``BaseEventBus`` (payload-type keyed).
- ``operation_handles`` — the async-operation Handle/Cancel facet (ADR-0066),
  pure token mint/settle/await/poll/cancel with zero operation-kind knowledge.

Measure keeps its tabs/run surface and ``OperationGate`` exclusion policy;
only the session-core mechanism lives here.
"""
