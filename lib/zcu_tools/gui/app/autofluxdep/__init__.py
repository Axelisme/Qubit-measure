"""autofluxdep-gui — automated flux-dependence measurement workflow (skeleton).

A *control-type* GUI (Phase B+ drives ZCU216 + flux device to sweep flux ×
multiple experiments), unlike the analysis-type fluxdep/dispersive GUIs. It is
NOT built on the runner-based ``experiment.v2.autofluxdep`` module; instead each
experiment is a **Node** declaring its dependencies (with same/prev/first time
scope) and the information it provides, and an ``Orchestrator`` sweeps flux ×
topologically-ordered Nodes — replacing the old ``cfg_maker`` lambda +
``ctx.env["info"]`` walrus chains with an explicit dependency model.

Skeleton (Phase A) scope — domain core only, no hardware, no Qt:
- ``nodes.spec``       — NodeSpec / Dependency (key + smooth flag) / build_cfg
- ``nodes.qubit_freq`` — one worked Node translating the notebook's cfg_maker
- ``orchestrator``     — runner-free flux × Node sweep + dependency resolution
- ``state`` / ``event_bus`` / ``controller`` / ``app`` — composition root shells

See ``task_plans/tool_gui/autofluxdep_gui_assessment.md`` for the full plan.
"""
