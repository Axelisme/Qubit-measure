"""autofluxdep-gui — automated flux-dependence measurement workflow.

A *control-type* GUI (drives ZCU216 + a flux device to sweep flux × multiple
experiments), unlike the analysis-type fluxdep/dispersive GUIs. It is NOT built
on the runner-based ``experiment.v2.autofluxdep`` module; instead each experiment
is a **Builder** (the stateless kind of provider) that produces a short-lived
**Node** per flux point, and an ``Orchestrator`` — a pure requirement resolver —
sweeps flux × the user-ordered providers, replacing the old ``cfg_maker`` lambda
+ ``ctx.env["info"]`` walrus chains with an explicit dependency model.

- ``nodes.builder``    — Builder / Node / PlacedNode + the RunEnv curried in
- ``nodes.spec``       — Dependency / ModuleDep declaration vocabulary
- ``nodes.qubit_freq`` — the worked measurement Builder (acquire → fit → liveplot)
- ``nodes.predictor``  — a Service Builder (pure-compute Node)
- ``orchestrator``     — runner-free flux × provider sweep + requirement resolution
- ``state`` / ``event_bus`` / ``controller`` / ``app`` — composition root + façade

See ``task_plans/tool_gui/autofluxdep_gui_assessment.md`` for the full plan and
``CONTEXT.md`` for the Builder/Node/Service/Result/Plotter glossary.
"""
