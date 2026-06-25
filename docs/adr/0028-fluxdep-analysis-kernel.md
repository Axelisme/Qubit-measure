---
status: accepted
---

# Flux-Dependence Analysis kernel lives outside notebook and GUI adapters

Flux-Dependence Analysis uses the same point-picking, filtering, line-selection,
and one-tone peak-detection rules from both notebook workflows and Qt GUIs. The
shared domain implementation lives in notebook-neutral `zcu_tools.analysis.fluxdep`;
`zcu_tools.notebook.analysis.fluxdep` and `zcu_tools.gui.app.fluxdep` are adapters
that translate UI events into the kernel interface. The first slice is deliberately
small: complex calculation stays in stateless functions, interaction state lives
in state-machine objects, and database search / plotting / export remain outside
this kernel until separately designed.

