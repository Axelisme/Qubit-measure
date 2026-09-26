---
status: accepted
---

# ADR-0061 — Measure interactive analysis owns a committed session

## Context

The measure flux-dependence pickers previously created their own session inside a generic Qt control renderer. The service received that View-owned session only at Done. This left no framework-owned state for `tab_interact` ([[0060]]) to read or change, and pointer movement could not distinguish a local preview from a committed selection. Notebook line pickers share numerical rules but have a different interaction lifecycle ([[0028]]).

## Decision

- `gui.app.main.interactive` owns a Qt-free `Session[S]` per active tab. Only the owner loop may read or mutate it ([[0053]]). Snapshots detach from stored state. An `Action` calculates and validates a complete replacement against the latest committed snapshot, commits it once, then notifies subscribers. A failed calculation publishes nothing; subscriber failures do not undo a commit. Large read-only spectrum inputs belong to the plugin, not the session.
- An adapter with `analysis=INTERACTIVE` explicitly supplies a stable `PluginDefinition` and a Qt-only frontend factory. The plugin declares typed Actions and wire commands with `ParamSpec`; GUI calls the Actions directly. Both entry points use the same numerical rules, without generic framework branches for flux-line roles or a second View-owned state. `done` is a reserved terminal command; cancellation uses the existing operation.
- `AnalyzeService` captures run result, context, adapter and parameters at start, opens the session and retains the existing per-tab operation handle. The frontend only requests finish or cancel. Finish checks the committed snapshot before closing input, then builds a result and follows the existing pane/draft/content-commit path ([[0048]]). A validation failure leaves the session editable. Result construction or recording failure settles the operation as failed; cancel and widget setup failure dispose the session. Late callbacks cannot mutate a disposed session. A GUI Figure can be attached to the terminal result; a widget-free caller does not promise a Figure.
- The flux frontend owns Matplotlib artists, pointer selection, timers and preview. The first left click near a line selects it; pointer movement without a pressed button only previews a candidate. A valid second left click recalculates on the latest session snapshot and commits; button release never commits. An invalid placement, external commit, Esc, focus loss or hide drops the preview. The plugin owns one Auto Align single-flight status for both GUI and remote, captures numeric inputs for background computation, and commits on the owner loop. Pending work does not prevent Done. Terminal delivery is discarded.
- Measure uses click-to-follow/click-to-place instead of a held drag. Notebook `InteractiveLines` retains its existing immediate-drag behavior and does not join the measure session. Both use `analysis.fluxdep.line_state` for numerical rules.

## Consequences

A GUI can show local preview while remote reads committed state; `tab.interact` identifies `preview_active` as presentation metadata, not another analytical state. The GUI-side wire handler discovers and validates plugin-declared commands, projects committed state, and finishes the original operation on `done`. It does not add an MCP bridge route or change the generic RPC channel ([[0059]], [[0060]]). There is no plugin discovery, undo history, conflict merge or session persistence.
