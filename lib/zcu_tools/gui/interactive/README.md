# Shared interactive analysis

**Last updated:** 2026-10-05

This Qt-free module owns detached state, owner-loop publication, subscriptions,
single-level undo, typed actions and command declarations for GUI applications.
Application services own sessions and terminal operation settlement.
Frontends own pointer preview and presentation, not a second committed state.
Domain plugins supply state, actions and terminal validation.

Session undo consumes the snapshot before the last successful commit.
It does not create redo. Closed input rejects mutations; disposal rejects reads.
Measure uses the shared contracts without changing its wire commands.

`SharedFluxPickPlugin` supplies device-axis line actions and single-flight auto
alignment for measure and fluxdep. Numerical calculations stay in
`analysis.fluxdep`; terminal result construction belongs to the injected app
binding. Alignment completion installs positions against the latest snapshot,
retains current presentation flags and ignores closed input.
