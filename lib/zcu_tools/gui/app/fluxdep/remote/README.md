# `zcu_tools.gui.app.fluxdep.remote` — Fluxdep driving adapter

**Last updated:** 2026-10-13，named interactive state and evidence

The adapter exposes the same Controller commands as Qt. Shared remote owns
framing, authentication, queues and holder admission. Fluxdep supplies method
schemas, readonly declarations, events, native result projection and application
control identity. MCP consumes this contract; the GUI does not own MCP tools.

Business methods require holder unless explicitly readonly. spectrum.set_active
is readonly display selection: it does not begin, edit or retire a draft. Pipeline writes,
search cancellation and interactive commands retain domain validation and busy
checks. Readonly queries and bounded search waits remain available to observers.
Reads do not establish per-connection version guards.

Each spectrum retains its own draft and Undo; Filter has one global context.
Interactive context IDs bind commands to the owning session and named target.
interactive.read(name) is inert; its optional context_id rejects cancelled,
removed or replaced drafts. Numeric replies contain no figure or command schema.
interactive.render(name), or name=selection for Filter, returns the current PNG
without opening input or switching display. Replacing an unfinished same-spectrum
stage reports discarded=align/sample. Retirement and terminal receipts remain app-owned. Search outcomes, publication failures and
partial output do not imply rollback. Timeout does not cancel accepted work.

GUI editing gates, modal Cancel/Close, pure viewing and shutdown remain frontend
responsibilities. Fluxdep has no hardware Stop exception; the user reclaims from
the indicator before cancelling search or an interactive session.

Authority and connection rules are in
[ADR-0070](../../../../../../docs/adr/0070-control-identity.md).
`project.setup` accepts chip/qubit identity only; GUI derives `result_dir` and
`data_root` from the controller root. Undeclared request keys follow shared
parameter handling and cannot override those paths. Spectrum list/snapshot and
selection snapshot report unfinished drafts separately from published data.
`spectrum.snapshot` retains complete arrays and the current published alignment
source. Draft edits do not change that source; finish publishes it with alignment.
`spectrum.render` is readonly and returns a PNG of published raw data, alignment
lines and points without creating a session or changing display.

Native pipeline and UI boundaries are in the [app README](../README.md), and MCP
projection is in the [MCP README](../../../../mcp/fluxdep/README.md).

Local modal dialogs refuse owner acquisition with `local_busy` and "Local 使用中".
Qt app adapters query the active modal on the owner turn; shared control remains
Qt-free. Readonly and existing Remote holder reconnects keep their contracts.
