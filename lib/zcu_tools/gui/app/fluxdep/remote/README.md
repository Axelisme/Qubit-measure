# `zcu_tools.gui.app.fluxdep.remote` — Fluxdep driving adapter

**Last updated:** 2026-10-12，原始譜證據與已發布對齊來源

The adapter exposes the same Controller commands as Qt. Shared remote owns
framing, authentication, queues and holder admission. Fluxdep supplies method
schemas, readonly declarations, events, native result projection and application
control identity. MCP consumes this contract; the GUI does not own MCP tools.

Business methods require holder unless explicitly readonly. Pipeline writes,
search cancellation and interactive commands retain domain validation and busy
checks. Readonly queries and bounded search waits remain available to observers.
Reads do not establish per-connection version guards.

Interactive context IDs bind commands to the owning session and target. Replies
use one committed capture for state and Agg PNG; replacement, retirement and
terminal receipts remain app-owned. Search outcomes, publication failures and
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
