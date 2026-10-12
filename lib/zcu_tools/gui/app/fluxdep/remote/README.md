# `zcu_tools.gui.app.fluxdep.remote` — Fluxdep driving adapter

**Last updated:** 2026-10-12，native fit render and database metadata

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

Fit inputs use `search_db`, distinct from the project's raw-data root.
`fit.set_params` remains complete replacement; omitted separate frequencies clear
those fields. The kernel rejects unknown categories before I/O and reads DB
metadata only when pairs or dynamic categories need the level count. Rejection
or I/O failure leaves the current fit untouched.
`fit.database_info(path)` returns candidate/level counts and inclusive parameter
ranges without loading energy values. `fit.render()` uses the same native Show
overlay as Qt, returns PNG without saving an artifact, and requires a fit result.
Both queries are readonly.

GUI editing gates, modal Cancel/Close, pure viewing and shutdown remain frontend
responsibilities. Fluxdep has no hardware Stop exception; the user reclaims from
the indicator before cancelling search or an interactive session.

Authority and connection rules are in
[ADR-0070](../../../../../../docs/adr/0070-control-identity.md).
Native pipeline and UI boundaries are in the [app README](../README.md), and MCP
projection is in the [MCP README](../../../../mcp/fluxdep/README.md).

Local modal dialogs refuse owner acquisition with `local_busy` and "Local 使用中".
Qt app adapters query the active modal on the owner turn; shared control remains
Qt-free. Readonly and existing Remote holder reconnects keep their contracts.
