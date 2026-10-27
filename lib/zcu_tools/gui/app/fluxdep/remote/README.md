# `zcu_tools.gui.app.fluxdep.remote` — Fluxdep driving adapter

**Last updated:** 2026-10-28

**讀者：** framework-dev。Fluxdep RPC 的控制權、方法宣告與結果投影契約。

The adapter exposes the same Controller commands as Qt. Shared remote owns
framing, authentication, queues and holder admission. Fluxdep supplies method
schemas, readonly declarations, native result projection and application
control identity. MCP consumes this contract; the GUI does not own MCP tools.

`command_specs` exposes named sampling/Filter ParamSpec declarations from the
same domain owners used by native plugins. MCP projects inclusive limits and
dependent-limit descriptions from these declarations, without constructing a
plugin, directly importing domain modules or generating one tool per command.

Business methods require holder unless explicitly readonly. Pipeline writes,
search cancellation and interactive commands retain domain validation and busy
checks. Readonly queries and bounded search waits remain available to observers.
Reads do not establish per-connection version guards.

Each spectrum retains its own draft and Undo; Filter has one global context.
Interactive context IDs bind commands to the owning session and named target.
`spectrum.interactive.open(name, kind="sample")` asks the GUI owner to choose the
picker from the published spectrum type. Unknown names and unaligned sources
receive native admission errors. The receipt reports onetone/twotone, never
sample. Explicit line/onetone/twotone requests keep their existing contracts.
interactive.read(target="spectrum", name=<literal name>) is inert; its optional
context_id rejects cancelled, removed, replaced or wrong-target drafts. Numeric replies contain no figure or command schema.
interactive.read/render require target="spectrum" with a nonempty literal name,
or target="selection" without any name key (including null). Missing/invalid
target or invalid target/name combinations fail with INVALID_PARAMS; no target
is inferred from names, display, existing drafts or IDs. The spectrum name
"selection" is legal and never addresses Filter. interactive.render returns the
current draft PNG without opening input or switching display. Filter finish publishes and closes
its global context. selection.render() always draws the published joint cloud
and per-spectrum masks, even while a draft exists. It does not open a Session.
Replacing an unfinished same-spectrum
stage reports discarded=align/sample. Retirement and terminal receipts remain app-owned. Search outcomes, publication failures and
partial output do not imply rollback. Timeout does not cancel accepted work.

Fit inputs use `search_db`, distinct from the project's raw-data root.
`fit.set_params` remains complete replacement; omitted separate frequencies clear
those fields. The kernel rejects unknown categories before I/O and reads DB
metadata only when pairs or dynamic categories need the level count. Rejection
or I/O failure leaves the current fit untouched.
`fit.database_info(path)` returns candidate/level counts and inclusive parameter
ranges without loading energy values. `fit.render()` uses the same native Show
overlay as Qt, returns PNG without saving an artifact, and requires a fit result.
Both queries are readonly. fit.result projects full-precision recorded kernel
quality, all compiled transition entries and native stale reasons. Null quality
means no recorded search diagnostics. Upstream publication retains the fit;
fit.export_params rejects stale results before I/O with fit_stale and reasons.
Stage finish receipts include the publication's transitive invalidated labels,
captured before event delivery. Reentrant successors do not change that receipt.
Private pipeline counters are absent from resources.versions.

Export/Restore saves sampled checkpoints, including empty completed spectra,
without Filter masks. Restore rejects the entire request before publication when
any name already exists and lists every conflict.

GUI editing gates, modal Cancel/Close, pure viewing and shutdown remain frontend
responsibilities. Fluxdep has no hardware Stop exception; the user reclaims from
the indicator before cancelling search or an interactive session.

Authority and connection rules are in
[ADR-0070](../../../../../../docs/adr/0070-control-identity.md).
`project.setup(project_root, entry, component?)` opens an existing entry under
`project_root/result`. Omitted component resolves the setup qubit alias. The
component must be a fluxonium qubit. `project.info` returns entry, component and
Result paths, with null fields before opening. The Database binding is unused.
`fit.export_params` accepts no path override. It writes the selected component
predictor, declared setup fields and analysis ledger, then returns savepath,
params, alignment_from, component, analysis_id, flux_writeback and predictor_writeback.
Both outcomes have status written/skipped, the intended device or null, and a
skip reason or null. flux_writeback describes component P1; predictor_writeback
independently describes the dataset predictor v2 file. Skipped predictors have
savepath=null and leave any existing predictor untouched. Setup/ledger still write.
The GUI displays both outcomes. The app README documents device/snapshot skip rules.
The app submits the injected planner's component-relative fields, not a
framework interpretation of lab flux schema. Missing planner or rejected
calibration fails before output. Failure preserves the successful write prefix
without rollback. Spectrum list/snapshot and
selection snapshot report unfinished drafts separately from published data.
`spectrum.list` reports each published loaded/aligned/sampled stage and the count
excluded by that spectrum's published Filter mask. Completed empty spectra remain
sampled; unaligned spectra with retained points remain loaded. Draft edits and
min_distance downsampling do not change the exclusion count.
`spectrum.snapshot` retains complete arrays and the current published alignment
source. Draft edits do not change that source; finish publishes it with alignment.
`spectrum.render` is readonly and returns a PNG of published raw data, alignment
lines and points without creating a session or changing display.

Native pipeline and UI boundaries are in the [app README](../README.md), and MCP
projection is in the [MCP README](../../../../mcp/fluxdep/README.md).

Local modal dialogs refuse owner acquisition with `local_busy` and "Local 使用中".
Qt app adapters query the active modal on the owner turn; shared control remains
Qt-free. Readonly and existing Remote holder reconnects keep their contracts.
