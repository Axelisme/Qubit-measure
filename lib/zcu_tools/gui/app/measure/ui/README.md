# Measure Qt views

**Last updated:** 2026-10-23

**讀者：** framework-dev。量測 Qt views、對話框與呈現生命週期。

Programmatic close uses the same drain, persistence and remote teardown as local
close. The remote adapter waits for terminal reply send before calling
`request_shutdown()`. Its deferred Qt close turn prevents reentrancy and does
not confirm socket delivery.

Views project controller snapshots and submit commands through control facets.
They do not own result data or Save policy. `MainDialogRegistry` retains truly
non-modal named windows. `MainWindow` synchronizes Setup, SoC, Devices,
Inspect and ARB editing on control transfer, including windows already open.
Run separately locks Setup and SoC submissions; Inspect, Devices and ARB keep
their existing Run policy. Setup… and SoC… have separate toolbar entries; the named `soc`
window participates in remote screenshots.
Error notifications are non-modal; confirmations, choosers and writeback editing
keep their modal protection. See the [Measure README](../README.md)
for pane ownership and operation lifecycles.

`MainWindow` uses one toggle to show experiment tabs or workflow tabs in the
same stacked area. New Tab lists registrations for the displayed kind.
`WorkflowTabs` subscribes to the shared workflow facet, so remote opens mount
panes without going through a GUI selection command. Same-name tabs keep
monotonic display numbers. The status badge returns to the active workflow.
The shared RUN lease locks competing panes and Setup/SoC submissions.

`WorkflowPane` renders the facet's Plan and Tunables drafts with `FormEditor`.
It forwards Refresh, Start, Pause, Resume, Stop and Apply to the facet.
The active_path label shows the root-to-first-fork chain. The live_paths label
shows each live leaf full path on its own line; neither label picks a latest child.
Busy Stop remains enabled under Remote control and reclaims locked Local before
calling the same facet. Pause, Resume, Answer and tunables retain the holder gate.
The services own source capture, lowering, accepted tunables and run lifetime.
Ask displays the authored question and driver-owned answer schema, with a JSON
answer field. A new question selects Ask; the status badge returns there from
other tabs. Answer submits to the retained worker. The pane preserves the last
exchange from the owner's `last_exchange`, including answers submitted through
facet/RPC/MCP. Only accepted answers display 「已回答」; pending submissions display
「回答已提交」. Cancelled questions display 「提問已取消」 and disable old answers.
`WorkflowFigures` owns only named canvas presentation through `QtPlotHost`.
Its selector separates Workflow and Current operation groups. Both use the
headless owner's `WorkflowPlots.figures()` projection, with operation/<name>
entries. Releasing an old operation removes its entire group before the next
attempt presents figures. A refused close keeps the pane.
After accepted close, the view detaches its subscriptions and editors and stops
its redraw timers before deleting children. Closing the last pane does not drain
or disconnect the session.

`ExpTabWidget` owns the left forms and ledgers, commands, subscriptions,
progress and splitter width. Its concrete owned child `ExperimentFigureArea`
owns the right-side canvases, named figure selectors and caches, interactive
presentation and Data gallery. It keeps selection, signal blocking and canvas
cleanup together. Data activation refreshes previews before switching pages,
then reflows the visible gallery. The tab's overridable public figure reader
remains the preview source; each failed pane read leaves only that preview empty.
Figure/container methods on the tab delegate to this owner. Results and Save
policy stay with their existing owners.

Run, Analysis and Post-Analysis show up to four figures together in a two-column
layout. Larger sets show one figure with a named selector. Data groups raster
previews by source stage and applies the same rule within each group. The
app-local plot host passes the operation's original named collection through the
render-host port after canvas attachment. It does not rename figures, copy their
collection or change FigureContainer membership. Clearing a pane retires its
presentation borrow, not the caller's saved figures. The existing foldable ledger and matching
form/action builders live in `experiment_ledger.py`.

[ADR-0067](../../../../../../docs/adr/0067-gui-application.md#experiment-figure-area)
describes how `ExpTabWidget` and `ExperimentFigureArea` divide presentation
responsibilities. `ExperimentFigureArea` is not a second application controller.

For every adapter, `MainWindow` projects the current tab's cfg invalid reason to the bottom status on interaction refresh and tab selection, while Run stays disabled.
A valid cfg clears only that cfg diagnostic; background tabs and valid refreshes do not clear unrelated status messages.

`AnalyzeFormWidget` keeps invalid optional scalar text visible. Only blank text
means None. A committed parse failure shows a field-labelled error and publishes
`validity_changed(False)` without publishing params. Valid edits publish a typed
params instance and `validity_changed(True)`. For a populated form, the tab
combines current validity with readiness and authority to gate Analyze and
Post-Analyze. Invalid edits keep `has_params()` true. An unpopulated form keeps
the existing readiness gate without claiming parameter validity. Hydration emits
neither signal. `read_params()` raises ValueError on unparseable input;
callers must not substitute the last valid params.

`RunLoadDialog` projects the opened entry\'s `runs/*/data.h5` metadata rows from
`SaveControlPort.list_saved_runs`. It displays historical start/finish times and
completion, not ledger event times. Unreadable metadata stays visible as a file
row with its error and a selectable path. Selection does not promise Load success;
the existing Load owner reports file/format errors after selection.

`ArtifactSaveCenter` shows each output on two lines: the filename, status and
actions above, and the secondary-colour path below. Failure messages appear
below the path and use the shared `gui.widgets.error_colour` rule.
