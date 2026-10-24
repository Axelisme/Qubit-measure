# Measure Qt views

**Last updated:** 2026-10-24

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
It forwards Refresh, Start, Pause, Resume, Stop, Apply and Discard to the facet.
Discard restores accepted tunables and clears renderer-local invalid text. It is
available only for changed drafts while editing is allowed. Entering paused
selects Tunables and shows the Resume/Apply hint; Resume and Stop hide it.
A new Ask takes focus after pause navigation. Stopping makes tunables read-only
and disables Apply and Discard.
The active_path label shows the root-to-first-fork chain. The live_paths label
shows each live leaf full path on its own line, but hides when its only leaf
matches active_path. Neither label picks a latest child.
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
It tiles workflow and current-operation figures together, up to four per page.
Larger sets show Previous page / Next page controls and an `i / N` label.
The hidden name model follows the current canvas: changing pages selects the
first figure on that page, and clicking a tile selects that figure. It uses the
headless owner's `WorkflowPlots.figures()` projection, with operation/<name>
entries. Releasing an old operation removes its entire group before the next
attempt presents figures. A refused close keeps the pane.
After accepted close, the view detaches its subscriptions and editors and stops
its redraw timers before deleting children. Closing the last pane does not drain
or disconnect the session.

`ExpTabWidget` owns the left forms and ledgers, commands, subscriptions,
progress and splitter width. Its concrete owned child `ExperimentFigureArea`
owns the right-side canvases, figure paging, named figure models and caches, interactive
presentation and Data gallery. It keeps selection, signal blocking and canvas
cleanup together. Data activation refreshes previews before switching pages,
then reflows the visible gallery. The tab's overridable public figure reader
remains the preview source; each failed pane read leaves only that preview empty.
Figure/container methods on the tab delegate to this owner. Results and Save
policy stay with their existing owners.

Run, Analysis, Post-Analysis and Workflow share a two-column tiled layout with
up to four figures per page. Larger sets show Previous page and Next page controls
with an i / N label instead of a named selector. `named_figure_presentation` keeps
the hidden name model and current tile in sync for experiment and workflow panes.
Changing pages selects its first figure; clicking a canvas tile selects that
figure for screenshots and current-figure reads. Re-presenting the same collection
retains the page and current figure. Data groups raster previews by source stage
and applies the same paging rule within each group. The
app-local plot host passes the operation's original named collection through the
render-host port after canvas attachment. It does not rename figures, copy their
collection or change FigureContainer membership. Clearing a pane retires its
presentation borrow, not the caller's saved figures. The existing foldable ledger and matching
form/action builders live in `experiment_ledger.py`.

[ADR-0067](../../../../../../docs/adr/0067-gui-application.md#experiment-figure-area)
describes how `ExpTabWidget` and `ExperimentFigureArea` divide presentation
responsibilities. `ExperimentFigureArea` is not a second application controller.

Cfg invalid reasons stay in the form area and disabled Run tooltip. A history tab
without a form retains the factory's original reason there. Edits, interaction
refreshes and tab selection preserve unrelated bottom-status messages. They do
not cover the tab or figure area with a cfg diagnostic. Load behavior is unchanged.

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

The Results menu retains an independent top-level Results browser. Closing the
measure window closes this browser. It reads the Controller results query once
and filters the shared index without changing the opened entry or point.
The navigation tree starts with entries, then experiment → point → run or
point → experiment → run. Both orders share focus history and keep entries
isolated even when experiment, point and run names match. Group galleries show
one representative PNG per run. Selecting a run shows all its saved PNGs in a
vertically scrolling gallery, separate from the measure figure pager.
Runs without PNGs remain selectable in the tree.
Both grouping orders and the experiment filter use the saved experiment string.
Browsing never queries the catalog. Only Open resolves an adapter through the
catalog owner, then delegates to existing Load admission. Success selects the
new history tab and raises the measure window; failure stays in the right-side
feedback. Deleted experiments and old cfg versions remain browsable.
A left click on a group image opens its run directly and selects that figure.
Within a run, left clicks select the current figure and open or replace one
non-modal Tool preview. Esc closes it; closing the browser hides it. Gallery
cards and decoded rasters stay local to one gallery, with at most one retained
card per indexed PNG. File identity, size and timestamps invalidate changed
members. Back/Forward and grouping reuse unchanged cards and rasters.
Context menus copy a member path or image and pin its run in ResultsComparison.
`ResultsComparison` shows the fixed-column table above a filename-aligned image
grid, separated by a vertical splitter. All pins are run columns; PNG filenames
are rows. Missing members show an em dash. Headers unpin or open a run. Pin count
updates the Comparison tab caption. Context-menu pinning reveals that PNG row;
there is no per-pin selected-image state.
`ResultsDetailsTree` presents metadata/cfg and analysis as collapsible Key/Value
trees. It bounds preview depth and value length, while Copy JSON retains the full
selected section. Headers remain mouse-resizable. Filter by date (UTC) enables
the greyed-out date edits; either their text area or arrow opens the calendar.
新 tab 的 Load 使用既有 application owner。缺少 analysis source 的原因同時顯示在瀏覽器結果與載入後的 tab。
A history tab whose form factory failed shows its original reason in the form
area and disabled Run tooltip. Reset can replace that notice with a real editor;
a failed retry leaves the notice and shows the current preparation error.

`ArtifactSaveCenter` shows each output on two lines: the filename, status and
actions above, and the secondary-colour path below. Failure messages appear
below the path and use the shared `gui.widgets.error_colour` rule.
