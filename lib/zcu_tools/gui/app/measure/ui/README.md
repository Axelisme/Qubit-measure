# Measure Qt views

**Last updated:** 2026-10-19，experiment figure area owner

Views project controller snapshots and submit commands through control facets.
They do not own result data or Save policy. See the [Measure README](../README.md)
for pane ownership and operation lifecycles.

`ExpTabWidget` owns the left forms and ledgers, commands, subscriptions,
progress and splitter width. Its concrete owned child `ExperimentFigureArea`
owns the right-side canvases, named figure selectors and caches, interactive
presentation and Data gallery. It keeps selection, signal blocking and canvas
cleanup together. Data activation refreshes previews before switching pages,
then reflows the visible gallery. The tab's overridable public figure reader
remains the preview source; each failed pane read leaves only that preview empty.
Figure/container methods on the tab delegate to this owner. Results and Save
policy stay with their existing owners. The existing foldable ledger and matching
form/action builders live in `experiment_ledger.py`.

[ADR-0067](../../../../../../docs/adr/0067-gui-application.md#experiment-figure-area)
records the 2026-10-19 decision and the conditions for similar Qt child ownership
extractions. This is presentation ownership, not a second application controller.

For every adapter, `MainWindow` projects the current tab's cfg invalid reason to the bottom status on interaction refresh and tab selection, while Run stays disabled.
A valid cfg clears only that cfg diagnostic; background tabs and valid refreshes do not clear unrelated status messages.

`ArtifactSaveCenter` shows each output on two lines: the filename, status and
actions above, and the secondary-colour path below. Failure messages appear
below the path and use the shared `gui.widgets.error_colour` rule.
