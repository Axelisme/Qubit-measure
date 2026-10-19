# Measure Qt views

**Last updated:** 2026-10-20，file-backed Load 清單

Views project controller snapshots and submit commands through control facets.
They do not own result data or Save policy. `MainDialogRegistry` retains truly
non-modal named windows. `MainWindow.set_writable` synchronizes Setup, Devices,
Inspect and ARB editing on control transfer, including windows already open.
Error notifications are non-modal; confirmations, choosers and writeback editing
keep their modal protection. See the [Measure README](../README.md)
for pane ownership and operation lifecycles.

For every adapter, `MainWindow` projects the current tab's cfg invalid reason to the bottom status on interaction refresh and tab selection, while Run stays disabled.
A valid cfg clears only that cfg diagnostic; background tabs and valid refreshes do not clear unrelated status messages.

`RunLoadDialog` projects the opened entry\'s `runs/*/data.h5` metadata rows from
`SaveControlPort.list_saved_runs`. It displays historical start/finish times and
completion, not ledger event times. Unreadable metadata stays visible as a file
row with its error and a selectable path. Selection does not promise Load success;
the existing Load owner reports file/format errors after selection.

`ArtifactSaveCenter` shows each output on two lines: the filename, status and
actions above, and the secondary-colour path below. Failure messages appear
below the path and use the shared `gui.widgets.error_colour` rule.
