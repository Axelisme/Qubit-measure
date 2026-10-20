# Measure Qt views

**Last updated:** 2026-10-20

Views project controller snapshots and submit commands through control facets.
They do not own result data or Save policy. `MainDialogRegistry` retains truly
non-modal named windows. `MainWindow.set_writable` synchronizes Setup, Devices,
Inspect and ARB editing on control transfer, including windows already open.
Error notifications are non-modal; confirmations, choosers and writeback editing
keep their modal protection. See the [Measure README](../README.md)
for pane ownership and operation lifecycles.

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

`ArtifactSaveCenter` shows each output on two lines: the filename, status and
actions above, and the secondary-colour path below. Failure messages appear
below the path and use the shared `gui.widgets.error_colour` rule.
