# Measure Qt views

**Last updated:** 2026-10-16

Views project controller snapshots and submit commands through control facets.
They do not own result data or Save policy. See the [Measure README](../README.md)
for pane ownership and operation lifecycles.

`ArtifactSaveCenter` shows each output on two lines: the filename, status and
actions above, and the secondary-colour path below. Failure messages appear
below the path and use the shared `gui.widgets.error_colour` rule.
