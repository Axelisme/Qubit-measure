# Measure Qt views

**Last updated:** 2026-10-16

Views project controller snapshots and submit commands through control facets.
They do not own result data or Save policy. See the [Measure README](../README.md)
for pane ownership and operation lifecycles.

`ArtifactSaveCenter` shows each output on two lines. The filename, saved mark and
right-aligned Open folder action share the first line. The full-width path below
uses smaller, secondary palette text without an input frame. It remains
read-only and selectable, elides in the middle, and keeps the full path in its
tooltip and folder action. A failed output adds a wrapping, selectable error
below the path. Its foreground uses the shared `gui.widgets.error_colour` rule
for light and dark palettes. Snapshot refreshes retain widgets and selections.
Save commands, output evidence and artifact-header status colours are unchanged.
