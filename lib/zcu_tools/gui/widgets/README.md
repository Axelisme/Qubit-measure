# Shared GUI widgets

**Last updated:** 2026-10-20, neutral file-dialog paths

`gui.widgets` owns app-independent Qt presentation helpers. Hosts keep domain
state and commands. The shared Form editor has its own [README](form/README.md).

`file_dialog.nearest_existing` chooses the deepest existing ancestor for a
file-dialog starting directory. It has no project identity or app dependency.

## Colours

`error_colour`, `success_colour`, `warning_colour` and `accent_colour` choose
readable text foregrounds from the owner's current Window palette lightness.
`secondary_colour` mixes palette Text toward Window for muted descriptions and
paths. Callers use the widget that owns the visible background, rather than a
child label with a transparent stylesheet background.

`set_foreground(widget, colour)` applies a colour through the palette to
WindowText, Text and the widget's foreground role. Labels and text edits use the
same mechanism for every status branch; stylesheets retain only non-colour
formatting. Table and list items use `setForeground` with the colour functions.
Owners refresh these colours when displaying new state. Writeback also refreshes
its foregrounds when its palette changes, and its ledger background follows
Base. Window and Base use the same light or dark theme. Fixed-background chips keep their paired background and foreground.
