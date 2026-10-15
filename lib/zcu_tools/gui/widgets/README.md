# Shared GUI widgets

**Last updated:** 2026-10-16

`gui.widgets` owns app-independent Qt presentation helpers. Hosts keep domain
state and commands. The shared Form editor has its own [README](form/README.md).

`error_colour(widget)` returns an error foreground from the widget's current
Window palette lightness. Form errors and Measure Save output errors use this
single rule. Callers apply the colour to text or borders and refresh it when
their palette changes.
