# `zcu_tools.plotting.fluxdep` — Fluxdep diagnostic figures

**Last updated:** 2026-10-01 — native terminal flux-pick builder

`make_search_diagnostic_figure(result)` accepts the completed `DatabaseSearchResult`
from [`analysis.fluxdep.search`](../../analysis/fluxdep/README.md) and returns a
pyplot-managed Matplotlib `Figure`. It draws the observed and predicted frequencies,
plus per-parameter distance scatter (including lower bounds for pruned entries).
It does not call `plt.show()` or accept/save the result. Notebook and GUI adapters
choose when to show, retain, or close the figure. GUI routing must already be active
when it creates the figure; pyplot registers it with Gcf until the caller closes it.

`pick.make_flux_pick_figure(inputs, state)` returns a separate native Figure with
an Agg canvas for an accepted device-axis selection. It reuses `TwoLinePicker`
rendering without pyplot registration, display, widgets, or figure adoption. The
caller owns naming, presentation and release. Numerical validation stays in the
shared analysis kernel.

This package does not own Qt canvas attachment, backend selection, interactive
`TwoLinePicker` gestures, or analysis state. Importing the root `plotting` package
does not load this renderer.
