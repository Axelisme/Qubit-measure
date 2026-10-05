# `zcu_tools.plotting.fluxdep` — Fluxdep diagnostic figures

**Last updated:** 2026-10-06 — shared OneTone and TwoTone state projection

`make_search_diagnostic_figure(result)` accepts the completed `DatabaseSearchResult`
from [`analysis.fluxdep.search`](../../analysis/fluxdep/README.md) and returns a
native Matplotlib `Figure` with an Agg canvas. It draws the observed and predicted frequencies,
plus per-parameter distance scatter (including lower bounds for pruned entries).
It does not register with pyplot, present, or accept/save the result. Notebook and
GUI adapters choose when to show and retain it. GUI adopts the named figure into
its explicit Plots owner; Notebook publishes the ordinary figure with display.

`pick.make_flux_pick_figure(inputs, state)` returns a separate native Figure with
an Agg canvas for an accepted device-axis selection. It reuses `TwoLinePicker`
rendering without pyplot registration, display, widgets, or figure adoption. The
caller owns naming, presentation and release. Numerical validation stays in the
shared analysis kernel. `pick.configure_flux_pick_axes(figure)` supplies the
coordinate-grid presentation shared by the fluxdep Qt preview and native output,
without changing analysis state or canvas ownership.

`onetone.OneTonePickPlot` projects captured OneTone inputs and committed peak
indices onto device/GHz and normalized-amplitude panels. The Qt OneTone view and
`make_onetone_pick_figure` share its layout and reusable artists. The builder
returns a separate Agg Figure; projecting onto a supplied GUI Figure does not
change its canvas. Neither path commits threshold or selection state.

`twotone.TwoTonePickPlot` renders a numerical `TwoTonePickView` onto a supplied
Figure in native device/GHz axes. Current points, optional mask and positional
added/removed changes share the Qt/native renderer. Stroke endpoint outlines use
normalized radius scaled by each axis span. `make_twotone_pick_figure` computes
the projection and returns a separate Agg Figure, with optional previous-snapshot
feedback for Undo. Neither renderer commits state or owns a Qt canvas.

This package does not own Qt canvas attachment, backend selection, interactive
`TwoLinePicker` gestures, or analysis state. Importing the root `plotting` package
does not load this renderer.
