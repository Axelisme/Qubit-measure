# `zcu_tools.plotting.fluxdep` — Fluxdep diagnostic figures

**Last updated:** 2026-10-24

**讀者：** framework-dev。通量頻譜搜尋與選點診斷圖的 builder 契約。

`make_search_diagnostic_figure(result)` accepts the completed `DatabaseSearchResult`
from [`analysis.fluxdep.search`](../../analysis/fluxdep/README.md) and returns a
native Matplotlib `Figure` with an Agg canvas. It draws the observed and predicted frequencies,
plus per-parameter distance scatter (including lower bounds for pruned entries).
The figure uses constrained layout, so each draw re-fits it to the current canvas:
after a Qt resize the title and every subplot's axis labels stay clear of the
neighbouring subplots. It does not register with pyplot, present, or accept/save the result. Notebook and
GUI adapters choose when to show and retain it. GUI adopts the named figure into
its explicit Plots owner; Notebook publishes the ordinary figure with display.

`pick.make_flux_pick_figure(inputs, state, *, onetone=False)` returns a separate native Figure with
an Agg canvas for an accepted device-axis selection. It reuses `TwoLinePicker`
rendering without pyplot registration, display, widgets, or figure adoption;
`onetone=True` gives a OneTone spectrum the sample picker's background. The
caller owns naming, presentation and release. Numerical validation stays in the
shared analysis kernel. `pick.configure_flux_pick_axes(figure)` supplies the
coordinate-grid presentation shared by the fluxdep Qt preview and native output,
without changing analysis state or canvas ownership.

`onetone.OneTonePickPlot` projects captured OneTone inputs and committed peak
indices onto device/GHz and normalized-amplitude panels. The Qt OneTone view and
`make_onetone_pick_figure` share its layout and reusable artists. The builder
returns a separate Agg Figure; projecting onto a supplied GUI Figure does not
change its canvas. Neither path commits threshold or selection state.
Its background is `analysis.fluxdep.processing.onetone_background` (smoothed
magnitude subtracted from its maximum, with percentile colour limits), so
resonator dips draw dark under `gray_r`. The fluxdep GUI result preview, the
OneTone line picker and both OneTone PNGs (line pick and spectrum) call the same
function, so every OneTone view shares one background. It lives in analysis
because `TwoLinePicker` needs it and analysis cannot import plotting.

`twotone.TwoTonePickPlot` renders a numerical `TwoTonePickView` onto a supplied
Figure in native device/GHz axes. Current points, optional mask and positional
added/removed changes share the Qt/native renderer. The latest stroke draws as
the band its brush swept along the whole path (`brush.BrushBand`), and
`show_pointer` draws an uncommitted drag the same way. Each update refits the
layout and wraps the title, so both title lines stay inside a narrow figure. `make_twotone_pick_figure` computes
the projection and returns a separate Agg Figure, with optional previous-snapshot
feedback for Undo. Neither renderer commits state or owns a Qt canvas.

`cross_selection.CrossSelectionPlot` shares calibrated flux/GHz rendering between
Qt and `make_cross_selection_figure`. Background cells follow sample centers,
including descending/nonuniform axes. Kept/dropped points, optional index-based
changes and the stroke's brush band use the captured joint bounds; drags and
the title behave as in the TwoTone renderer.
Native output owns a separate Agg canvas; rendering preserves the supplied canvas.

`spectrum.make_spectrum_figure` renders a published raw spectrum in native
device/GHz axes, with optional committed half/integer lines and sampled points.
`onetone` selects the OneTone dip-depth background; TwoTone draws `|signals|`.
It returns an independent Agg Figure without opening an interactive session.
The GUI RPC owner captures the published data and encodes the PNG; MCP decides
when to request this evidence.

`brush.BrushBand` draws the region a normalized-radius brush covers along a
stroke: one compound path of the circle at every vertex and the strip between
consecutive vertices, filled once under the nonzero rule. It follows the
circle-per-sample geometry of `analysis.fluxdep.stroke`, so the band shows what
the stroke will select or erase.

`palette` holds the point colours every fluxdep view uses, Qt pickers and
previews as well as notebook and MCP PNGs. They come from the Okabe–Ito palette,
which red–green colour-blind users can tell apart: Kept/Current orange, Added
blue hollow circles, Removed reddish-purple crosses, Dropped grey and the brush
sky blue. `transition_color(i)` cycles the eight Okabe–Ito colours for the fit's
compiled transitions, so Show's model lines and attributed points match. Half/integer flux lines use `HALF_FLUX_COLOR`/`INTEGER_FLUX_COLOR`
from `analysis.fluxdep.line_picker`, the line picker's own red and blue, in
pickers, previews and the spectrum PNG.

OneTone and TwoTone share this package's private sample-centered background
renderer. It keeps descending and nonuniform coordinates with their signal data;
it does not depend on lab live tools or a standalone liveplot backend.

This package does not own Qt canvas attachment, backend selection, interactive
`TwoLinePicker` gestures, or analysis state. Importing the root `plotting` package
does not load this renderer.
