# Mux Ramsey

**Readers:** operator, agent. Run mux Ramsey from the GUI or MCP and read its results.

**Last updated:** 2026-10-21

## GUI setup

Bind `$mux` to a MuxGroup, then open `mux/t2ramsey` from **New Tab**.
The captured parameter source must contain every qubit and its matching resonator
in readout order. Readout references the module slot `$mux.module.readout` when
present, preserving the library's tones and scalar settings. Use a mux-family
module with the full group tone count. If the slot is empty, Readout starts
inline from the blank mux template with group calibration tones. Complete all
required fields and verify the hardware settings before running.

The Tones picker selects the element to edit. Participants selects the qubits to
run. Tone names, count and order stay fixed. Set Participants to a nonempty JSON
list of unique existing names, for example `["Q3", "Q1"]`. This runs original
positions 2/0 in that order, even while the picker shows Q2. Empty, duplicate or
unknown participants block Run and remain editable for correction.

Every tone needs valid drives, including nonparticipants. Each tone uses its own
drive generator, distinct from the readout generator. Errors in another picker element
appear on the Tones row. Select that element and correct it to enable Run.
Reps, rounds and recovery delay are shared. Recovery initially uses five times
the largest captured T1, with the default for missing T1 values. A stored invalid
T1 leaves recovery blank for correction.

## Measurement and analysis

Use fixed x90 drives and an individual detune in MHz for each tone.
All tones share one finite, nonnegative requested-delay range in us.

The sequence is common reset, simultaneous first x90 pulses, common delay,
simultaneous phase-ramped final x90 pulses, then mux readout.
The delay starts after the longest first branch. Pre/post delays stay configured.
With zero pulse delays, a shorter pulse waits an extra `max_len - len_i`.
This can change fringe amplitude and constant phase.
Analysis uses the common requested delay. Enable Fit phase when a constant phase
offset matters. The final phase uses each source phase plus its detune times delay.
Native timing can differ by a few ticks.

Fit fringe selects damped-cosine fitting; disabling it selects exponential decay.
A fringe fit needs six finite points and decay needs four.
Fixed fringe phase uses the nearest-zero 0/180-degree policy; Fit phase frees it.
Decay uses the same model regardless of Fit phase.
T2* and error are in us; fringe rate and error are in MHz.
Sampling quality uses the largest gap between acquired delays and the larger
absolute fitted or actual detune. At least ten samples per cycle and a complete
cycle in the scan window are required for adequate sampling. Fewer samples per
cycle are undersampled. A shorter window, zero frequency or no delay gap is
unavailable. Decay fits have no fringe sampling quality. Adequate describes the
supplied frequencies; it does not rule out unknown higher-frequency aliases.

Actual detune retains the hardware phase-wrap convention, including for signed
detunes. Check the scan range before treating it as a signed frequency correction.
A nonzero nominal detune with no positive rounded delay can yield NaN actual detune.

## Writeback

Initial T2/sample selection requires positive finite T2, r2 at least 0.9,
normalized residual RMS at most 0.1 and relative T2 error at most 0.5.
Frequency candidates also require adequate sampling, at least 10 points per full
fringe cycle, and relative frequency error at most 0.5. Zero fitted frequency has
no relative error and is not initially selected. Mux Ramsey primarily shows the
decay/T2 trend; a frequency candidate is not precision-detuning evidence.
Failed or unavailable checks leave candidates unselected with explanatory notes.
You may select them manually after reviewing the source and fit.

Each qualifying tone proposes `<qubit>.t2r` in us and a T2R sample at captured bias.
A frequency candidate additionally requires source configuration, a fringe fit,
finite source carrier, finite actual detune and finite fitted fringe rate.
It uses source carrier + actual detune - fitted fringe rate in MHz.
NaN actual detune suppresses only the frequency correction.
Review phase/rate conventions, uncertainty and captured destination before applying.

Frequency writeback changes the qubit calibration, not the existing x90/x180
library pulses. Recalibrate the pulses at the new frequency, for example with
Rabi or zigzag, then save and select new numeric modules with the recalibrated
gains and lengths. Changing only frequency does not calibrate the pulses.
Calibration writeback, New from a template and `ml_edit` can create modules.
Template expressions read calibration at creation, not on later Runs.
Reset downstream tabs after changing module selections. Historical Run/Load
sources retain their original pulses.

## Results and saved sources

Run draws one `measurement_i` figure per participant, without a shared overlay.
Each figure uses the common requested-delay axis.
The index `i` is the original tone position in the saved configuration.
Analyze uses the selected Run/Load source. Use the Analysis figure picker for
`fit_i`.
Here `i` is the original tone position, so Q3/Q1 uses indices 2/0.
Summary rows retain that order and show values, quality or a failure reason.
A missing source configuration leaves the name unavailable.

Use Data to save measurement data and analysis artifacts.
Load Data selects a native file or saved run and preserves current Form inputs.
The selected source fixes the historical pulse, binding and writeback destination.
Cancellation retains acquired data and leaves missing data as NaN.
Partial data from a failed operation retains its failure status.
Only successful, fully finite rows qualify for calibration candidates.
Missing captured bindings suppress candidates; inconsistent bindings report an error.
Read each Writeback destination, value, quality and uncertainty.
Deselect unwanted items, then use **Apply Selected** to write the selected items.
Analyze and preview only calculate results. Each candidate needs its own selection.

Native and Labber outputs preserve IQ on the delay/tone grid and detune/source
frequency on the tone-only grid. Check that both outputs report `saved`.
Inspect each output's error when saving fails.

## Drafts and MCP inputs

Saved drafts retain invalid field text and Participants for correction.
Restoration reports dropped fields and keeps the original parameter source.
Unreadable saved inputs produce a diagnostic and fresh defaults.
Reset captures current parameters and rebuilds initial values.

MCP field paths use original tone positions, for example
`("tones", 0, "drive", "gain")`. The integer selects the full tone list.
See the [mux recipe guide](../../../recipes/mux-guide.md#mux_t2ramsey) for typed overrides,
parameter origins and explicit writeback decisions.

The MCP recipe is `mux_t2ramsey`.
