# Mux Ramsey

**Readers:** operator, agent. Run mux Ramsey from the GUI or MCP and read its results.

**Last updated:** 2026-10-22

After changing the parameter source, use Defaults to recapture initial values. Refresh only updates sources and choices while preserving authored inputs; Defaults can overwrite unsaved inputs.

## GUI setup

Create a `kind: group/mux` component and bind scalar alias `mux` to it, then
open `mux/t2ramsey` from **New Tab**. Without this group and alias, opening the
form reports an error.
The captured parameter source must contain every qubit and its matching resonator
in readout order. Readout references the module slot `$mux.module.readout` when
present, preserving the library's tones and scalar settings. Use a mux-family
module with the full group tone count. If the slot is empty, Readout starts
inline from the blank mux template with group calibration tones.
The generator comes from `$mux.wiring.gen_ch`. Tone i's ADC channel is an
integer seed from the fixed capture's `group.wiring.ro_ch[i]`, in the group's
resonator order. Each tone starts
with gain 0.1. These values are editable initial inputs, not verified hardware
settings. Complete the remaining required fields and confirm wiring, gain and
timing before running.

In Readout, **Length** sets how long the multi-tone pulse plays. **RO length**
sets how long the ADC integrates. **Pre delay** sets when playback starts, and
**Trig offset** sets when the ADC triggers. Both are measured from the start of
the readout segment. **Post delay** is the wait after playback ends. Readout
requires **Mixer freq**.

Each tone's optional **Classifier** stores classification settings such as
centers, radius and a confusion matrix. Mux Ramsey does not use them for the
final readout's mean IQ, its projection, fits or writeback candidates.

The Tones picker selects the element to edit. Participants selects the qubits to
run. Tone names, count and order stay fixed. Set Participants to a nonempty JSON
list of unique original indices, for example `[2, 0]`. This runs original
positions 2/0 in that order, even while the picker shows Q2. Empty, duplicate or
out-of-range participants block Run and remain editable for correction.

Leave **Reset** blank to skip reset. Selected tones need drives on independent
generators, distinct from readout. Open/Defaults fills **Drive** for selected
tones. Unselected tones start blank and can stay blank. Leave **Mixer freq**
blank under Drive to skip configuring a mixer for that drive.
Open/Defaults captures the Aliases row `participants`, for example `[Q3, Q1]`,
and maps it against the captured `$mux.qubits` order to original indices `[2, 0]`.
Alias list elements may use `$` links. A missing, unbound or empty row selects all
original indices in order. An unbound element in a nonempty list rejects Open/Defaults.
An out-of-group name stays visible as an invalid Participants draft until corrected.
Changing aliases requires Defaults or reopening; ordinary participant edits keep tone values.

Editing participants preserves every tone value. Fill any newly selected tone's
missing settings at the indicated tone path. Removing a tone keeps its values. Every readout
tone still needs valid frequency and ADC identity, even when not selected.

## FTF cost example

A two-participant offline Run used 16 delays from 0 to 3.75 µs, 0.5 MHz
nominal detune on both tones, 0.5 µs const x90 pulses, 300 reps and 1 round.
Recovery was 1007.509 µs; readout and ADC integration were 2 and 1.5 µs.
Recorded RPC Run wall time was 19.095 s, about 1.19 s per joint delay point.
It includes first simulation compilation and Run completion waiting, not just
acquisition. Analyze and Save are outside that timer.

This is one startup/cache state, not a cold benchmark for every Run.
Do not compare it with other slots or experiments as a speed test.
Delay point count and range, detune, pulse lengths, participants, reset,
recovery and readout windows affect cost. Hardware repeats acquisitions for
reps and rounds. FTF compilation and cached evolution can dominate, so reps
do not imply proportional wall time. Budget from a short Run with the intended
delay window. This example had about 7.96 samples per cycle and was undersampled.
Its short-scan fit is not calibration evidence.

## Measurement and analysis

For a fringe fit, plan at least 10 samples per expected fringe period: for
an expected rate of 0.25 MHz, use a step no larger than 0.4 µs. Set the
scan span to roughly three times the expected T2* so it covers the decaying
envelope. If the window shows only the first part of the decay, the fitted
T2* relies on extrapolation.

The live figure shows acquired delay points. Both views use mean IQ projected
onto the direction of greatest IQ variation for each tone. The sign and scale
of this projection do not make it a population. The solid curve in Analyze
is the selected fit, and the vertical marker shows T2* when it is within the
scan. R² measures the fraction of the variation explained by the fit.
Normalized residual RMS measures the typical mismatch relative to the signal
range. The ± values are fit parameter errors, not point noise. In Live and Analyze, the
vertical bars show the standard error of the mean (shot SEM): the sample
standard deviation of the same projected shots divided by the square root of
the acquired shot count. Older mean-only files still load and fit without bars.
Points with fewer than two shots have no bar.

Layout reference: Ding Fig. 5-9(b,e) (printed p. 152) couples a Ramsey-like
sequence to phase interpretation. Mux Ramsey scans free-evolution time and
shows projected IQ rather than the conditional gate phase.

Use fixed x90 drives and set **Applied phase-ramp detune** in MHz for each
participating tone. This Form field sets the nominal phase slope of the final
x90 pulse. All tones share one finite, nonnegative requested-delay range in µs.

The sequence is optional common reset, simultaneous first x90 pulses, common
delay, simultaneous phase-ramped final x90 pulses, then mux readout.
The delay starts after the longest first branch. Pre/post delays stay configured.
With zero pulse delays, a shorter pulse waits an extra `max_len - len_i`.
This can change fringe amplitude and constant phase.
Analysis uses the common requested delay. Before rounding, the final pulse
adds nominal detune × delay cycles to the source phase. Native timing can differ
by a few ticks.

**Fit fringe** selects a damped cosine. Turn it off for exponential decay.
A fringe fit needs six finite points and decay needs four. **Fit phase offset
(fringe only)** lets the fringe fit estimate a constant phase offset. If off,
the fit uses the nearest-zero 0/180-degree policy. This switch does not affect
an exponential decay fit.
T2* and error are in µs; fringe rate and error are in MHz.
Each tone returns its index, fit values, selected model and quality, including
fringe sampling quality when applicable.
Analysis plots retain the delay coordinates, measured values and fitted curves.
The returned summary does not include these arrays.

Read **Applied detune** in the Analyze summary. It is the per-tone phase-ramp
slope in MHz that Run estimates from its quantized delays and phases. It keeps
the native phase-wrap convention, including for signed detunes. It can differ
from the Form's nominal **Applied phase-ramp detune**. The fit figure reports
**Fitted fringe rate**, the oscillation frequency estimated from the IQ projection.

To check sampling, Analyze multiplies the largest gap between acquired delays
in µs by the larger magnitude of Fitted fringe rate and Applied detune in MHz.
The reciprocal gives samples per cycle. If the Run or an old file has no finite
Applied detune, Analyze uses the fitted rate alone, not the current Form value.
A nonzero nominal detune can yield a nonfinite Applied detune if no rounded
delay is positive. Adequate sampling requires at least ten samples per cycle
and a full cycle in the scan window. Fewer samples per cycle are undersampled.
A shorter window, a zero rate for this check or no delay gap makes the check
unavailable. Decay fits have no fringe sampling quality. Adequate sampling at
these rates does not rule out unknown higher-frequency aliases.

## Multi-tone readout control

Changing Participants changes both the active drives and the emitted readout
tones. A Q1-only Run versus Q1/Q2 therefore does not isolate the effect of Q2
drive. A second readout tone can contaminate raw IQ and produce apparent decay
or fringes before analysis.

For a readout control, keep Participants, all drive pulses, ADC channels and
timing unchanged. Set only the second readout tone gain to zero, keeping its
ADC declared. Compare raw IQ and magnitude with the original Run.
Changing Participants to Q1-only is a separate reference, not this control.
If testing Q2 drive separately, zero its drive gains without changing pulse
lengths, then change only RO2 gain in the next comparison.

Do not apply fitted frequency or T2 directly from a contaminated trace, even
when a fit converges or a candidate starts selected. Inspect raw IQ, sampling,
residuals and uncertainty, then review each candidate before Apply.
These controls do not change the readout phase, integration window or simulation
policy. In normal hardware use, avoid measuring coupled qubits simultaneously
when choosing a mux group. The offline diagnostic deliberately measured a
coupled Q1/Q2 pair; it does not establish that every mux group has this distortion.

In the offline control, Q2 drive gains were already zero. Disabling only RO2
raised the Q1 fringe fit r2 from 0.632 to 0.953.
That identifies readout contamination in that configuration, not a calibrated
change in T2. The scan had only 16 points over 3.75 µs.
Use mux Ramsey mainly for decay/T2 trends, not precise detuning.
Prefer reviewing decay candidates; treat frequency candidates conservatively.
Adequate sampling and automatic initial selection do not prove the absence
of cross-tone contamination.

## Writeback

Initial T2/sample selection requires positive finite T2, r2 at least 0.9,
normalized residual RMS at most 0.1 and relative T2 error at most 0.5.
Frequency candidates also require adequate sampling, at least 10 points per full
fringe cycle, and relative frequency error at most 0.5. Zero fitted frequency has
no relative error and is not initially selected. Mux Ramsey primarily shows the
decay/T2 trend; a frequency candidate is not precision-detuning evidence.
Failed or unavailable checks leave candidates unselected with explanatory notes.
You may select them manually after reviewing the source and fit.

Each qualifying tone proposes `<qubit>.t2r` in µs and a T2R sample at captured bias.
A frequency candidate additionally requires source configuration, a fringe fit,
finite source carrier, finite Applied detune and finite Fitted fringe rate.
Read the π/2 pulse source frequency and **Applied detune** from the same Run's
Analyze summary. Read **Fitted fringe rate** from that Run's fit figure. The
proposed frequency is source frequency + Applied detune − Fitted fringe rate,
in MHz. If the fitted rate is 0.002 MHz above Applied detune, the candidate is
0.002 MHz below the source frequency. If it is 0.002 MHz below Applied detune,
the candidate is 0.002 MHz above the source frequency. Do not use this
correction for decay-only fits, missing Applied detune or a fringe with an
ambiguous sign. Check the scan range for phase wrap. Review the sign convention,
fit uncertainty and captured destination before applying.

Frequency writeback changes the qubit calibration, not the existing x90/x180
library pulses. Recalibrate the pulses at the new frequency, for example with
Rabi or zigzag, then save and select new numeric modules with the recalibrated
gains and lengths. Changing only frequency does not calibrate the pulses.
Calibration writeback, New from a template and `ml_edit` can create modules.
Template expressions read calibration at creation, not on later Runs.
Choose Defaults in downstream tabs after changing module selections. Historical Run/Load
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
Acquisition errors propagate, and that run publishes no new record.
A stop request does not determine whether returned data is usable for analysis.
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
Defaults captures current parameters and rebuilds initial values.

Recipe writes and override paths use dotted strings, for example
`"tones.0.drive.gain"`. The decimal index addresses the complete tone list.
Omit a Form input to keep its value. Use `null` to clear a field that can be
left blank. `participants` still requires a nonempty list of unique, in-range
tone indices. `null` does not select all tones.
The recipe applies overrides last, then calls `check_ready()` before Run.

Pass `fit_fringe` and `fit_phase` as booleans. Omit either to keep its current
Analyze setting. Analyze rejects explicit `null`; use `false` to turn a switch off.

See the [mux recipe guide](../../../recipes/mux-guide.md#mux_t2ramsey) for dotted-path overrides,
parameter origins and explicit writeback decisions.

The MCP recipe is `mux_t2ramsey`.
