# Mux Ramsey

**Readers:** operator, agent. Run mux Ramsey from the GUI or MCP and read its results.

**Last updated:** 2026-10-22

After changing the parameter source, use Reset to recapture initial values. Refresh only updates sources and choices while preserving authored inputs; Reset can overwrite unsaved inputs.

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

## FTF cost example

A two-participant offline Run used 16 delays from 0 to 3.75 us, 0.5 MHz
nominal detune on both tones, 0.5 us const x90 pulses, 300 reps and 1 round.
Recovery was 1007.509 us; readout and ADC integration were 2 and 1.5 us.
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
change in T2. The scan had only 16 points over 3.75 us.
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
