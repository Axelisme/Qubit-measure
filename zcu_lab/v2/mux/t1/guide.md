# Mux T1

**Readers:** operator, agent. Run mux T1 from the GUI or MCP and read its results.

**Last updated:** 2026-10-23

After changing the parameter source, use Reset to recapture initial values. Refresh only updates sources and choices while preserving authored inputs; Reset can overwrite unsaved inputs.

## GUI setup

Create a `kind: group/mux` component and bind scalar alias `mux` to it, then
open `mux/t1` from **New Tab**. Without this group and alias, opening the
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
list of unique original indices, for example `[2, 0]`. This runs original
positions 2/0 in that order, even while the picker shows Q2. Empty, duplicate or
out-of-range participants block Run and remain editable for correction.

Selected tones need drives on independent generators, distinct from readout.
Open/Reset fills selected tones' drive defaults and sets
these fields to None on other tones. The nullable fields remain editable.
Open/Reset captures the Aliases row `participants`, for example `[Q3, Q1]`,
and maps it against the captured `$mux.qubits` order to original indices `[2, 0]`.
Alias list elements may use `$` links. A missing, unbound or empty row selects all
original indices in order. An unbound element in a nonempty list rejects Open/Reset.
An out-of-group name stays visible as an invalid Participants draft until corrected.
Changing aliases requires Reset or reopening; ordinary participant edits keep tone values.

Editing participants preserves every tone value. Fill any newly selected tone's
missing settings at the indicated tone path. Removing a tone keeps its values. Every readout
tone still needs valid frequency and ADC identity, even when not selected.

## FTF cost example

A two-participant offline Run used 11 equally spaced requested delays from
0.02383 to 0.25635 us, 0.02 us const drive pulses, 50 reps, 2 rounds and
0.1 us recovery. Readout used 0.1 us pulse and integration windows, with a
0.4 us trigger offset. The hardware-sweep Run took 1.152 s on CPUs 4 and 5 after
warming the simulation caches.
This window only checks execution. It does not constrain a physical T1.

The timer includes the ComponentDriver button action and terminal wait.
It is not acquisition-only time and excludes screenshots, Analyze and Save.
Do not compare it with timings from other slots or experiments as a speed test.
Long-delay T1 scans can cost much more than this short example because the
simulated evolution window changes. Delay point count and range, drive duration,
participants, reset, recovery and readout windows all affect cost.
Reps and rounds repeat acquisitions on hardware. FTF compilation and cached
evolution can dominate instead of scaling with reps. Time a short scan that
includes the intended longest delay before budgeting the full scan.

## Measurement and analysis

Use calibrated fixed pi pulses. All qubits share one delay axis in us,
executed by one program's hardware loop. Requested window endpoints and the
linear input axis must be finite, nonnegative and strictly increasing.

**Uniform (linear) sweep** defaults to true. Turn it off to concentrate delay
samples in the decay region within the same start/stop window, using the same
sampling policy as single-qubit T1. Every participant uses the same cycle table.
Nonuniform samples must remain distinct and increasing after hardware
quantization; otherwise Run reports an error before acquisition.

The sequence is common reset, simultaneous pi pulses, common delay, then mux
readout. The delay starts after the longest pulse branch, including pre/post delays.
With zero pulse delays, a shorter pulse waits an extra `max_len - len_i`.
That fixed wait changes an exponential decay's amplitude while preserving T1.
With uniform sampling, plots and fits retain requested delays; hardware timing
may differ by a few ticks. With nonuniform sampling, plots, fits and native
files retain the quantized delays. Choose a window that constrains both decay
and baseline.

Skip removes leading points before finite-IQ filtering.
Choose the single- or dual-exponential model and inspect quality and uncertainty.
T1 and errors are in us.

## Writeback

Each qualifying row proposes its captured qubit's `t1` in us with fitted error,
plus a T1 sample at the captured bias.
Review the model, uncertainty and captured destination before Apply.

## Results and saved sources

Run draws one `measurement_i` figure per participant, without a shared overlay.
Each figure uses the common recorded-delay axis.
The index `i` is the original tone position in the saved configuration.
Analyze uses the selected Run/Load source. Use the Analysis figure picker for
`fit_i`.
Here `i` is the original tone position, so Q3/Q1 uses indices 2/0.
Summary rows retain that order and show values, quality or a failure reason.
A missing source configuration leaves the name unavailable.

Use Data to save measurement data and analysis artifacts.
Load Data selects a native file or saved run and preserves current Form inputs.
The selected source fixes the historical pulse, binding and writeback destination.
Live plots update after each completed round across the full delay axis.
Cancellation retains completed-round averages across that axis.
Before the first completed round, signals remain NaN.
Partial data from a failed operation retains its failure status.
Only successful, fully finite rows qualify for calibration candidates.
Missing captured bindings suppress candidates; inconsistent bindings report an error.
Read each Writeback destination, value, quality and uncertainty.
Deselect unwanted items, then use **Apply Selected** to write the selected items.
Analyze and preview only calculate results. Each candidate needs its own selection.

## Drafts and MCP inputs

Saved drafts retain invalid field text and Participants for correction.
Restoration reports dropped fields and keeps the original parameter source.
Unreadable saved inputs produce a diagnostic and fresh defaults.
Reset captures current parameters and rebuilds initial values.

Recipe writes and override paths use dotted strings, for example
`"tones.0.drive.gain"`. The decimal index addresses the complete tone list.
Omission retains a Form input; null clears it. Analysis inputs also use
omission to retain settings and pass explicit null to their owner.
The recipe applies overrides last, then calls `check_ready()` before Run.
See the [mux recipe guide](../../../recipes/mux-guide.md#mux_t1) for dotted-path overrides,
parameter origins and explicit writeback decisions.

The MCP recipe is `mux_t1`. Set the generated Form field through an override
such as `{"path": "uniform", "value": false}`. The recipe applies overrides
before `check_ready()`; omission keeps the current value.

## Simulation limits

For an offline physical smoke, use `ftf(settings, workers=1)` with typed YAML
`readout.drive.kind: mux`, an ADC receiver list and independent drive lines.
Use replay archives for backend `ftf-segment-v7`; older backends are rejected.
Small truncations and short smoke runs demonstrate execution only.
Establish convergence and calibrate pulses before using fitted values.
