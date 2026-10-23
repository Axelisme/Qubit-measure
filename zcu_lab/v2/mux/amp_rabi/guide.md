# Mux amplitude Rabi

**Readers:** operator, agent. Run mux amplitude Rabi from the GUI or MCP and read its results.

**Last updated:** 2026-10-22

After changing the parameter source, use Reset to recapture initial values. Refresh only updates sources and choices while preserving authored inputs; Reset can overwrite unsaved inputs.

## GUI setup

Create a `kind: group/mux` component and bind scalar alias `mux` to it, then
open `mux/amp_rabi` from **New Tab**. Without this group and alias, opening the
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
Open/Reset fills selected tones' drive and per-tone sweep defaults and sets
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

A two-participant offline Run used 8 gain points from 0 to 0.3, 100 reps
and 1 round. Each drive was a 0.02 us const pulse, with 0.1 us recovery.
The recorded Run action took 2.105 s, about 0.26 s per joint sweep point.
This is an order-of-magnitude budget for that short smoke configuration,
not a calibrated Rabi measurement.

The timer includes the ComponentDriver button action, terminal wait and
screenshots. It is not acquisition-only time and excludes Analyze and Save.
Do not compare it with timings from other slots or experiments as a speed test.
Gain point count, pulse shape and duration, participant count, reset, recovery
and readout windows affect cost. Reps and rounds add repeated acquisitions on
hardware. FTF compilation and cached evolution can dominate, so changing reps
need not scale wall time proportionally. Time a short Run with your settings
before increasing point count or rounds.

## Measurement and analysis

Each tone supplies a gain range in normalized units within `[-1, 1]`.
Set each tone's Start and Stop in gain units. Expts follows Tones and sets one
shared positive point count, for selected tones (default 51).
One program advances the selected drives' gains together in a shared hardware loop.
Each drive retains its own Start and Stop. Plots and saved axes use the actual
hardware-quantized gains; integer register steps can shift points from a separately
rounded linear grid.
The sequence is common reset, simultaneous drives, then mux readout after the
longest drive. Pulse pre/post delays stay as configured, with early-finishing
drives waiting for the longest branch.

Analysis uses the undamped Rabi model and zero-gain phase policy.
Skip removes leading scan points. Each tone needs at least six finite IQ points
after that removal. Review fit quality as well as the numerical result.

## Writeback

Candidates copy the measured source pulse and set fitted pi/pi2 gains at
`<qubit>.pulses.pi_amp` and `<qubit>.pulses.pi2_amp`.
Separate items set module slots `module.x180` and `module.x90` to those keys.
The `ext.pi_product` candidate is fitted pi gain times the source pulse's scalar
length. Pulse proposals require the saved source configuration.
Inspect pulse shape and uncertainty before selecting pulse and slot updates.

## Results and saved sources

Run draws one `measurement_i` figure per participant, without a shared overlay.
Each figure uses that tone's hardware-rounded scan coordinates.
The index `i` is the original tone position in the saved configuration.
Analyze uses the selected Run/Load source. Use the Analysis figure picker for
`fit_i`.
Here `i` is the original tone position, so Q3/Q1 uses indices 2/0.
Summary rows retain that order and show values, quality or a failure reason.
A missing source configuration leaves the name unavailable.

Use Data to save measurement data and analysis artifacts.
Load Data selects a native file or saved run and preserves current Form inputs.
The selected source fixes the historical pulse, binding and writeback destination.
Live plots update the full sweep after each completed round. Stop retains the
completed-round average across all points. If no round completes, signals remain NaN.
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
See the [mux recipe guide](../../../recipes/mux-guide.md#mux_amp_rabi) for dotted-path overrides,
parameter origins and explicit writeback decisions.

The MCP recipe is `mux_amp_rabi`.
