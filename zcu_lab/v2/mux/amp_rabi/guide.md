# Mux amplitude Rabi

**Readers:** operator, agent. Run mux amplitude Rabi from the GUI or MCP and read its results.

**Last updated:** 2026-10-22

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
shared positive point count, including nonparticipants (default 51).
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

MCP field paths use original tone positions, for example
`("tones", 0, "drive", "gain")`. The integer selects the full tone list.
See the [mux recipe guide](../../../recipes/mux-guide.md#mux_amp_rabi) for typed overrides,
parameter origins and explicit writeback decisions.

The MCP recipe is `mux_amp_rabi`.
