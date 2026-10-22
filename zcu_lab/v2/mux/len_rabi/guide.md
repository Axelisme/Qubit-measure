# Mux length Rabi

**Readers:** operator, agent. Run mux length Rabi from the GUI or MCP and read its results.

**Last updated:** 2026-10-22

## GUI setup

Create a `kind: group/mux` component and bind scalar alias `mux` to it, then
open `mux/len_rabi` from **New Tab**. Without this group and alias, opening the
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

A two-participant offline Run used 8 requested lengths from 0.02 to 0.16 us,
const pulses at gain 0.1, 100 reps, 1 round and 0.1 us recovery.
The recorded Run action took 2.006 s, about 0.25 s per joint sweep point.
These short, uncalibrated inputs demonstrate execution, not a pi-length calibration.
Keep the hardware-rounded coordinates when reviewing the result.

The timer includes the ComponentDriver button action, terminal wait and
screenshots. It is not acquisition-only time and excludes Analyze and Save.
Do not compare it with timings from other slots or experiments as a speed test.
Sweeping longer pulses changes the simulated evolution as well as hardware
playback. Point count, waveform, participants, reset, recovery and readout
windows also affect cost. Hardware repeats work for reps and rounds.
FTF compilation and cache reuse can outweigh that repetition, so do not assume
linear scaling with reps. Time a short Run at the intended pulse lengths first.

## Measurement and analysis

Use const or flat_top drives for every tone, including nonparticipants.
For flat_top, total length includes both ramps and must exceed their duration.
Switch an arb source to a supported waveform explicitly before Run.

Each tone supplies positive lengths in us. All tones use the same point count.
At each point, the selected drives use their corresponding lengths together.
The sequence is common reset, simultaneous drives, then mux readout after that
point's longest drive. Short drives wait for that branch; pulse pre/post delays
stay as configured. Hardware rounding can change total playback by a few ticks.
Nonpositive or overlapping adjacent rounded lengths block Run.

Each tone uses its own rounded requested-length axis and needs six finite IQ points.
Decay defaults to a damped cosine and can be disabled.
Fit phase defaults to fixed 0 degrees with signed amplitude. Enable it for a free
phase. Lengths and one-sigma errors are in us; Rabi frequency and error are in MHz.
Review the curve and fit quality, especially for small-truncation simulations.

## Writeback

Each qualifying tone proposes `<qubit>.ext.rabi_freq` in MHz.
With source configuration, candidates copy the measured const/flat_top pulse and
set fitted pi/pi2 lengths at `<qubit>.pulses.pi_amp` and
`<qubit>.pulses.pi2_amp`. Separate candidates set `module.x180` and
`module.x90` to those keys.
The `ext.pi_product` candidate is source gain times fitted pi length.
Review the source shape, flat_top ramps and each pulse/slot destination.

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
Cancellation retains acquired data and leaves missing data as NaN.
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
See the [mux recipe guide](../../../recipes/mux-guide.md#mux_len_rabi) for typed overrides,
parameter origins and explicit writeback decisions.

The MCP recipe is `mux_len_rabi`.
