# Mux amplitude Rabi

**Readers:** operator, agent. Run mux amplitude Rabi from the GUI or MCP and read its results.

**Last updated:** 2026-10-21

## GUI setup

Bind `$mux` to a MuxGroup, then open `mux/amp_rabi` from **New Tab**.
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

Each tone supplies a gain range in normalized units within `[-1, 1]`.
All tones, including nonparticipants, use the same point count.
At each point, the selected drives use their corresponding gain values together.
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

Run draws a combined `measurement` figure.
Each line uses its own hardware-rounded scan coordinates, even when ranges differ.
Legends show the original index and source name in Participants order.
Individual `measurement_i` figures remain available.
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
See the [mux recipe guide](../../../recipes/mux-guide.md) for typed overrides,
parameter origins and explicit writeback decisions.

The MCP recipe is `mux_amp_rabi`.
