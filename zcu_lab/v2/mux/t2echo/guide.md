# Mux Echo

**Readers:** operator, agent. Run mux Echo from the GUI or MCP and read its results.

**Last updated:** 2026-10-21

## GUI setup

Bind `$mux` to a MuxGroup, then open `mux/t2echo` from **New Tab**.
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

Each tone needs a fixed x90 drive and x180 pi pulse on the same generator.
The two pulses may have different waveforms, carriers, gains and phases.
Validate both pulses for every tone, including nonparticipants.
Use finite, nonnegative bounds for the shared total-delay range in us.
Each tone has its own detune in MHz.

The sequence is common reset, simultaneous x90 pulses, half the requested delay,
simultaneous x180 pulses, the other half-delay, simultaneous final x90 pulses,
then mux readout. Each stage starts together and the next stage waits for its
longest branch. Pre/post delays stay as configured.
Shorter pulses add fixed waits that can change amplitude, phase and echo-interval
balance. Analysis uses the common requested total delay.
The final x90 phase includes each tone's detune times that total delay.
Native timing can differ by a few ticks.

Fit method defaults to decay; select fringe explicitly for an oscillatory fit.
The original first point is omitted before finite-IQ filtering.
A fringe fit needs six remaining points; decay needs four.
Fringe phase is fixed at zero unless Fit phase is enabled.
Decay uses the same model regardless of Fit phase.
T2E and error are in us; fringe rate and error are in MHz.
Review fit quality and pulse calibration before accepting the estimate.

## Writeback

Each qualifying tone proposes its captured qubit's `t2e` in us and a T2E sample
at captured bias. Echo candidates update coherence only.
An unavailable actual detune still allows an otherwise usable T2E fit.
Review the model, uncertainty and captured destination before applying.

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
See the [mux recipe guide](../../../recipes/mux-guide.md#mux_t2echo) for typed overrides,
parameter origins and explicit writeback decisions.

The MCP recipe is `mux_t2echo`.
