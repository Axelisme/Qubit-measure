# Mux Echo

**Readers:** operator, agent. Run mux Echo from the GUI or MCP and read its results.

**Last updated:** 2026-10-22

## GUI setup

Create a `kind: group/mux` component and bind scalar alias `mux` to it, then
open `mux/t2echo` from **New Tab**. Without this group and alias, opening the
form reports an error.
The captured parameter source must contain every qubit and its matching resonator
in readout order. Readout references the module slot `$mux.module.readout` when
present, preserving the library's tones and scalar settings. Use a mux-family
module with the full group tone count. If the slot is empty, Readout starts
inline from the blank mux template with group calibration tones.
The generator comes from `$mux.wiring.gen_ch`, and tone i's ADC channel comes
from `$mux.wiring.ro_ch[i]`, in the group's resonator order. Each tone starts
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

A two-participant offline Run used 16 total delays from 0 to 3.75 us,
0.5 us const x90/x180 pulses, 300 reps and 1 round.
Q1 detune was 0 and Q2 detune was 0.5 MHz.
Recovery was 1007.509 us; readout and ADC integration were 2 and 1.5 us.
Recorded RPC Run wall time was 0.689 s, about 0.043 s per joint delay point.
It measures Run and completion waiting, not acquisition alone, and excludes
Analyze and Save. The GUI had already executed earlier Runs, so this is not
a cold benchmark.

Do not compare it with other slots or experiments as a speed test.
Echo has an extra x180 stage; pulse lengths, total delay window, point count,
participants, reset, recovery and readout windows affect cost.
Hardware repeats acquisitions for reps and rounds. FTF compilation and cache
reuse can dominate instead of scaling with reps. Time a short Run with the
intended delay window before enlarging the scan.
This example only validates execution and model selection.
The nonzero-detune fringe fit had enormous T2 uncertainty and was not calibration.

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

Fit method defaults to decay. Zero source detune forces decay, even if fringe
is selected. Select fringe for nonzero source detune to fit an oscillation;
explicit decay remains available. Analysis uses the saved cfg detune, or finite
saved actual detune when cfg is absent. A fringe request without either source
reports a missing-detune error rather than guessing.
The original first point is omitted before finite-IQ filtering.
A fringe fit needs six remaining points; decay needs four.
Fringe phase is fixed at zero unless Fit phase is enabled.
Decay uses the same model regardless of Fit phase.
T2E and error are in us; fringe rate and error are in MHz.
Review fit quality and pulse calibration before accepting the estimate.

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
reduced Q1's raw magnitude range from 0.952 to 0.310 ADC units.
The Q1-only reference was 0.302 ADC units, but its decay fit r2 was only 0.580.
The control points to readout contamination; it does not validate an absolute T2.
An Echo fringe rate is not a frequency-writeback calibration.
For the related mux Ramsey measurement, use decay/T2 trends rather than precise
detuning and treat frequency candidates conservatively.

## Writeback

Initial T2/sample selection requires positive finite T2, r2 at least 0.9,
normalized residual RMS at most 0.1 and relative T2 error at most 0.5.
Failed or unavailable checks leave candidates unselected with explanatory notes.
You may select them manually after reviewing the source and fit.

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
