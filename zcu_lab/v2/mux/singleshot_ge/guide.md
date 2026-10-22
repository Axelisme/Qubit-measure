# Mux singleshot GE

**Readers:** operator, agent. Run simultaneous mux GE readout from the GUI or MCP and read its results.

**Last updated:** 2026-10-22

## GUI setup

Create a `kind: group/mux` component and bind scalar alias `mux` to it, then
open `mux/singleshot_ge` from **New Tab**. Without this group and alias, opening the
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
Recovery delay is shared. Recovery initially uses five times
the largest captured T1, with the default for missing T1 values. A stored invalid
T1 leaves recovery blank for correction.

## FTF cost example

A two-participant offline Run used 300 shots per preparation, two preparations,
0.02 us const probe pulses and 0.1 us recovery. Reps and rounds stayed fixed
at 1. The recorded Run action took 6.118 s for the pair, about 3.06 s per
preparation. There is no frequency or delay sweep point here.
This preparation average is not per-shot acquisition latency.

The timer includes the ComponentDriver button action, terminal wait and
screenshots. It is not acquisition-only time and excludes Analyze and Save.
Do not compare it with timings from other slots or experiments as a speed test.
Shot count controls raw data volume and hardware repetitions. Preparation count,
init/probe lengths, participants, reset, recovery and readout windows affect cost.
FTF compilation and cache reuse can dominate the first Run, so do not extrapolate
linearly from shots or assume hardware reps/rounds scaling describes simulation.
Start with a small shot count and inspect both clouds before collecting more.
Both tone fits failed in this uncalibrated example with an initial-guess bounds
error. The raw clouds remain evidence of acquisition, not accepted fidelity.

## Measurement and analysis

Reps and rounds are fixed at one in the Form.
Shots sets the number of raw acquisitions in each probe-off/on preparation.
Each tone's optional init pulse runs in both preparations on its drive generator.

Each preparation applies common reset, then simultaneous branches containing init
followed by probe for the probe-on preparation, then mux readout after the longest
branch. Probe-off/on data keeps the same shot index across all active tones.
The two preparations are separate acquisitions.
These rows describe probe off/on, rather than guaranteed pure ground/excited states.
The IQ is normalized by each ADC integration length and retains individual shots.

Analyze fits each tone's probe-off/on clouds. Initial state describes the
predominant state after reset/init and before probe. It maps analysis rows to g/e
while preserving the stored data order.
Options include PCA/center fitting, angle in degrees, log histogram, T1 alignment
and a dimensionless length ratio.
Incomplete pairs and numerical fitting/classification failures retain a reason;
other tones continue. Plotting errors report the diagnostic failure.

Post uses primary results from the same selected Run/Load source.
Radius is automatic when empty, or a finite nonnegative value in normalized ADC
units. Consider other affects only automatic-radius selection.
Post shows a per-qubit 3x3 g/e/other diagnostic with an isolated-other row.
An infinite condition number or unresolved clusters makes it unsuitable for
calibration. Four-state joint assignment needs a separate calibration.

## Assignment and model quality

Fidelity assigns the two prepared labels using their best projected-IQ histogram
boundary, with equal preparation weights. Probe off/on need not be pure g/e
states. The score includes thermal mixing and pulse errors. A threshold can
exploit small within-cluster differences; it is not latent pure-state readout
fidelity.

Initial populations come from the histogram fit. Post-analysis uses nearest-center
regions and a radius, then solves the fitted population matrix to remove
preparation mixing. It does not use the primary histogram threshold. Low joint
r2 makes fitted populations and the corrected post matrix unreliable, even if
the two raw clusters look separate or the prepared-label score is high.
Inspect each tone's raw IQ, residuals and uncertainties before applying a
calibration. A numerical fit alone is not evidence that the model is adequate.

## Writeback

Primary candidates are initially selected only when joint r2 is at least 0.9,
normalized residual RMS at most 0.1 and relative width error at most 0.5.
Post radius/confusion also require a finite condition number at most 10.
Required missing/nonfinite metrics or their invalid issues leave candidates
unselected with notes. Unrelated fixed-zero parameter issues do not reject them.
You may select a candidate manually after inspecting the model and raw IQ.

A successful, finite primary row must pass the GE calibration validity check.
Its candidates update `ext.readout_cal.fidelity`, `sigma`, `g_center` and
`e_center` on the captured qubit.
Post candidates contain `radius` and the 3x3 `confusion_matrix`.
Review cluster separation, quality, confusion diagnostics and destination.

## Results and saved sources

Run draws one `measurement_i` IQ scatter per participant.
Blue and red shots distinguish the g (probe-off) and e (probe-on) preparations.
These preparations may contain state mixtures; they are not classified shots.
Analyze draws each participant's raw IQ clouds in `fit_i`.
Analyze uses the selected Run/Load source. Use the Analysis figure picker for
`fit_i` and the Post figure picker for `post_i`.
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
See the [mux recipe guide](../../../recipes/mux-guide.md#mux_singleshot_ge) for typed overrides,
parameter origins and explicit writeback decisions.

The MCP recipe is `mux_singleshot_ge`.

## Simulation limits

For an offline physical smoke, use `ftf(settings, workers=1)` with typed YAML
`readout.drive.kind: mux`, an ADC receiver list and independent drive lines.
Use replay archives for backend `ftf-segment-v7`; older backends are rejected.
Small truncations and short smoke runs demonstrate execution only.
Establish convergence and calibrate pulses before using fitted values.
All receiver IQ at a shot index must come from the same joint-state draw.
