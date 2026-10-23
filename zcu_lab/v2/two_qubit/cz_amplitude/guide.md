# CZ common-amplitude calibration

**Readers:** operator, agent. Run common-amplitude CZ calibration through the GUI or MCP and review the first-return pulse proposals.

**Last updated:** 2026-10-21

Readout references `$mux.module.readout` when present, preserving the library's complete tones and scalar settings. Without a slot, fill every required readout field in the blank mux template.

Scan a common scale on two calibrated charge-line gains. Use the first complete
joint return cycle to propose two CZ pulses. FIT creates proposals;
Apply Selected writes accepted rows to the library.

After changing the parameter source, use Reset to recapture initial values. Refresh only updates sources and choices while preserving authored inputs; Reset can overwrite unsaved inputs.

## Preparation and inputs

Bind `f1` and `f2` to distinct fluxonium components, `coupler` to their transmon,
and `mux` to a MuxGroup containing both qubits. Open or Reset captures aliases,
parameters and library once. After changing bindings or calibrations, Reset and
check the whole Form again. Run uses the fixed roles.

Supply calibrated `$f1.module.x180`, `$f2.module.x180`, a complete mux readout,
and two GE center pairs compatible with that readout route and integration
window. Missing or coincident centers are invalid. F1/F2 tones come from their
positions in the fixed `$mux.qubits` source. Use those positions for role identity.
Keep every tone in the original order.

The two ordinary CZ pulses start from `$coupler.module.cz_f1` and `cz_f2` when
available. Without those slots, enter frequency, signed gain, Nyquist zone and
cosine duration explicitly. Generator fields come from captured role wiring.
Both pulses must have matching frequency, duration
and pre-delay on independent generators. Their post-delays can differ; the
sequence waits for both. Only the two charge lines are driven; the coupler stays
DC-biased. Supply a calibrated two-line cfg, either from constructive-phase
calibration or from another verified measurement.

## Run and inspect

The sequence is `B11 → simultaneous CZ → Xπ⊗Xπ → mux`. `B11` applies both Xπ
pulses to the thermal initial population. All thermal shots remain in the
ensemble and proceed directly to readout after the contrast pulses.

Enter a nonnegative, strictly increasing common gain-scale range. At each
coordinate the scan multiplies both original signed gains by that scale.
Frequency, waveform, duration, relative phase and signed gain ratio stay fixed.
Set Reps, Rounds and relaxation time for the intended acquisition budget.

The live figure shows completed shots. FIT displays the same-shot joint
00/01/10/11 probabilities and each original tone's mean I/Q versus scale. GE
classification uses the nearest saved center, retaining every shot. Equidistant
shots go to G. Every shot and its original readout effects remain in the analysis.

The joint 00 return after the contrast pulses is a **leakage proxy**. Thermal
population, preparation and readout errors remain in it. Two-class GE readout
leaves higher levels unresolved. Physical gate fidelity and π conditional phase
require separate measurements, including when a high return is observed.

## Fit the first complete return

FIT uses the same-shot joint 00 probability and the fixed-zero-phase model
`offset + contrast cos(2π scale/period)`. Phase is fixed at zero. The candidate
common scale is the measured first positive return, `period`.

Measure scale=0 and continue through at least one complete cycle. The fit uses
the contiguous prefix of complete scales, ending at the first incomplete scale.
A candidate must lie inside that contiguous measured prefix. A cancelled suffix
after a complete first cycle still allows analysis of the measured cycle.
Use enough scale points to resolve the oscillation; sparse sampling can alias a faster oscillation.

The figure overlays the fixed-phase fit and labels the candidate or failure
reason. Summary includes offset, contrast, period, period standard error and
`fit_quality`. Numeric failure, zero contrast, a degenerate model or nonfinite
uncertainty stops the calibration proposal. Incomplete coverage also stops the
proposal. Keep the raw curves and the reported reason as the failed fit result.

Uncertainty and residuals are diagnostics for operator review. A finite,
identifiable fit remains eligible even with low nonzero contrast.
Inspect the curves and uncertainty before accepting a proposal. Thermal mixtures and large-gain
AC Stark shifts can violate the single-frequency, fixed-phase model; a finite
fit still needs comparison with the measured device response.

## Preview and selected Apply

A valid measured first cycle proposes four independent rows:

| Target | Proposal |
| --- | --- |
| `$coupler.pulses.cz_f1` | Source F1 ordinary pulse with gain multiplied by the fitted scale. |
| `$coupler.module.cz_f1` | Concrete key for that F1 pulse in the Run/Load capture. |
| `$coupler.pulses.cz_f2` | Source F2 ordinary pulse with gain multiplied by the same fitted scale. |
| `$coupler.module.cz_f2` | Concrete key for that F2 pulse in the same capture. |

Preview each row. Check the destination component, generator, signed gain,
frequency, relative phase and waveform. Only the gains change. The source is
this run's saved cfg. Missing saved cfg prevents analysis;
a nonempty proposal also requires its matching captured role identities.

Select only the rows you accept, then press Apply Selected. Module rows and
slot rows apply independently. Apply each required slot explicitly and Reset
other already-open forms to capture the accepted calibration.
Check the complete cfg before the next Run.

Alternate amplitude and conditional-phase/frequency manually, preferably for
three rounds. Reset the next Form to capture accepted pulses. After changing
pulse shape, duration, relative phase, gain, frequency or bias, recheck CZ and
remeasure both local-Z corrections. Start each calibration step manually.

## Save, load and partial data

Save raw IQ, source cfg, numeric fit summary and named figures. Native restore
retains raw and cfg so the same fit can be reproduced offline. Raw data keeps
all tones and same-shot indices; rounds occupy fixed slots and are concatenated
without averaging. Native Load restores that run without changing current Form
inputs. Analysis and proposals use only saved cfg. Missing saved cfg stops
classification with a source error.

Cancellation retains completed raw rounds. A scale contributes to classification
only when all requested rounds are complete; other scales stay visibly missing.
A nonfinite acquired IQ or partially missing round is an error. Whole unwritten
rounds use NaN sentinels to mark missing measurements. Execution errors must be
resolved separately from interpreting a completed low-contrast scan.

Validate the actual FTF control chain with experimental data, separately from
offline synthetic checks.
