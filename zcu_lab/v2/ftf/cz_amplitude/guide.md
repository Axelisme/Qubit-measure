# CZ common-amplitude calibration

Scan a common scale on two calibrated charge-line gains. Use the first complete
joint return cycle to propose two ordinary CZ pulses. FIT creates proposals;
it never writes the library.

## Preparation and inputs

Bind `f1` and `f2` to distinct fluxonium components, `coupler` to their transmon,
and `mux` to a MuxGroup containing both qubits. Open or Reset captures aliases,
parameters and library once. After changing bindings or calibrations, Reset and
check the whole Form again. Run never rebinds roles.

Supply calibrated `$f1.module.x180`, `$f2.module.x180`, a complete mux readout,
and two GE center pairs compatible with that readout route and integration
window. Missing or coincident centers are invalid. F1/F2 tones come from their
positions in captured `$mux.qubits`, not ADC numbers or tone labels. Keep every
tone in the original order.

The two ordinary CZ pulses start from `$coupler.module.cz_f1` and `cz_f2` when
available. Without those slots, enter frequency, signed gain, Nyquist zone and
cosine duration explicitly. Generator fields come from captured role wiring,
not hard-coded FTF defaults. Both pulses must have matching frequency, duration
and pre-delay on independent generators. Their post-delays can differ; the
sequence waits for both. No coupler AC line is used. This experiment does not
require the constructive-phase Form to have been run. Operators may provide a
calibrated two-line cfg directly.

## Run and inspect

The sequence is `B11 → simultaneous CZ → Xπ⊗Xπ → mux`. `B11` applies both Xπ
pulses to the thermal initial population. It does not prepare a pure state.
There is no heralding, shot exclusion or pre-readout reset.

Enter a nonnegative, strictly increasing common gain-scale range. At each
coordinate the runner multiplies both original signed gains by that scale.
Frequency, waveform, duration, relative phase and signed gain ratio stay fixed.
Set Reps, Rounds and relaxation time for the intended acquisition budget.

The live figure shows completed shots. FIT displays the same-shot joint
00/01/10/11 probabilities and each original tone's mean I/Q versus scale. GE
classification uses the nearest saved center, retaining every shot. Equidistant
shots go to G. It does not apply a radius cut or invert a confusion matrix.

The joint 00 return after the contrast pulses is a **leakage proxy**. Thermal
population, preparation and readout errors remain in it. Two-class GE readout
does not resolve higher levels. Neither a high return nor a population minimum
establishes physical gate fidelity or a π conditional phase.

## Fit the first complete return

FIT uses the same-shot joint 00 probability and the fixed-zero-phase model
`offset + contrast cos(2π scale/period)`. It does not fit a free phase. The
candidate common scale is the first positive return, `period`, never an endpoint
or the saved input amplitude.

Measure scale=0 and continue through at least one complete cycle. The fit uses
the contiguous prefix of complete scales, ending at the first incomplete scale.
It cannot bridge a missing interior point or extrapolate a candidate beyond
that measured prefix. A cancelled suffix after the complete first cycle does
not prevent analysis of the measured cycle. Use enough scale points to resolve
the oscillation; sparse sampling can alias a faster oscillation.

The figure overlays the fixed-phase fit and labels the candidate or failure
reason. Summary includes offset, contrast, period, period standard error and
`fit_quality`. Numeric failure, zero contrast, a degenerate model or nonfinite
uncertainty produces no calibration. Incomplete coverage also produces no
calibration. Keep the raw curves and the reported reason; do not replace a
failed result with zero or a default scale.

Uncertainty and residuals are diagnostics, not automatic statistical gates.
There is no χ², contrast/σ, frequency/σ or fidelity threshold. Low but nonzero
contrast alone does not reject a finite identifiable fit. Inspect the curves
and uncertainty before accepting a proposal. Thermal mixtures and large-gain
AC Stark shifts can violate the single-frequency, fixed-phase model; a finite
fit does not establish that the model describes the device.

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
this run's cfg, not today's library pulse. Missing saved cfg prevents analysis;
a nonempty proposal also requires its matching captured role identities.

Select only the rows you accept, then press Apply Selected. Module rows and
slot rows can be applied separately. This is not a cross-file transaction.
Applying a module does not update its slot or another already-open Form.
Check the complete cfg before the next Run.

Alternate amplitude and conditional-phase/frequency manually, preferably for
three rounds. Reset the next Form to capture accepted pulses. After changing
pulse shape, duration, relative phase, gain, frequency or bias, recheck CZ and
remeasure both local-Z corrections. No automatic calibration chain runs.

## Save, load and partial data

Save raw IQ, source cfg, numeric fit summary and named figures. Native restore
retains raw and cfg so the same fit can be reproduced offline. Raw data keeps
all tones and same-shot indices; rounds occupy fixed slots and are concatenated
without averaging. Native Load restores that run without changing current Form
inputs. Analysis and proposals use only saved cfg, never today's Form/library
to fill historical gaps. Missing saved cfg refuses classification.

Cancellation retains completed raw rounds. A scale contributes to classification
only when all requested rounds are complete; other scales stay visibly missing.
A nonfinite acquired IQ or partially missing round is an error. Whole unwritten
rounds use NaN sentinels and are not measurements. Execution errors are not
successful low-contrast scans.

Offline synthetic component validation is not an FTF simulator smoke. The
first-batch acceptance ticket checks the actual FTF control chain separately.
