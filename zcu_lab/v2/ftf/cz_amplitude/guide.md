# CZ common-amplitude acquisition

This checkpoint provides the amplitude scan and saved-source classification
plots. FIT currently computes diagnostics only. It does not estimate a CZ
amplitude or produce calibration proposals.

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

## Calibration limits

This version does not fit a complete return cycle or select an amplitude.
It supplies no pulse-module or slot writeback rows. Do not use an endpoint,
a population minimum or the saved input amplitude as a fitted CZ calibration.
Keep the curves and source cfg for subsequent analysis.

## Save, load and partial data

Save raw IQ, source cfg, classification summary and named diagnostic figures. Raw data keeps
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
