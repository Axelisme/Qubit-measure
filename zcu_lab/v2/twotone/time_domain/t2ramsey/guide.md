**Readers:** operator, agent. Measure free-evolution coherence and use a resolved Ramsey fringe to check the qubit frequency.

## What Ramsey measures

Optional **Reset** → calibrated **π/2 pulse** → swept **Free-evolution delay** → second π/2 with a phase ramp → calibrated **Readout**. Both pulses use the same RF carrier. The fitted envelope gives T₂*; the fringe frequency measures the magnitude of the accumulated phase rate relative to that carrier and ramp.

## Key settings

- Confirm the π/2 pulse carrier and readout timing before Run. **Free-evolution delay** starts at `0…1.5 × qubit.t2r` in µs; fill **Stop** if T₂* is not stored. Span about **3 × T₂***. A shorter window shows only partial decay and forces extrapolation.
- For a fringe fit, take **at least 10 points per fringe**: at `f` MHz, set **Step** ≤ `1/(10f)` µs. Increase **Expts** when extending the window; sparse points can hide cycles or show an aliased, slower fringe.
- **Detune ratio (fringes/step)** sets the second pulse phase ramp. Positive advances phase, negative reverses it; requested ramp in MHz = ratio ÷ Step (µs). Use a known positive ramp large enough to stay on the positive-frequency branch when calibrating frequency.
- **Fit fringe** and **Fit phase offset (fringe only)** start on. Turning the latter off fixes the phase to 0° or 180° based on the shortest-delay IQ. On resonance, set the ratio to zero and turn off **Fit fringe** for a decay-only measurement.
- **Relax delay** starts at `5 × qubit.t1` (150 µs if T₁ is absent). In the acquisition budget, **Reps** is per delay per round; **Rounds** limits complete passes. See [`zcu_lab/v2/operator-workflow.md`](../../../operator-workflow.md) for the shared GUI steps.

## Read the plots

Live and Analyze plot the principal-axis projection of mean IQ (a.u.; I if the axis is undefined), not population, against delay in µs. The live title counts retained shots per point. Each bar is one **shot SEM**: retained-shot sample standard deviation along that axis divided by √(retained shot count). It is not shot spread, round-to-round scatter or a fit error. Points without shot statistics or with fewer than two shots have no bar; mean-only files can still load and fit.

Analyze overlays the fitted model and reports T₂* and (for a fringe fit) fringe frequency with fit standard errors. R² compares residuals with measured variation; it is not an acceptance threshold. Check that the fit follows the data across the window and that its uncertainty supports the intended correction. **Applied detune** is the same Run's realized phase ramp after rounding, not the fitted fringe frequency. **Reps × Rounds** is the configured budget.

## Next step

Writeback proposes `qubit.t2r` and, when source carrier and applied detune are available for a fringe fit, `qubit.freq`; it does not retune a saved pulse.

For a resolved positive-branch fringe, use the π/2 **Frequency** from the same run: `qubit.freq = source carrier + Applied detune − fringe frequency` (MHz). For example, 5800.000 + 0.400 − 0.402 = 5799.998 MHz. If the branch is unclear, repeat with a different known positive ramp and verify that the fringe shifts in the same direction. Recalibrate the pulse at the new carrier. A reopened native file lacks applied detune and does not offer frequency correction; do not substitute the current form's detune.

For `t2ramsey`, null disables **Reset**, but null **Readout** or **π/2 pulse** fails during selection, before overrides. Null delay Stop/Expts, Detune ratio, Reps or Rounds leaves a blank; required blanks left after overrides fail readiness before Run. Null for **Fit fringe** or **Fit phase offset (fringe only)** is not a Boolean and fails at Analyze after acquisition; omit the input to keep its value.
