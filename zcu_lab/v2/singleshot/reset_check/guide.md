## Behavior

**Readers:** operator, agent. How to compare single-shot reset branches from the GUI or MCP and interpret the results.

Hardware gain and three-branch sweep: Rabi / Rabi + reset / Rabi + reset + Rabi. Both pulses share the swept gain. Saves classified G/E/Other populations; live view and analysis show Ground / Excited / Other with color for state and solid / dashed / dotted lines for branch.

## Saved populations

The saved `signals` array has axes `(gain, reset branch, population state)`. The final axis contains the measured fractions `[g,e,other]`, with `population_states=[0,1,2]`. Other counts rejected shots, including ties, rather than a reconstructed remainder. Unmeasured points remain NaN.

Older files saved only two population columns under the `GE Population` axis label. The new loader rejects that label because it expects `Population`. Those files cannot enter the new Analyze directly. Keep the old analysis version for those files; this version does not reconstruct Other or convert old results. The cfg schema remains 3.0 because the configuration fields did not change.

## Parameter sources

Open and Reset take a snapshot of the current aliases, parameters and module choices. Relax delay is 5 * `$qubit.t1`; the gain sweep ends at 4 * `$qubit.module.x180.gain`.

Classification uses the readout module's circle classifier, visible under Readout → Advanced (per tone for Mux). Run requires a classifier on every declared ADC and a positive radius; missing radius prompts singleshot GE Post. Linear has no Other state and is rejected. Analyze corrects populations only with the saved readout cfg's confusion matrix. Historical cfg without a classifier or matrix skips correction.

Pulse templates use `$qubit.freq`, `$qubit.wiring.ch`, `$resonator.freq`, `$resonator.wiring.ch` and `$resonator.wiring.ro_ch`. The readout trigger uses `$resonator.time_of_fly` + 0.05 us. Missing required sources leave the corresponding fields Invalid; supply them before Run.

T1-derived defaults use the Open/Reset snapshot. A missing or null T1, or an unavailable snapshot, uses 30 us, so the 5*T1 wait starts at 150 us. An invalid stored T1 leaves the required derived fields empty when the form opens. Alias, parameter-path or source errors are reported to the operator.

These starting values remain editable and separate from calibration writeback. Active reset keeps the same default wait; adjust it explicitly if needed.

## Module slots

Rabi uses `$qubit.module.x180` or pi_pulse. Tested reset uses `$qubit.module.reset` or reset template. Upstream reset is optional; readout uses `$resonator.module.readout` or readout template. Pulse values remain editable, including those replaced by the sweep. You can select module slots by key. Run uses the Open/Reset snapshot.

## Typical writeback

This is a diagnostic only; calibration parameters stay unchanged. It reports the reset-only mean and worst sampled excited population and the maximum Other fraction. Saved data contain classified populations, so readout refitting requires a separate raw-IQ acquisition. Interpret these summaries as sampled populations rather than histogram confidence intervals.

## Recommended

Reps are shots per gain per branch per round; rounds average repeated hardware sweeps and update the live plot. Branches are independent preparations. Other counts shots outside the accepted circle regions or tied between them. Calibrated leakage and reset-channel fidelity require separate characterization.

**Last updated:** 2026-10-29
