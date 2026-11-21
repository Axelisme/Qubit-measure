## Behavior

**Readers:** operator, agent. How to compare single-shot reset branches from the GUI or MCP and interpret the results.

Hardware gain and three-branch sweep: Rabi / Rabi + reset / Rabi + reset + Rabi. Both pulses share the swept gain. Saves G/E/Other class counts; live view and analysis use one panel per branch and one color per outcome.

## Saved counts

The saved `counts` array has axes `(gain, reset branch, population state)`. The final axis contains class counts `[g,e,other]` summed over acquired rounds, with `population_states=[0,1,2]`. Other counts rejected shots, including ties, rather than a reconstructed remainder. Unmeasured points remain NaN.

Live plots and Analyze divide each row by its total count before displaying populations or applying confusion correction. A completed row sums to reps × rounds. Older files without the `counts` variable fail to load with a missing `counts` error. No conversion is provided. The current cfg schema is 4.0. Older files with native counts but without a compatible saved cfg retain the counts; Analyze can show classified fractions, but cannot use unavailable readout correction.

## Parameter sources

Open and Defaults take a snapshot of the current aliases, parameters and module choices. Relax delay is 5 * `$qubit.t1`; Gain → Stop starts at 4 * `$qubit.module.x180.gain` or 1.0, whichever is lower.

Readout accepts only the pulse family.
A mux readout in the library is incompatible.
Classification uses the readout's circle classifier under Readout → Advanced.
Run requires a positive radius; missing radius prompts singleshot GE Post.
Linear has no Other state and is rejected.
Analyze corrects populations only with the saved readout cfg's confusion matrix.
Historical cfg without a classifier or matrix skips correction.

Pulse templates use `$qubit.freq`, `$qubit.wiring.ch`, `$resonator.freq`, `$resonator.wiring.ch` and `$resonator.wiring.ro_ch`. The readout trigger uses `$resonator.time_of_fly` + 0.015 µs. Missing required sources leave the corresponding fields Invalid; supply them before Run.

T1-derived defaults use the Open/Defaults snapshot. A missing or null T1, or an unavailable snapshot, uses 30 µs, so the 5*T1 wait starts at 150 µs. An invalid stored T1 leaves the required derived fields empty when the form opens. Alias, parameter-path or source errors are reported to the operator.

These starting values remain editable and separate from calibration writeback. Active reset keeps the same default wait; adjust it explicitly if needed.

## Module slots

Rabi uses `$qubit.module.x180` or pi_pulse. Tested reset uses `$qubit.module.reset` or reset template. Upstream reset is optional; readout uses `$resonator.module.readout` or readout template. Pulse values remain editable, including those replaced by the sweep. You can select module slots by key. Run uses the Open/Defaults snapshot.

## Typical writeback

This is a diagnostic only; calibration parameters stay unchanged. It reports the reset-only mean and worst sampled excited population and the maximum Other fraction. Saved data contain class counts, so readout refitting requires a separate raw-IQ acquisition. Interpret these summaries as sampled populations rather than histogram confidence intervals.

The returned analysis summary contains `reset_mean_excited_population`, `reset_max_excited_population`, `reset_max_other_population`, `worst_sample_gain` and `analyzed_reset_points`. Corrected population arrays stay local to analysis for the three-branch plot.

## Recommended

Reps are shots per gain per branch per round; rounds average repeated hardware sweeps and update the live plot. Branches are independent preparations. Other counts shots outside the accepted circle regions or tied between them. Calibrated leakage and reset-channel fidelity require separate characterization.

## Reading the figures

Compare P(g) and P(Other) at the same gain across the three branch panels. The Reset only panel reports mean and maximum P(e) across the scan, not a new calibration. Read the branch contrast at the intended π gain when judging the tested reset. If the prepared e population is low without reset, adjust the π pulse first. Analyze draws shot SEM from each measured branch's retained class counts, with the covariance propagated through saved readout correction when available.

**Last updated:** 2026-11-22
