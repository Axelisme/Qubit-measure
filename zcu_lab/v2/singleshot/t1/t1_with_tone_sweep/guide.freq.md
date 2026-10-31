## Behavior

**Readers:** operator, agent. How to map single-shot T1 transition rates versus probe frequency from the GUI or MCP and interpret the results.

Single-shot T1-with-tone, probe-frequency sweep: at each probe frequency it sweeps the wait-plus-probe length from both |g> and |e> initial states (Branch), classifies each shot during acquisition against the |g>/|e> centres, fits the dual transition rates per frequency, and renders the 3×3 rate landscape over (freq, time). Runs on real hardware.

## Saved counts

The saved `counts` array has axes `(frequency, initial state, length, population state)`. The final axis contains class counts `[g,e,other]` summed over acquired rounds, with `population_states=[0,1,2]`. Other counts rejected shots, including ties, rather than a reconstructed remainder. Unmeasured points remain NaN.

Live plots and Analyze divide each row by its total count before displaying populations or applying confusion correction. A completed row sums to reps × rounds. Older files without the `counts` variable fail to load with a missing `counts` error. No conversion is provided. The cfg schema remains 3.0 because the configuration fields did not change.

## Parameter sources

Open and Reset take a snapshot of the current aliases, parameters and module choices. Relax delay starts at 5*T1; length stop uses 5 times the tone T1 time scale. Frequency form sweeps `$resonator.freq` +/- 1.5 * `$resonator.kappa`; gain form sweeps 0 to 1. Each form exposes only one outer axis.

Classification uses the readout module's circle classifier, visible under Readout → Advanced (per tone for Mux). Run requires a circle classifier with a positive radius on the selected readout (the first tone for Mux); missing radius prompts singleshot GE Post. Linear has no Other state and is rejected. Analyze corrects populations only with the saved readout cfg's confusion matrix. Historical cfg without a classifier or matrix skips correction.

Pulse templates use `$qubit.freq`, `$qubit.wiring.ch`, `$resonator.freq`, `$resonator.wiring.ch` and `$resonator.wiring.ro_ch`. The readout trigger uses `$resonator.time_of_fly` + 0.05 us. Missing required sources leave the corresponding fields Invalid; supply them before Run.

The tone time scale uses the stored `$qubit.ext.t1_with_tone`, then ordinary T1, then 30 us. Missing or null sources advance to the next choice. An invalid stored tone T1 leaves the required stop empty for correction. Natural relaxation always uses ordinary T1.

T1-derived defaults use the Open/Reset snapshot. A missing or null T1, or an unavailable snapshot, uses 30 us, so the 5*T1 wait starts at 150 us. An invalid stored T1 leaves the required derived fields empty when the form opens. Alias, parameter-path or source errors are reported to the operator.

These starting values remain editable and separate from calibration writeback. Active reset keeps the same default wait; adjust it explicitly if needed.

## Module slots

Pi pulse uses `$qubit.module.x180` or pi_pulse. Probe starts from res_probe. Reset is optional; readout uses `$resonator.module.readout` or readout template. Pulse values remain editable, including those replaced by the sweep. You can select module slots by key. Run uses the Open/Reset snapshot.

## Typical writeback

This is a diagnostic only. Read the rate landscape from the grid; calibration parameters stay unchanged.

## Recommended

Run singleshot/ge Primary and Post, then Apply to the selected readout module. Keep the frequency sweep coarse (the inner length sweep multiplies the run time). 'sampling="linear"' (default) uses a linear length sweep; set 'sampling="decay"' to cluster points along the expected exponential decay while preserving the configured window and count. Configure the time window with `delay: SweepCfg`; explicit delay lists are not accepted.

**Last updated:** 2026-10-29
