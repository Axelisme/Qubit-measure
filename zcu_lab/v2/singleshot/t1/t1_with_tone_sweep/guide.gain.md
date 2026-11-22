## Behavior

**Readers:** operator, agent. How to map single-shot T1 transition rates versus probe gain from the GUI or MCP and interpret the results.

At each probe gain, Run sweeps the wait-plus-probe length from both |g> and |e> initial states. It classifies each shot against the |g>/|e> centers. Analyze fits the transition rates at each gain using both preparations. Runs on real hardware.

## Saved counts

The saved `counts` array has axes `(gain, initial state, length, population state)`. The final axis contains class counts `[g,e,other]` summed over acquired rounds, with `population_states=[0,1,2]`. Other counts rejected shots, including ties, rather than a reconstructed remainder. Unmeasured points remain NaN.

Live plots and Analyze divide each row by its total count before displaying populations or applying confusion correction. A completed row sums to reps × rounds. Older files without the `counts` variable fail to load with a missing `counts` error. No conversion is provided. The cfg schema is 4.0 after renaming the time window to delay and uniform to sampling.

## Parameter sources

Open and Defaults take a snapshot of the current aliases, parameters and module choices. Relax delay starts at 5*T1; delay stop uses 5 times the tone T1 time scale. Frequency form sweeps `$resonator.freq` +/- 1.5 * `$resonator.kappa`; gain form sweeps 0 to 1. Each form exposes only one outer axis.

Readout accepts only the pulse family.
A mux readout in the library is incompatible.
Classification uses the readout's circle classifier under Readout → Advanced.
Run requires a positive radius; missing radius prompts singleshot GE Post.
Linear has no Other state and is rejected.
Analyze corrects populations only with the saved readout cfg's confusion matrix.
Historical cfg without a classifier or matrix skips correction.

Pulse templates use `$qubit.freq`, `$qubit.wiring.ch`, `$resonator.freq`, `$resonator.wiring.ch` and `$resonator.wiring.ro_ch`. The readout trigger uses `$resonator.time_of_fly` + 0.015 µs. Missing required sources leave the corresponding fields Invalid; supply them before Run.

The tone time scale uses the stored `$qubit.ext.t1_with_tone`, then ordinary T1, then 30 µs. Missing or null sources advance to the next choice. An invalid stored tone T1 leaves the required stop empty for correction. Natural relaxation always uses ordinary T1.

T1-derived defaults use the Open/Defaults snapshot. A missing or null T1, or an unavailable snapshot, uses 30 µs, so the 5*T1 wait starts at 150 µs. An invalid stored T1 leaves the required derived fields empty when the form opens. Alias, parameter-path or source errors are reported to the operator.

These starting values remain editable and separate from calibration writeback. Active reset keeps the same default wait; adjust it explicitly if needed.

## Module slots

Pi pulse uses `$qubit.module.x180` or pi_pulse. Probe starts from res_probe. Reset is optional; readout uses `$resonator.module.readout` or readout template. Pulse values remain editable, including those replaced by the sweep. You can select module slots by key. Run uses the Open/Defaults snapshot.

## Read the plots

Run shows six population maps. The columns are the initial g and e states, and the rows show the final g, e and Other fractions. The x-axis is probe gain (a.u.) and the y-axis is probe duration (µs). The bottom row shows the three fractions versus probe duration at the current gain, one panel per preparation.

In Analyze, the first two rows each contain three population maps, ordered Ground, Excited and Other. The first row starts in g and the second starts in e. The y-axis is probe duration (µs); color shows the population. The x-axis is the same outer coordinate used in the rate panels. It is probe gain (a.u.) unless `ac_stark_coeff` is available. If that coefficient is supplied, Analyze plots the coefficient times the squared gain; the rate-panel label still reads Probe gain (a.u.).

The bottom three panels show transition rates in µs⁻¹ on logarithmic y-axes. From left to right they compare g→Other with g→e, e→Other with e→g, and e→Other with g→Other. Each line joins rates fitted separately at the outer sweep points. Bars at the first and every fifth point after it show each rate's own fit standard error, the square root of that rate's covariance diagonal entry. They are not shot SEM.

## Typical writeback

This is a diagnostic only. Analyze does not change calibration parameters.

## Recommended

Run singleshot/ge Primary and Post, then Apply to the selected readout module. Keep the gain sweep coarse (the inner length sweep multiplies the run time). Provide '`$qubit.ext.ac_stark_coeff`' for a photon-number x-axis. 'sampling="linear"' (default) uses a linear length sweep; set 'sampling="decay"' to cluster points along the expected exponential decay while preserving the configured window and count. Configure the time window with `delay: SweepCfg`; explicit delay lists are not accepted.

**Last updated:** 2026-10-29
