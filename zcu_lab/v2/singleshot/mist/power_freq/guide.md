## Behavior

**Readers:** operator, agent. How to map MIST populations versus probe gain and frequency from the GUI or MCP and interpret the results.

MIST probe power × frequency landscape: drives a probe pulse while sweeping its gain and frequency (2D), classifying each shot during acquisition against the |g>/|e> single-shot centres, and plotting the ground/excited/other populations as 2D maps over (gain, freq). Runs on real hardware. Inspect the acquired populations directly.

## Parameter sources

Open and Reset take a snapshot of the current aliases, parameters and module choices. Relax delay is 5 * `$qubit.t1`. Probe frequency is `$resonator.freq` +/- 1.5 * `$resonator.kappa`; gain sweep is 0 to 1.

Classification uses the readout module's circle classifier, visible under Readout → Advanced (per tone for Mux). Run requires a classifier on every declared ADC and a positive radius; missing radius prompts singleshot GE Post. Linear has no Other state and is rejected. Analyze corrects populations only with the saved readout cfg's confusion matrix. Historical cfg without a classifier or matrix skips correction.

Pulse templates use `$qubit.freq`, `$qubit.wiring.ch`, `$resonator.freq`, `$resonator.wiring.ch` and `$resonator.wiring.ro_ch`. The readout trigger uses `$resonator.time_of_fly` + 0.05 us. Missing required sources leave the corresponding fields Invalid; supply them before Run.

T1-derived defaults use the Open/Reset snapshot. A missing or null T1, or an unavailable snapshot, uses 30 us, so the 5*T1 wait starts at 150 us. An invalid stored T1 leaves the required derived fields empty when the form opens. Alias, parameter-path or source errors are reported to the operator.

These starting values remain editable and separate from calibration writeback. Active reset keeps the same default wait; adjust it explicitly if needed.

## Module slots

Probe starts from res_probe. Optional init/reset use `$qubit.module.x180`/reset. Readout uses `$resonator.module.readout` or readout template. Pulse values remain editable, including those replaced by the sweep. You can select module slots by key. Run uses the Open/Reset snapshot.

## Typical writeback

This is a diagnostic only. Read the population landscapes from the plot; calibration parameters stay unchanged.

## Recommended

Run singleshot/ge Primary and Post, then Apply to the selected readout module. Sweep the probe gain across the MIST onset and the frequency around the qubit/resonator line of interest.

## Saved counts

Save keeps `counts` shaped `(Ngain, Nfreq, 3)` in `[g,e,other]` order, with
`population_states=[0,1,2]`. Other counts shots outside the accepted circles
or tied between them. It is not a calibrated leakage population.
Each point sums to the shots acquired across completed rounds. Live plots and
Analyze divide each point by its total before displaying populations or applying
the saved confusion matrix. Sweep coordinates and units are unchanged. Cfg schema
remains 3.0.

Old files storing `signals` instead of `counts` are unsupported. Load reports
the missing `counts` variable. Acquire a new run; Load does not convert stored
fractions to counts.

**Last updated:** 2026-10-29
