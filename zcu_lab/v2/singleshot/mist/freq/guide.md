## Behavior

**Readers:** operator, agent. How to measure MIST populations versus probe frequency from the GUI or MCP and interpret the results.

MIST probe-frequency sweep: drives a probe pulse whose frequency is swept and classifies each shot during acquisition against the |g>/|e> single-shot centres, plotting the ground/excited/other populations versus probe frequency. Runs on real hardware. Inspect the acquired populations directly.

## Parameter sources

Open and Reset take a snapshot of the current aliases, parameters and module choices. Relax delay is 5 * `$qubit.t1`. Probe sweep is `$resonator.freq` +/- 1.5 * `$resonator.kappa`.

Classification starts from `$qubit.ext.readout_cal.g_center`, `e_center` and `radius`. Missing calibration leaves these fields Invalid.

Pulse templates use `$qubit.freq`, `$qubit.wiring.ch`, `$resonator.freq`, `$resonator.wiring.ch` and `$resonator.wiring.ro_ch`. The readout trigger uses `$resonator.time_of_fly` + 0.05 us. Missing required sources leave the corresponding fields Invalid; supply them before Run.

T1-derived defaults use the Open/Reset snapshot. A missing or null T1, or an unavailable snapshot, uses 30 us, so the 5*T1 wait starts at 150 us. An invalid stored T1 leaves the required derived fields empty when the form opens. Alias, parameter-path or source errors are reported to the operator.

These starting values remain editable and separate from calibration writeback. Active reset keeps the same default wait; adjust it explicitly if needed.

## Module slots

Probe starts from res_probe. Optional init/reset use `$qubit.module.x180`/reset. Readout uses `$resonator.module.readout` or readout template. Pulse values remain editable, including those replaced by the sweep. You can select module slots by key. Run uses the Open/Reset snapshot.

## Typical writeback

This is a diagnostic only. Read the population curves from the plot; calibration parameters stay unchanged.

## Recommended

Set the classification calibration directly or seed it with 'singleshot/ge'. A frequency span around the resonator line captures the MIST response; use a large enough shot count (reps) for clean populations.

**Last updated:** 2026-10-21
