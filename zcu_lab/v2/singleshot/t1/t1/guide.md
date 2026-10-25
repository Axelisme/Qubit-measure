## Behavior

**Readers:** operator, agent. How to measure single-shot T1 from the GUI or MCP and interpret the results.

Single-shot T1: applies a π pulse, waits for a variable delay, reads out, and classifies each shot during acquisition against the |g>/|e> IQ-cluster centres. Repeats from both |g> and |e> initial states (Branch). Plots the ground / excited / other populations versus delay time with dual-transition-rate fits for T1 and T1_b. Runs on real hardware.

## Parameter sources

Open and Reset take a snapshot of the current aliases, parameters and module choices. Relax delay and length stop are 5 * `$qubit.t1`.

Classification starts from `$qubit.ext.readout_cal.g_center`, `e_center` and `radius`. Missing calibration leaves these fields Invalid.

Pulse templates use `$qubit.freq`, `$qubit.wiring.ch`, `$resonator.freq`, `$resonator.wiring.ch` and `$resonator.wiring.ro_ch`. The readout trigger uses `$resonator.time_of_fly` + 0.05 us. Missing required sources leave the corresponding fields Invalid; supply them before Run.

The ordinary T1 sweep stop is 5 times the same time scale, initially 150 us when T1 is missing.

T1-derived defaults use the Open/Reset snapshot. A missing or null T1, or an unavailable snapshot, uses 30 us, so the 5*T1 wait starts at 150 us. An invalid stored T1 leaves the required derived fields empty when the form opens. Alias, parameter-path or source errors are reported to the operator.

These starting values remain editable and separate from calibration writeback. Active reset keeps the same default wait; adjust it explicitly if needed.

## Module slots

Pi pulse uses `$qubit.module.x180` or pi_pulse. Reset is optional; readout uses `$resonator.module.readout` or readout template. Pulse values remain editable, including those replaced by the sweep. You can select module slots by key. Run uses the Open/Reset snapshot.

## Typical writeback

T1 appears in the plot title; calibration parameters stay unchanged. Use 'singleshot/t1_tone' to propose the tone T1 scalar in the parameters.

## Recommended

Set the classification calibration directly or seed it with 'singleshot/ge'. A delay sweep reaching ~5*T1 lets the decay flatten. Missing T1 starts length stop and relax delay at 150 us; stored invalid T1 leaves both required and empty. Set 'uniform=True' to sweep linearly; leave False to cluster more points along the expected exponential decay while preserving the configured start/stop window and point count.

**Last updated:** 2026-10-21
