## Behavior

**Readers:** operator, agent. How to measure single-shot T1 under a probe tone from the GUI or MCP and interpret the results.

Single-shot T1-with-tone: applies a π pulse and a simultaneous probe tone during the wait, sweeps the wait-plus-probe length, and classifies each shot during acquisition against the |g>/|e> IQ-cluster centres. Plots the ground / excited / other populations versus delay time with dual-transition-rate fits for T1 and T1_b. The fit proposes the T1 value at `$qubit.ext.t1_with_tone`. Runs on real hardware.

## Parameter sources

Open and Reset take a snapshot of the current aliases, parameters and module choices. Relax delay is 5 * `$qubit.t1`. Length stop uses 5 times the tone T1 time scale.

Classification starts from `$qubit.ext.readout_cal.g_center`, `e_center` and `radius`. Missing calibration leaves these fields Invalid.

Pulse templates use `$qubit.freq`, `$qubit.wiring.ch`, `$resonator.freq`, `$resonator.wiring.ch` and `$resonator.wiring.ro_ch`. The readout trigger uses `$resonator.time_of_fly` + 0.05 us. Missing required sources leave the corresponding fields Invalid; supply them before Run.

The tone time scale uses the stored `$qubit.ext.t1_with_tone`, then ordinary T1, then 30 us. Missing or null sources advance to the next choice. An invalid stored tone T1 leaves the required stop empty for correction. Natural relaxation always uses ordinary T1.

T1-derived defaults use the Open/Reset snapshot. A missing or null T1, or an unavailable snapshot, uses 30 us, so the 5*T1 wait starts at 150 us. An invalid stored T1 leaves the required derived fields empty when the form opens. Alias, parameter-path or source errors are reported to the operator.

These starting values remain editable and separate from calibration writeback. Active reset keeps the same default wait; adjust it explicitly if needed.

## Module slots

Pi pulse uses `$qubit.module.x180` or pi_pulse. Probe starts from res_probe. Optional init/reset use `$qubit.module.x180`/reset. Readout uses `$qubit.module.readout` or readout template. Pulse values remain editable, including those replaced by the sweep. You can select module slots by key. Run uses the Open/Reset snapshot.

## Typical writeback

Proposes the fitted T1 relaxation time under the probe tone at `$qubit.ext.t1_with_tone`, in us.

## Recommended

Set the classification calibration directly or seed it with 'singleshot/ge'. Use 'uniform=False' (default) to cluster points along the expected exponential decay while preserving the configured window and point count; use 'uniform=True' for a linear sweep. The probe-tone gain and frequency are set in the `probe_pulse` module slot.

**Last updated:** 2026-10-21
