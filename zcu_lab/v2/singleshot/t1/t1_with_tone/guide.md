## Behavior

Single-shot T1-with-tone: applies a π pulse and a simultaneous probe tone during the wait, sweeps the wait-plus-probe length, and classifies each shot in-program against the |g>/|e> IQ-cluster centres. Plots the ground / excited / other populations versus delay time with dual-transition-rate fits for T1 and T1_b. The fitted T1 value is written back to '`$qubit.ext.t1_with_tone`' in the parameter context. Runs on real hardware.

## Expects (parameter context)

Open and Reset capture the current aliases, parameters and library. Relax delay is 5 * `$qubit.t1.` Length stop is 5 * `$qubit.ext.t1_with_tone`; there is no fallback to t1. Classification seeds `$qubit.ext.readout_cal.g_center`/e_center/radius; missing calibration stays Invalid. Inline templates read `$qubit.freq`, `$qubit.wiring.ch` and `$resonator.freq`/wiring.ch/wiring.ro_ch; readout trigger uses `$resonator.ext.time_of_fly` + 0.05 us. Missing required sources remain Invalid; no numeric fallback is supplied.

## Expects (ModuleLibrary)

Pi pulse uses `$qubit.module.x180` or pi_pulse. Probe starts from res_probe. Optional init/reset use `$qubit.module.x180`/reset. Readout uses `$qubit.module.readout` or readout template. Sweep-overwritten pulse values remain editable and library keys stay selectable. Run uses this fixed capture.

## Typical writeback

Proposes the fitted T1 relaxation time under the probe tone into parameter context '`$qubit.ext.t1_with_tone`' (us).

## Recommended

Set calibration cfg directly or seed it with 'singleshot/ge'. Use 'uniform=False' (default) to cluster points along the expected exponential decay while preserving the configured window and point count; use 'uniform=True' for a linear sweep. The probe-tone gain and frequency are set inside the probe_pulse module.
