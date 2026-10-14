## Behavior

Single-shot T1: applies a π pulse, waits for a variable delay, reads out, and classifies each shot in-program against the |g>/|e> IQ-cluster centres. Repeats from both |g> and |e> initial states (Branch). Plots the ground / excited / other populations versus delay time with dual-transition-rate fits for T1 and T1_b. Runs on real hardware.

## Expects (parameter context)

Open and Reset capture the current aliases, parameters and library. Relax delay and length stop are 5 * `$qubit.t1.` Classification seeds `$qubit.ext.readout_cal.g_center`/e_center/radius; missing calibration stays Invalid. Inline templates read `$qubit.freq`, `$qubit.wiring.L01.ch` and `$resonator.freq`/wiring.ch/wiring.ro_ch; readout trigger uses `$resonator.ext.time_of_fly` + 0.05 us. Missing required sources remain Invalid; no numeric fallback is supplied.

## Expects (ModuleLibrary)

Pi pulse uses `$qubit.module.x180` or pi_pulse. Reset is optional; readout uses `$qubit.module.readout` or readout template. Sweep-overwritten pulse values remain editable and library keys stay selectable. Run uses this fixed capture.

## Typical writeback

No writeback — T1 is shown in the plot title only. Use 'singleshot/t1_tone' if you need the T1 scalar in the parameter context.

## Recommended

Set calibration cfg directly or seed it with 'singleshot/ge'. A delay sweep reaching ~5*T1 lets the decay flatten. Missing t1 leaves the length stop and relax delay Invalid. Set 'uniform=True' to sweep linearly; leave False to cluster more points along the expected exponential decay while preserving the configured start/stop window and point count.
