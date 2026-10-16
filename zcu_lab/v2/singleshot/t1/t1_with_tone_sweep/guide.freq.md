## Behavior

Single-shot T1-with-tone, probe-frequency sweep: at each probe frequency it sweeps the wait-plus-probe length from both |g> and |e> initial states (Branch), classifies each shot in-program against the |g>/|e> centres, fits the dual transition rates per frequency, and renders the 3×3 rate landscape over (freq, time). Runs on real hardware.

## Expects (parameter context)

Open and Reset capture the current aliases, parameters and library. Relax delay and length stop are 5 * `$qubit.t1.` Frequency form sweeps `$resonator.freq` +/- 1.5 * `$resonator.kappa`; gain form sweeps 0 to 1. Each form exposes only one outer axis. Classification seeds `$qubit.ext.readout_cal.g_center`/e_center/radius; missing calibration stays Invalid. Inline templates read `$qubit.freq`, `$qubit.wiring.ch` and `$resonator.freq`/wiring.ch/wiring.ro_ch; readout trigger uses `$resonator.ext.time_of_fly` + 0.05 us. Missing required sources remain Invalid; no numeric fallback is supplied.

## Expects (ModuleLibrary)

Pi pulse uses `$qubit.module.x180` or pi_pulse. Probe starts from res_probe. Reset is optional; readout uses `$qubit.module.readout` or readout template. Sweep-overwritten pulse values remain editable and library keys stay selectable. Run uses this fixed capture.

## Typical writeback

No writeback — the rate landscape is read off the grid by eye.

## Recommended

Set calibration cfg directly or seed it with 'singleshot/ge'. Keep the frequency sweep coarse (the inner length sweep multiplies the run time). 'uniform=True' (default) uses a linear length sweep; set False to cluster points along the expected exponential decay while preserving the configured window and count.
