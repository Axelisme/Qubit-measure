## Behavior

Single-shot T1-with-tone, probe-frequency sweep: at each probe frequency it sweeps the wait-plus-probe length from both |g> and |e> initial states (Branch), classifies each shot in-program against the |g>/|e> centres, fits the dual transition rates per frequency, and renders the 3×3 rate landscape over (freq, time). Runs on real hardware.

## Expects (parameter context)

Open and Reset capture the current aliases, parameters and library. Relax delay starts at 5*T1; length stop uses 5 times the tone T1 time scale. Frequency form sweeps `$resonator.freq` +/- 1.5 * `$resonator.kappa`; gain form sweeps 0 to 1. Each form exposes only one outer axis. Classification seeds `$qubit.ext.readout_cal.g_center`/e_center/radius; missing calibration stays Invalid. Inline templates read `$qubit.freq`, `$qubit.wiring.ch` and `$resonator.freq`/wiring.ch/wiring.ro_ch; readout trigger uses `$resonator.ext.time_of_fly` + 0.05 us. Other missing required sources remain Invalid.

The tone time scale uses historical t1_with_tone, then ordinary T1, then 30 us. Only missing/null sources advance to the next choice; stored invalid tone T1 leaves stop empty. Natural relaxation always uses ordinary T1.

T1-derived defaults use the fixed Open/Reset capture. A missing/null T1 or absent capture uses 30 us, so the 5*T1 wait starts at 150 us. Stored invalid T1 leaves derived required fields empty while the Form opens. Alias/path/source failures propagate. Defaults remain editable, are not written back as calibration, and do not shorten automatically for active reset.

## Expects (ModuleLibrary)

Pi pulse uses `$qubit.module.x180` or pi_pulse. Probe starts from res_probe. Reset is optional; readout uses `$qubit.module.readout` or readout template. Sweep-overwritten pulse values remain editable and library keys stay selectable. Run uses this fixed capture.

## Typical writeback

No writeback — the rate landscape is read off the grid by eye.

## Recommended

Set calibration cfg directly or seed it with 'singleshot/ge'. Keep the frequency sweep coarse (the inner length sweep multiplies the run time). 'uniform=True' (default) uses a linear length sweep; set False to cluster points along the expected exponential decay while preserving the configured window and count.
