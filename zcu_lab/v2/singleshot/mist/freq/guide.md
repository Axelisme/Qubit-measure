## Behavior

MIST probe-frequency sweep: drives a probe pulse whose frequency is swept and classifies each shot in-program against the |g>/|e> single-shot centres, plotting the ground/excited/other populations versus probe frequency. Runs on real hardware; the result is already populations (no per-point fit).

## Expects (parameter context)

Open and Reset capture the current aliases, parameters and library. Relax delay is 5 * `$qubit.t1.` Probe sweep is `$resonator.freq` +/- 1.5 * `$resonator.kappa.` Classification seeds `$qubit.ext.readout_cal.g_center`/e_center/radius; missing calibration stays Invalid. Inline templates read `$qubit.freq`, `$qubit.wiring.ch` and `$resonator.freq`/wiring.ch/wiring.ro_ch; readout trigger uses `$resonator.ext.time_of_fly` + 0.05 us. Other missing required sources remain Invalid.

T1-derived defaults use the fixed Open/Reset capture. A missing/null T1 or absent capture uses 30 us, so the 5*T1 wait starts at 150 us. Stored invalid T1 leaves derived required fields empty while the Form opens. Alias/path/source failures propagate. Defaults remain editable, are not written back as calibration, and do not shorten automatically for active reset.

## Expects (ModuleLibrary)

Probe starts from res_probe. Optional init/reset use `$qubit.module.x180`/reset. Readout uses `$qubit.module.readout` or readout template. Sweep-overwritten pulse values remain editable and library keys stay selectable. Run uses this fixed capture.

## Typical writeback

No writeback — the population curves are read off the plot by eye.

## Recommended

Set calibration cfg directly or seed it with 'singleshot/ge'. A frequency span around the resonator line captures the MIST response; use a large enough shot count (reps) for clean populations.
