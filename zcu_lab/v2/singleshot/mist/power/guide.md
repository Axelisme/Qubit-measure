## Behavior

MIST probe-power sweep: drives a probe pulse whose gain is swept and classifies each shot in-program against the |g>/|e> single-shot centres, plotting the ground/excited/other populations versus probe gain (or photon number when an AC-Stark coefficient is known). Runs on real hardware; the result is already populations (no per-point fit).

## Expects (parameter context)

Open and Reset capture the current aliases, parameters and library. Relax delay is 5 * `$qubit.t1.` Probe gain sweep runs from 0 to 1. Classification seeds `$qubit.ext.readout_cal.g_center`/e_center/radius; missing calibration stays Invalid. Inline templates read `$qubit.freq`, `$qubit.wiring.L01.ch` and `$resonator.freq`/wiring.ch/wiring.ro_ch; readout trigger uses `$resonator.ext.time_of_fly` + 0.05 us. Missing required sources remain Invalid; no numeric fallback is supplied.

## Expects (ModuleLibrary)

Probe starts from res_probe. Optional init/reset use `$qubit.module.x180`/reset. Readout uses `$qubit.module.readout` or readout template. Sweep-overwritten pulse values remain editable and library keys stay selectable. Run uses this fixed capture.

## Typical writeback

No writeback — the population curves are read off the plot by eye.

## Recommended

Set calibration cfg directly or seed it with 'singleshot/ge'. Sweep the probe gain across the MIST onset; provide '`$qubit.ext.ac_stark_coeff`' (from the AC-Stark experiment) for a photon-number x-axis.
