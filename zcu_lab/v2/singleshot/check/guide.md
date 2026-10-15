## Behavior

Single-shot classification check: takes 'shots' single-shot readouts of the prepared state and draws the IQ scatter with the |g>/|e> discrimination circles overlaid, reporting the fraction classified ground / excited / other. Runs on real hardware; the run itself needs no centres, but the analysis classifies the scatter against them.

## Expects (parameter context)

Open and Reset capture the current aliases, parameters and library. Relax delay is 5 * `$qubit.t1.` Fresh inline probe uses `$qubit.freq` and `$qubit.wiring.L01.ch.` Inline templates read `$qubit.freq`, `$qubit.wiring.L01.ch` and `$resonator.freq`/wiring.ch/wiring.ro_ch; readout trigger uses `$resonator.ext.time_of_fly` + 0.05 us. Missing required sources remain Invalid; no numeric fallback is supplied.

## Expects (ModuleLibrary)

Optional init/reset use `$qubit.module.x180`/reset. Probe starts from qub_probe. Readout uses `$qubit.module.readout` or the readout template. Run uses this fixed capture.

## Typical writeback

No writeback — the check is a visual diagnostic of the existing single-shot discrimination.

## Recommended

Run after 'singleshot/ge'. Use a large 'shots' (~5000+) for a well-sampled scatter; a tight cluster inside the matching circle indicates good discrimination.
