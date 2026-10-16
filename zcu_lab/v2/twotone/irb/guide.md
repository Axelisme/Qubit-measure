## Behavior

Paired single-qubit IRB: reference and interleaved arms share random Cliffords, each with its own full inverse. Depth counts random Cliffords; interleaved inserts one target per Clifford. Hardware sweeps depth and repetitions; software visits rounds, seeds and alternating arm order. Raw IQ has fixed axes round/seed/arm/depth, arm 0=reference, 1=interleaved.

## Expects (parameter context)

Depth runs 0 to 100 random Cliffords over 21 points, seed 42 and 20 paired sequences. Relax delay is 5 * qubit.t1 us. Missing calibration leaves the affected fields empty and Invalid; no numeric fallback.

## Expects (ModuleLibrary)

Uses captured qubit.module.readout with a readout template fallback, and qubit.module.reset with no fallback (disabled when absent). Drive slots use qubit.module.x180/x90 with pi_pulse/pi2_pulse template fallbacks. Unset or dangling slots use the named template, not a legacy blank. Key choices remain available. Inline templates read qubit.freq, qubit.wiring.ch, resonator.freq, resonator.wiring.ch/ro_ch and resonator.ext.time_of_fly + 0.05 us; missing required calibration remains Invalid.

## Typical writeback

No writeback. Fits both A*p**depth+B curves on a common IQ projection. Reports target error=(1-p_interleaved/p_reference)/2 and fidelity. 95% paired-seed bootstrap is statistical only; coherent or gate-dependent noise can bias this estimate. Nonphysical estimates are reported unclipped.

## Recommended

Default target X90. Use at least 4 distinct depths spanning decay and 20 or more seeds for uncertainty. Start with a short execution check. Analysis uses complete round/seed pairs; unfinished raw IQ is still saved.
