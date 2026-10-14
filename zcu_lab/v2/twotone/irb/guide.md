## Behavior

Paired single-qubit IRB: reference and interleaved arms share random Cliffords, each with its own full inverse. Depth counts random Cliffords; interleaved inserts one target per Clifford. Hardware sweeps depth and repetitions; software visits rounds, seeds and alternating arm order. Raw IQ has fixed axes round/seed/arm/depth, arm 0=reference, 1=interleaved.

## Expects (parameter context)

Uses 5*t1 relaxation (us), fallback 200 us.

## Expects (ModuleLibrary)

Uses calibrated X90/X180 and readout modules. Y gates use phase-shifted X pulses.

## Typical writeback

No writeback. Fits both A*p**depth+B curves on a common IQ projection. Reports target error=(1-p_interleaved/p_reference)/2 and fidelity. 95% paired-seed bootstrap is statistical only; coherent or gate-dependent noise can bias this estimate. Nonphysical estimates are reported unclipped.

## Recommended

Default target X90. Use at least 4 distinct depths spanning decay and 20 or more seeds for uncertainty. Start with a short execution check. Analysis uses complete round/seed pairs; unfinished raw IQ is still saved.
