## Behavior

Single-qubit Clifford RB: each independent random seed supplies a shared prefix across depths, followed by its inverse and readout. Depth counts Cliffords, not physical pulses. Z rotations are virtual. Runs one hardware program per seed, sweeping depth within it.

## Expects (parameter context)

Depth runs 0 to 100 Cliffords over 21 points, seed 42 and 20 sequences. Relax delay is 5 * qubit.t1 us. Missing calibration leaves the affected fields empty and Invalid; no numeric fallback.

## Expects (ModuleLibrary)

Uses captured qubit.module.readout with a readout template fallback, and qubit.module.reset with no fallback (disabled when absent). Drive slots use qubit.module.x180/x90 with pi_pulse/pi2_pulse template fallbacks. Unset or dangling slots use the named template, not a legacy blank. Key choices remain available. Inline templates read qubit.freq, qubit.wiring.ch, resonator.freq, resonator.wiring.ch/ro_ch and resonator.ext.time_of_fly + 0.05 us; missing required calibration remains Invalid.

## Typical writeback

No writeback. Fits mean projected IQ to A*p**depth+B; reports epc=(1-p)/2 and fidelity=1-epc per Clifford, not per physical gate.

## Recommended

Calibrate frequency and pulse amplitudes first. Start with few seeds and shallow depths to verify execution, then use independent seeds and depths spanning the decay. Seed controls reproducibility. Review residuals and seed variation before interpreting fidelity.
