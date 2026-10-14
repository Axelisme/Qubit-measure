## Behavior

Single-qubit Clifford RB: each independent random seed supplies a shared prefix across depths, followed by its inverse and readout. Depth counts Cliffords, not physical pulses. Z rotations are virtual. Runs one hardware program per seed, sweeping depth within it.

## Expects (parameter context)

Uses t1 (us) for a 5*t1 recovery delay, fallback 200 us.

## Expects (ModuleLibrary)

Requires calibrated X90 (pi2_amp/pi2_len), X180 (pi_amp/pi_len) and readout modules. Verify both drive frequencies and channels. Identity uses X90 at zero gain. Reset is optional.

## Typical writeback

No writeback. Fits mean projected IQ to A*p**depth+B; reports epc=(1-p)/2 and fidelity=1-epc per Clifford, not per physical gate.

## Recommended

Calibrate frequency and pulse amplitudes first. Start with few seeds and shallow depths to verify execution, then use independent seeds and depths spanning the decay. Seed controls reproducibility. Review residuals and seed variation before interpreting fidelity.
