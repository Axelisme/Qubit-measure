## Behavior

Single-qubit Clifford RB: each independent random seed supplies a shared prefix across depths, followed by its inverse and readout. Depth counts Cliffords, not physical pulses. Z rotations are virtual. Runs one hardware program per seed, sweeping depth within it.

## Expects (parameter context)

Depth starts in Log mode with max=128 and points=9, seed 42 and 20 sequences. Relax delay is 5 * qubit.t1 us. Other missing required calibration leaves the affected fields empty and Invalid.

T1-derived defaults use the fixed Open/Reset capture. A missing/null T1 or absent capture uses 30 us, so the 5*T1 wait starts at 150 us. Stored invalid T1 leaves derived required fields empty while the Form opens. Alias/path/source failures propagate. Defaults remain editable, are not written back as calibration, and do not shorten automatically for active reset.

## Expects (ModuleLibrary)

Uses captured qubit.module.readout with a readout template fallback, and qubit.module.reset with no fallback (disabled when absent). Drive slots use qubit.module.x180/x90 with pi_pulse/pi2_pulse template fallbacks. Unset or dangling slots use the named template, not a legacy blank. Key choices remain available. Inline templates read qubit.freq, qubit.wiring.ch, resonator.freq, resonator.wiring.ch/ro_ch and resonator.ext.time_of_fly + 0.05 us; missing required calibration remains Invalid.

## Typical writeback

No writeback. Fits mean projected IQ to A*p**depth+B; reports epc=(1-p)/2 and fidelity=1-epc per Clifford, not per physical gate.

## Depth input

The run cfg stores depths as a nonempty ordered list of unique nonnegative integers.
New Log forms produce [0,1,2,4,8,16,32,64,128]. Points includes zero and
must be at least 2. Log always includes 0 and max; two points yields [0,max].
Log rounds and deduplicates geometric samples. Effective points shows the actual
count when rounding collapses samples. Switch to List to edit the entire list.
Switching back fills max and points from the list and may resample it.
Old sweep.depth form input is dropped on restore. Stored IQ data and its depth
axis remain readable, even when its old cfg cannot be restored.

## Recommended

Calibrate frequency and pulse amplitudes first. Start with few seeds and shallow depths to verify execution, then use independent seeds and depths spanning the decay. Seed controls reproducibility. Review residuals and seed variation before interpreting fidelity.
