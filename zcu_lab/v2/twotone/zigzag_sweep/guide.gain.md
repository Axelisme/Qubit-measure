## Behavior

Zig-zag gain scan: runs the zig-zag repetition sequence at each gain of the repeated pulse, and picks the gain where the signal stays flattest across repetitions. Runs on real hardware.

## Expects (parameter context)

The gain scan runs 0.8 to 1.2 * qubit.module.x180.gain; for X90 repetition adjust the window to x90. The frequency scan runs qubit.freq ± 2 MHz. Both use 101 points and maximum repetitions 6. Relax delay is 5 * qubit.t1 us. Other missing required calibration leaves the affected fields empty and Invalid.

T1-derived defaults use the fixed Open/Reset capture. A missing/null T1 or absent capture uses 30 us, so the 5*T1 wait starts at 150 us. Stored invalid T1 leaves derived required fields empty while the Form opens. Alias/path/source failures propagate. Defaults remain editable, are not written back as calibration, and do not shorten automatically for active reset.

## Expects (ModuleLibrary)

Uses captured qubit.module.readout with a readout template fallback, and qubit.module.reset with no fallback (disabled when absent). Drive slots use qubit.module.x180/x90 with pi_pulse/pi2_pulse template fallbacks. Unset or dangling slots use the named template, not a legacy blank. Key choices remain available. Inline templates read qubit.freq, qubit.wiring.ch, resonator.freq, resonator.wiring.ch/ro_ch and resonator.ext.time_of_fly + 0.05 us; missing required calibration remains Invalid.

## Typical writeback

No writeback. The summary reports 'best_value', the best gain for the repeated pulse; update the pulse module yourself.

## Recommended

Analysis picks the sweep value whose signal changes least across repetitions (smoothed sum of step differences). Set 'Search from' / 'Search to' to ignore edges of the sweep. Keep n_times small (~6); the sweep multiplies the run time.
