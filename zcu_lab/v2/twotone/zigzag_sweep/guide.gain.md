## Behavior

**Readers:** operator, agent. Zig-zag gain scan: runs the zig-zag repetition sequence at each gain of the repeated pulse, and picks the gain where the signal stays flattest across repetitions. Runs on real hardware.

**Last updated:** 2026-10-21

## Parameter sources

The gain scan runs 0.8 to 1.2 * qubit.module.x180.gain; for X90 repetition adjust the window to x90. The frequency scan runs qubit.freq ± 2 MHz. Both use 101 points and maximum repetitions 6. Relax delay is 5 * qubit.t1 us. Other missing required calibration leaves the affected fields empty and Invalid.

Open and Reset capture the T1 used for starting values. A missing or null T1, or an absent calibration snapshot, uses 30 us, giving a 150 us wait. An invalid stored T1 leaves required derived fields empty. Alias or parameter lookup failures report an error. Review and edit these starting values before Run; they are form inputs only. Active reset keeps the same wait unless you change it.

## Module slots

Readout uses the qubit.module.readout slot captured at Open/Reset, or a readout template. Reset uses qubit.module.reset and stays disabled when absent. Drive slots use qubit.module.x180/x90 with pi_pulse/pi2_pulse template fallbacks. Unset slots or slots pointing to missing modules use the named template. Saved modules remain selectable. Pulse templates read qubit.freq, qubit.wiring.ch, resonator.freq, resonator.wiring.ch/ro_ch and resonator.time_of_fly + 0.05 us; missing required calibration remains Invalid.

## Typical writeback

Analysis reports results for review; calibration stays unchanged. The summary reports 'best_value', the best gain for the repeated pulse; update the pulse module yourself.

## Recommended

Analysis picks the sweep value whose signal changes least across repetitions (smoothed sum of step differences). Set 'Search from' / 'Search to' to ignore edges of the sweep. Keep n_times small (~6); the sweep multiplies the run time.
