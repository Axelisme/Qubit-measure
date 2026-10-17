## Behavior

Zig-zag: applies an X90 pulse, then repeats the chosen pulse 0..n_times times before readout, amplifying small pulse-amplitude errors into a growing zig-zag of the signal versus repetition count. Runs on real hardware. Run after amplitude Rabi has calibrated the pi and pi/2 pulses.

## Expects (parameter context)

Maximum repetitions defaults to 10. Relax delay is 5 * qubit.t1 us. Other missing required calibration leaves the affected fields empty and Invalid.

T1-derived defaults use the fixed Open/Reset capture. A missing/null T1 or absent capture uses 30 us, so the 5*T1 wait starts at 150 us. Stored invalid T1 leaves derived required fields empty while the Form opens. Alias/path/source failures propagate. Defaults remain editable, are not written back as calibration, and do not shorten automatically for active reset.

## Expects (ModuleLibrary)

Uses captured qubit.module.readout with a readout template fallback, and qubit.module.reset with no fallback (disabled when absent). Drive slots use qubit.module.x180/x90 with pi_pulse/pi2_pulse template fallbacks. Unset or dangling slots use the named template, not a legacy blank. Key choices remain available. Inline templates read qubit.freq, qubit.wiring.ch, resonator.freq, resonator.wiring.ch/ro_ch and resonator.ext.time_of_fly + 0.05 us; missing required calibration remains Invalid.

## Typical writeback

No analysis and no writeback. Preparation coherence, contrast, and drive history can also change the trace; compare controls before changing gate calibration.

## Recommended

Repeat on X180_pulse to check the pi pulse; repeat on X90_pulse (applied in pairs) to check the pi/2 pulse. About 10 repetitions usually show a mis-calibration clearly. Use the zig-zag scan adapters to sweep the repeated pulse's gain or frequency. reset_phase_cycle alternates the final reset pi phase by 180 degrees between averaging sweeps (even reps, two-pulse/bath reset). It adds no RF pulse or programmed wait; verify population and phase witnesses.
