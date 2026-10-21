## Behavior

**Readers:** operator, agent. Zig-zag: applies an X90 pulse, then repeats the chosen pulse 0..n_times times before readout, amplifying small pulse-amplitude errors into a growing zig-zag of the signal versus repetition count. Runs on real hardware. Run after amplitude Rabi has calibrated the pi and pi/2 pulses.

**Last updated:** 2026-10-21

## Parameter sources

Maximum repetitions defaults to 10. Relax delay is 5 * qubit.t1 us. Other missing required calibration leaves the affected fields empty and Invalid.

Open and Reset capture the T1 used for starting values. A missing or null T1, or an absent calibration snapshot, uses 30 us, giving a 150 us wait. An invalid stored T1 leaves required derived fields empty. Alias or parameter lookup failures report an error. Review and edit these starting values before Run; they are form inputs only. Active reset keeps the same wait unless you change it.

## Module slots

Readout uses the qubit.module.readout slot captured at Open/Reset, or a readout template. Reset uses qubit.module.reset and stays disabled when absent. Drive slots use qubit.module.x180/x90 with pi_pulse/pi2_pulse template fallbacks. Unset slots or slots pointing to missing modules use the named template. Saved modules remain selectable. Pulse templates read qubit.freq, qubit.wiring.ch, resonator.freq, resonator.wiring.ch/ro_ch and resonator.ext.time_of_fly + 0.05 us; missing required calibration remains Invalid.

## Typical writeback

Review the measured trace manually; calibration stays unchanged. Preparation coherence, contrast, and drive history can also change the trace; compare controls before changing gate calibration.

## Recommended

Repeat on X180_pulse to check the pi pulse; repeat on X90_pulse (applied in pairs) to check the pi/2 pulse. About 10 repetitions usually show a mis-calibration clearly. Use the zig-zag gain or frequency scans to sweep the repeated pulse's gain or frequency.
