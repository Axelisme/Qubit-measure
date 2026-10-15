## Behavior

Length Rabi: drives the qubit at its known frequency and sweeps the drive-pulse length, fitting the Rabi oscillation to find the pi and pi/2 pulse lengths. Runs on real hardware. Run once you know the qubit frequency, to calibrate a pi pulse by duration (the pulse gain stays fixed).

## Expects (parameter context)

Length runs 0.03 to 4 * qubit.module.x180.waveform.length us over 101 points; Run overwrites drive length. Relax delay is 5 * qubit.t1 us. Missing calibration leaves the affected fields empty and Invalid; no numeric fallback.

## Expects (ModuleLibrary)

Uses captured qubit.module.readout with a readout template fallback, and qubit.module.reset with no fallback (disabled when absent). The drive uses qubit.module.x180 with a pi_pulse template fallback. Unset or dangling slots use the named template, not a legacy blank. Key choices remain available. Inline templates read qubit.freq, qubit.wiring.L01.ch, resonator.freq, resonator.wiring.ch/ro_ch and resonator.ext.time_of_fly + 0.05 us; missing required calibration remains Invalid.

## Typical writeback

Proposes qubit.ext.rabi_freq in MHz from the fit. With a source cfg, copies the drive pulse into captured qubit.pulses.pi_amp and pi2_amp, replacing waveform.length with fitted pi and pi/2 lengths. Separately proposes qubit.module.x180/x90 pointing to those resolved keys, even when already bound to them, plus qubit.ext.pi_product = source gain * fitted pi length (a.u. * us). Every item is independently selectable. Without cfg only rabi_freq remains. No separate length scalars or live defaults.

## Recommended

Phase defaults to fixed 0/180 degrees; enable Fit phase offset when pulse shaping shifts the length oscillation. Analysis defaults to fitting a decay envelope on the oscillation; keep it on when the Rabi oscillation visibly damps over the sweep, turn it off for a pure undamped cosine fit. A length sweep spanning a few pi lengths (up to 4*pi_len when calibrated) captures a full oscillation; widen it if no full period is visible.
