## Behavior

Length Rabi: drives the qubit at its known frequency and sweeps the drive-pulse length, fitting the Rabi oscillation to find the pi and pi/2 pulse lengths. Runs on real hardware. Run once you know the qubit frequency, to calibrate a pi pulse by duration (the pulse gain stays fixed).

## Expects (parameter context)

Uses the tab's captured qubit and resonator roles. Length sweep stop is 4 * qubit.module.x180.waveform.length (fallback 0.3 us); relax delay is 5 * qubit.t1 (fallback 30.5 us). Fresh drive uses qubit.freq (fallback 4000 MHz) and wiring.L01.ch (fallback 0). Fresh readout uses resonator.freq (fallback 6000 MHz) and wiring.ch/wiring.ro_ch (fallback 0), with trigger offset 0.55 us.

## Expects (ModuleLibrary)

Mounts captured qubit.module.x180/readout/reset slots. Missing or dangling drive/readout slots use fresh inline pulses; reset stays disabled. The length sweep owns waveform.length, locked at 1 us in the cfg. Inline drive gain is 0.3; inline readout gain is 0.1, const length 1 us and ADC window 0.9 us. No named waveform is adopted.

## Typical writeback

Proposes qubit.ext.rabi_freq in MHz from the fit. With a source cfg, copies the drive pulse into captured qubit.pulses.pi_amp and pi2_amp, replacing waveform.length with fitted pi and pi/2 lengths. Separately proposes qubit.module.x180/x90 pointing to those resolved keys, even when already bound to them, plus qubit.ext.pi_product = source gain * fitted pi length (a.u. * us). Every item is independently selectable. Without cfg only rabi_freq remains. No separate length scalars or live defaults.

## Recommended

Phase defaults to fixed 0/180 degrees; enable Fit phase offset when pulse shaping shifts the length oscillation. Analysis defaults to fitting a decay envelope on the oscillation; keep it on when the Rabi oscillation visibly damps over the sweep, turn it off for a pure undamped cosine fit. A length sweep spanning a few pi lengths (up to 4*pi_len when calibrated) captures a full oscillation; widen it if no full period is visible.
