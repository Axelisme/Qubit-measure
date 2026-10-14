## Behavior

Amplitude Rabi: drives the qubit at its known frequency and sweeps the drive-pulse gain, fitting the resulting Rabi oscillation to find the pi and pi/2 pulse amplitudes. Runs on real hardware. Run once you know the qubit frequency, to calibrate a pi pulse by amplitude (the pulse length stays fixed).

## Expects (parameter context)

Uses the tab's captured qubit and resonator roles. Gain sweep stop is 1.2 * qubit.module.x180.gain (fallback 0.6); relax delay is 5 * qubit.t1 (fallback 30.5 us). Fresh drive defaults read qubit.freq and qubit.wiring.L01.ch, with length 1.01 * qubit.module.x180.waveform.length (fallback 1.1 us). Fresh readout reads resonator.freq, wiring.ch, wiring.ro_ch and ext.time_of_fly + 0.05 us (fallback 0.55 us).

## Expects (ModuleLibrary)

Mounts captured qubit.module.x180, qubit.module.readout and qubit.module.reset slots. Missing/null or dangling drive and readout slots use fresh inline pulses; reset stays disabled. The gain sweep owns drive gain, locked at zero in the cfg. Inline readout uses a const waveform, not a named waveform.

## Typical writeback

Copies the source drive pulse into captured qubit.pulses.pi_amp and pi2_amp, replacing gain with the fitted pi and pi/2 gains. Separately proposes qubit.module.x180 and x90 pointing to those resolved library keys, even when already bound to the same keys. Also proposes qubit.ext.pi_product as fitted pi gain times the source pulse length (a.u. * us). Every item is independently selectable; none is proposed without a source cfg. Does not write separate gain scalars or read current calibration.

## Recommended

Analysis takes a 'Skip points' count (default 0) to drop leading low-gain sweep points before the cosine fit; raise it only if the first few points are distorted. A gain sweep spanning roughly two pi amplitudes captures at least a full oscillation; widen it if no full period is visible.
