## Behavior

Amplitude Rabi: drives the qubit at its known frequency and sweeps the drive-pulse gain, fitting the resulting Rabi oscillation to find the pi and pi/2 pulse amplitudes. Runs on real hardware. Run once you know the qubit frequency, to calibrate a pi pulse by amplitude (the pulse length stays fixed).

## Expects (parameter context)

Gain runs -0.3 to 1.2 * qubit.module.x180.gain over 51 points; Run overwrites drive gain. Relax delay is 5 * qubit.t1 us. Missing calibration leaves the affected fields empty and Invalid; no numeric fallback.

## Expects (ModuleLibrary)

Uses captured qubit.module.readout with a readout template fallback, and qubit.module.reset with no fallback (disabled when absent). The drive uses qubit.module.x180 with a pi_pulse template fallback. Unset or dangling slots use the named template, not a legacy blank. Key choices remain available. Inline templates read qubit.freq, qubit.wiring.L01.ch, resonator.freq, resonator.wiring.ch/ro_ch and resonator.ext.time_of_fly + 0.05 us; missing required calibration remains Invalid.

## Typical writeback

Copies the source drive pulse into captured qubit.pulses.pi_amp and pi2_amp, replacing gain with the fitted pi and pi/2 gains. Separately proposes qubit.module.x180 and x90 pointing to those resolved library keys, even when already bound to the same keys. Also proposes qubit.ext.pi_product as fitted pi gain times the source pulse length (a.u. * us). Every item is independently selectable; none is proposed without a source cfg. Does not write separate gain scalars or read current calibration.

## Recommended

Analysis takes a 'Skip points' count (default 0) to drop leading low-gain sweep points before the cosine fit; raise it only if the first few points are distorted. A gain sweep spanning roughly two pi amplitudes captures at least a full oscillation; widen it if no full period is visible.
