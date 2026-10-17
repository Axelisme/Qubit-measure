## Behavior

Amplitude Rabi: drives the qubit at its known frequency and sweeps the drive-pulse gain, fitting the resulting Rabi oscillation to find the pi and pi/2 pulse amplitudes. Runs on real hardware. Run once you know the qubit frequency, to calibrate a pi pulse by amplitude (the pulse length stays fixed).

## Expects (parameter context)

Gain runs -0.3 to 1.0 over 51 points; Run overwrites drive gain. Relax delay is 5 * qubit.t1 us. Other missing required calibration leaves the affected fields empty and Invalid.

Both Amp Rabi forms sweep gain from -0.3 to 1.0 regardless of calibration. For x180 with a persistent scalar waveform length, the inline copy starts at length = 1.5*pi_len*pi_gain us. pi_len must be a finite positive non-bool number and pi_gain must be finite, non-bool and in (0, 1]. Missing/null or invalid leaves leave length required and empty. Finite invalid gain stays visible; nonfinite gain leaves both gain and length empty. Without x180, qub_probe starts at gain=1 and length=1 us. Arb x180 keeps its original Slot, shape and data without copying or overriding length or reading assets; change waveform first to edit duration.

T1-derived defaults use the fixed Open/Reset capture. A missing/null T1 or absent capture uses 30 us, so the 5*T1 wait starts at 150 us. Stored invalid T1 leaves derived required fields empty while the Form opens. Alias/path/source failures propagate. Defaults remain editable, are not written back as calibration, and do not shorten automatically for active reset.

## Expects (ModuleLibrary)

Uses captured qubit.module.readout with a readout template fallback, and qubit.module.reset with no fallback (disabled when absent). The drive follows the shared Rabi source policy above, using qubit.module.x180 when present and qub_probe when absent. Unset or dangling slots use the named template, not a legacy blank. Key choices remain available. Inline templates read qubit.freq, qubit.wiring.ch, resonator.freq, resonator.wiring.ch/ro_ch and resonator.ext.time_of_fly + 0.05 us; missing required calibration remains Invalid.

## Typical writeback

Copies the source drive pulse into captured qubit.pulses.pi_amp and pi2_amp, replacing gain with the fitted pi and pi/2 gains. Separately proposes qubit.module.x180 and x90 pointing to those resolved library keys, even when already bound to the same keys. Also proposes qubit.ext.pi_product as fitted pi gain times the source pulse length (a.u. * us). Every item is independently selectable; none is proposed without a source cfg. Does not write separate gain scalars or read current calibration.

## Recommended

Analysis takes a 'Skip points' count (default 0) to drop leading low-gain sweep points before the cosine fit; raise it only if the first few points are distorted. A gain sweep spanning roughly two pi amplitudes captures at least a full oscillation; widen it if no full period is visible.
