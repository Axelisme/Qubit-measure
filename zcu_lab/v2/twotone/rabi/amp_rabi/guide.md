## Behavior

**Readers:** operator, agent. Amplitude Rabi: drives the qubit at its known frequency and sweeps the drive-pulse gain, fitting the resulting Rabi oscillation to find the pi and pi/2 pulse amplitudes. Runs on real hardware. Run once you know the qubit frequency, to calibrate a pi pulse by amplitude (the pulse length stays fixed).

**Last updated:** 2026-10-21

## Parameter sources

Gain runs -0.3 to 1.0 over 51 points; Run overwrites drive gain. Relax delay is 5 * qubit.t1 us. Other missing required calibration leaves the affected fields empty and Invalid.

The gain sweep runs from -0.3 to 1.0 regardless of calibration. For an x180 pulse with a stored numeric waveform length, the editable copy starts at length = 1.5*pi_len*pi_gain us. Here pi_len is the calibrated x180 length and pi_gain its gain. Enter a finite positive pi_len and a finite pi_gain in (0, 1]; missing or invalid values leave length required and empty. A finite invalid gain stays visible; a nonfinite gain leaves both gain and length empty. If x180 is absent, qub_probe starts at gain=1 and length=1 us. An arbitrary-waveform x180 keeps its saved module, shape and data. Switch waveform type first to edit duration.

Open and Reset capture the T1 used for starting values. A missing or null T1, or an absent calibration snapshot, uses 30 us, giving a 150 us wait. An invalid stored T1 leaves required derived fields empty. Alias or parameter lookup failures report an error. Review and edit these starting values before Run; they are form inputs only. Active reset keeps the same wait unless you change it.

## Module slots

Readout uses the qubit.module.readout slot captured at Open/Reset, or a readout template. Reset uses qubit.module.reset and stays disabled when absent. The drive uses the sources described above, with qubit.module.x180 when present and qub_probe when absent. Unset slots or slots pointing to missing modules use the named template. Saved modules remain selectable. Pulse templates read qubit.freq, qubit.wiring.ch, resonator.freq, resonator.wiring.ch/ro_ch and resonator.time_of_fly + 0.05 us; missing required calibration remains Invalid.

## Typical writeback

Copies the source drive pulse into captured qubit.pulses.pi_amp and pi2_amp, replacing gain with the fitted pi and pi/2 gains. Separately proposes qubit.module.x180 and x90 pointing to those saved pulses, even when already bound to the same keys. Also proposes qubit.ext.pi_product as fitted pi gain times the source pulse length (a.u. * us). Every item is independently selectable. Proposals require acquired run settings and use that run's source pulse. Gain updates are part of the pulse copies.

## Recommended

Analysis takes a 'Skip points' count (default 0) to drop leading low-gain sweep points before the cosine fit; raise it only if the first few points are distorted. A gain sweep spanning roughly two pi amplitudes captures at least a full oscillation; widen it if no full period is visible.
