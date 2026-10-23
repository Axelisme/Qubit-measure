## Behavior

**Readers:** operator, agent. Length Rabi: drives the qubit at its known frequency and sweeps the drive-pulse length, fitting the Rabi oscillation to find the pi and pi/2 pulse lengths. Runs on real hardware. Run once you know the qubit frequency, to calibrate a pi pulse by duration (the pulse gain stays fixed).

**Last updated:** 2026-10-21

After changing the parameter source, use Reset to recapture initial values. Refresh only updates sources and choices while preserving authored inputs; Reset can overwrite unsaved inputs.

## Parameter sources

Length runs 0.03 to 4 * qubit.module.x180.waveform.length us over 101 points; Run overwrites drive length. Relax delay is 5 * qubit.t1 us. Other missing required calibration leaves the affected fields empty and Invalid.

The length sweep starts at 0.03 us and stops at 4*pi_len, using the stored x180 waveform length as pi_len. A missing or null pi_len, including an arbitrary waveform with no stored numeric length, uses a 1 us stop. An invalid stored pi_len leaves stop empty. Enter a finite positive numeric length. An existing x180 keeps its saved module selection and gain; if x180 is absent, qub_probe starts at gain=1. Starting values use stored calibration only.

Open and Reset capture the T1 used for starting values. A missing or null T1, or an absent calibration snapshot, uses 30 us, giving a 150 us wait. An invalid stored T1 leaves required derived fields empty. Alias or parameter lookup failures report an error. Review and edit these starting values before Run; they are form inputs only. Active reset keeps the same wait unless you change it.

## Module slots

Readout uses the qubit.module.readout slot captured at Open/Reset, or a readout template. Reset uses qubit.module.reset and stays disabled when absent. The drive uses the sources described above, with qubit.module.x180 when present and qub_probe when absent. Unset slots or slots pointing to missing modules use the named template. Saved modules remain selectable. Pulse templates read qubit.freq, qubit.wiring.ch, resonator.freq, resonator.wiring.ch/ro_ch and resonator.time_of_fly + 0.05 us; missing required calibration remains Invalid.

## Typical writeback

Proposes qubit.ext.rabi_freq in MHz from the fit. With acquired run settings, copies the drive pulse into captured qubit.pulses.pi_amp and pi2_amp, replacing waveform.length with fitted pi and pi/2 lengths. Separately proposes qubit.module.x180/x90 pointing to those saved pulses, even when already bound to them, plus qubit.ext.pi_product = source gain * fitted pi length (a.u. * us). Every item is independently selectable. For data with IQ only, rabi_freq is the sole proposal. Pulse updates use the acquired run settings, with fitted lengths stored in the pulse copies.

## Recommended

Phase defaults to fixed 0/180 degrees; enable Fit phase offset when pulse shaping shifts the length oscillation. Analysis defaults to fitting a decay envelope on the oscillation; keep it on when the Rabi oscillation visibly damps over the sweep, turn it off for a pure undamped cosine fit. A length sweep spanning a few pi lengths (up to 4*pi_len when calibrated) captures a full oscillation; widen it if no full period is visible.

## Four starting situations

1. With an empty kickstart project, Open the Form and inspect its required blanks.
   Set a fixed drive frequency and gain, plus readout timing. A prior pi calibration is not required for this sweep.
   Nominal wiring and fallback values are candidates, not measured calibration.
2. With frequencies but no module slots, Reset to recapture those values.
   Fill the remaining pulse, timing and calibration errors before Run. Refresh keeps existing inputs.
3. With a readout slot, Reset and review the captured module and remaining errors separately.
   A slot does not prove that drive pulses, sweep bounds or GE calibration are valid.
4. Through the agent, call `time_rabi` after inspecting the same Form errors.
   points maps to length Expts; max_length_us sets the stop. frequency_mhz and gain override the fixed drive. decay and fit_phase change the existing analysis model.
   Common library/inline inputs select sources; overrides apply last. Null optional inputs retain Form values.
   Missing inputs return needs_parameters before acquisition. Reset may replace unsaved inputs.
