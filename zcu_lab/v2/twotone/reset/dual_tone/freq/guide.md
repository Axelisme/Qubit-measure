## Behavior

Dual-tone reset frequency map: a 2D sweep of the two reset-tone frequencies (pulse1 × pulse2 of the tested two-pulse reset), imaging the qubit response to locate the pair of sideband frequencies that jointly dump the excitation. Runs on real hardware in hard-sweep mode (both axes are QICK register sweeps).

## Expects (parameter context)

Seeds two 201-point axes at `$qubit.ext.reset_f1` and `$qubit.ext.reset_f2`, each +/- 50 MHz. Tested gains read `$qubit.ext.reset_gain1` and `$qubit.ext.reset_gain2.` Relax delay starts at 5*T1. Open and Reset evaluate starting expressions once into direct input using captured sources. Inline qubit pulses read `$qubit.freq` and `$qubit.wiring.ch.` Inline readout reads `$resonator.freq`, `$resonator.wiring.ch`, `$resonator.wiring.ro_ch` and `$resonator.ext.time_of_fly` + 0.05 us for trigger timing. Other missing or invalid required sources leave the affected fields Invalid; supply calibration or direct values before Run.

T1-derived defaults use the fixed Open/Reset capture. A missing/null T1 or absent capture uses 30 us, so the 5*T1 wait starts at 150 us. Stored invalid T1 leaves derived required fields empty while the Form opens. Alias/path/source failures propagate. Defaults remain editable, are not written back as calibration, and do not shorten automatically for active reset.

## Expects (ModuleLibrary)

Tested reset starts inline from the two_pulse_reset template. Optional init pulse uses `$qubit.module.x180` without fallback; an unset/dangling slot or unbound alias leaves it disabled and Valid. `$qubit.module.readout` selects a captured library link, otherwise a fresh inline readout template with a const waveform. Its required parameter sources must still be present. Optional `$qubit.module.reset` has no fallback: an unset/dangling slot or unbound alias leaves it disabled and Valid.

## Typical writeback

Proposes the fitted frequencies to `$qubit.ext.reset_f1` and `$qubit.ext.reset_f2` only. DualTonePower proposes the tested reset module after analysing gains; length analysis has no automatic writeback.

## Recommended

Keep each axis a tight span (a few MHz) around the predicted sideband; the 2D scan cost grows with both axes. Analysis denoises the map before peak finding; wavelet smoothing is the default and Gaussian remains available for comparison.
